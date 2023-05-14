import numpy as np
import scipy
from scipy.optimize import linprog
import qpsolvers
import json
import matplotlib
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.pyplot as plt
import typing as t
import click

from time import sleep

DEFAULT_RESOLUTION = 100  # Runtime is O(n^2) with respect to resolution!
DEFAULT_MAX_THRUSTS = [-2.9, 3.71]  # Lifted from the BlueRobotics public performance data (kgf)
# coefficients of the quadratic approximating current draw as a function of thrust in the forward direction in the form:
# ax^2 + bx + c
# Both regressions are in terms of the same variable, thrust, which is negative in the reverse direction
DEFAULT_FWD_CURRENT = [.741, 1.89, -.278]
DEFAULT_REV_CURRENT = [1.36, -2.04, -.231]
DEFAULT_MAX_CURRENT = 22


class Thruster3D:
    def __init__(self, x, y, z, theta, phi, max_thrusts, fwd_current, rev_current):
        self.pos = np.array([x, y, z])
        self.max_thrusts = max_thrusts
        self.fwd_current = fwd_current
        self.rev_current = rev_current

        # Calculate the unit vector in the direction specified by theta and phi
        theta = np.radians(theta)
        phi = np.radians(phi)
        self.orientation = np.array([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi)
        ])

    def torque(self):
        return np.cross(self.pos, self.orientation)


def get_column_span(mat: np.ndarray) -> np.ndarray:
    """
    Find the column span of a matrix (remove columns which to not increase the number of dimensions the columns span)

    @param mat:The input matrix
    @return: A full rank matrix constructed from the column vectors of the input
    """
    upper_triangular = scipy.linalg.lu(mat)[2]

    output_vectors = []
    for row in upper_triangular:
        for i, value in enumerate(row):
            if abs(value) > 1e-10:
                output_vectors.append(mat[..., i])
                break

    return np.array(output_vectors).transpose()


# rref function by joni on Stack Overflow:
# https://stackoverflow.com/a/66412719
# This function is licenced under CC BY-SA 4.0
# https://creativecommons.org/licenses/by-sa/4.0/
def rref(A, tol=1.0e-12):
    m, n = A.shape
    i, j = 0, 0
    jb = []

    while i < m and j < n:
        # Find value and index of largest element in the remainder of column j
        k = np.argmax(np.abs(A[i:m, j])) + i
        p = np.abs(A[k, j])
        if p <= tol:
            # The column is negligible, zero it out
            A[i:m, j] = 0.0
            j += 1
        else:
            # Remember the column index
            jb.append(j)
            if i != k:
                # Swap the i-th and k-th rows
                A[[i, k], j:n] = A[[k, i], j:n]
            # Divide the pivot row i by the pivot element A[i, j]
            A[i, j:n] = A[i, j:n] / A[i, j]
            # Subtract multiples of the pivot row from all the other rows
            for k in range(m):
                if k != i:
                    A[k, j:n] -= A[k, j] * A[i, j:n]
            i += 1
            j += 1
    # Finished
    return A, jb


def rotate_to_vector(vectors: np.ndarray, target_dir: np.ndarray) -> np.ndarray:
    """
    Rotate a group of vectors so that the a specified vector is along the +x axis
    :param vectors: A 2d numpy array in which each column is a 3d vector
    :param target_dir: A 3d vector in the target direction
    :return: A np array with the same size as vectors, with the same rotation applied to each column
    """
    target_dir = target_dir / np.linalg.norm(target_dir)  # Make target_dir a unit vector

    new_bases = np.empty((3, 3))  # Create an empty 3x3 change of basis matrix
    new_bases[..., 0] = target_dir  # The first basis is our target direction

    if not (target_dir[1] == 0 and target_dir[2] == 0):  # Make sure the cross product computed below isn't 0
        second_basis = np.cross(target_dir, np.array([1, 0, 0]))  # Choose a second basis perpendicular the first
    else:
        second_basis = np.cross(target_dir, np.array([0, 1, 0]))
    second_basis /= np.linalg.norm(second_basis)  # Make the second basis a unit vector

    new_bases[..., 1] = second_basis
    third_basis = np.cross(target_dir, second_basis)  # Calculate a third basis perpendicular the first two
    third_basis /= np.linalg.norm(third_basis)  # Make the third basis a unit vector
    new_bases[..., 2] = third_basis

    # Invert the matrix. The original matrix maps (1, 0, 0) onto the target direction. We want a matrix
    # that maps the target direction onto (1, 0, 0).
    inverse_transform = np.linalg.inv(new_bases)

    # Calculate the transformation with matrix_vector multiplication
    transformed_orientations = inverse_transform.dot(vectors)

    return transformed_orientations


def get_max_effort(thrusters: t.List[Thruster3D], objective: np.ndarray, constraints: t.Optional[np.ndarray],
                   max_current: float):
    thruster_count = len(thrusters)

    # First Simplex run. Find the maximum thrust in the desired direction
    bounds = [thruster.max_thrusts for thruster in thrusters]

    right_of_equality = np.zeros(constraints.shape[0]) if constraints is not None else None  # All constraints must be 0

    max_effort_result = linprog(c=-objective, A_ub=None, b_ub=None, A_eq=constraints, b_eq=right_of_equality,
                                bounds=bounds, method="highs")

    max_effort = -.99999 * max_effort_result.fun  # some sort of precision/numerical error makes this bullshit necessary

    if max_effort < 0.00000001:
        # The thruster layout is incapable of producing effort in the target direction
        return 0.0

    # Find the minimum current that produces the same effort as the first result
    # Each thruster is split into reverse and forwards, so there are double the elements in the objective

    # The objective function (total current as a function of thruster forces) is quadratic
    x_squared_coefficients = np.zeros((thruster_count * 2, thruster_count * 2))  # Holds the coefficients of x^2
    x_coefficients = np.empty(thruster_count * 2)  # Holds the coefficients of x

    # The reverse half thrusters are indexed 0 to numb_thrusters - 1
    # Forward half thrusters are indexed numb_thrusters to 2 * numb_thrusters - 1
    for i, thruster in enumerate(thrusters):
        x_squared_coefficients[i][i] = thruster.rev_current[0]
        x_squared_coefficients[i + thruster_count][i + thruster_count] = thruster.fwd_current[0]

        x_coefficients[i] = thruster.rev_current[1]
        x_coefficients[i + thruster_count] = thruster.fwd_current[1]

    # All 6 degrees of freedom are constrained
    thruster_constraints_mincurrent = np.row_stack((objective, constraints)) if constraints is not None else \
        np.array(objective)
    # Each thruster is split in two, the constraints are the same for each half of a thruster
    left_of_equality_mincurrent = np.column_stack((thruster_constraints_mincurrent, thruster_constraints_mincurrent))

    lower_bounds = np.array([thruster.max_thrusts[0] for thruster in thrusters] + [0.0 for _ in thrusters])
    upper_bounds = np.array([0.0 for _ in thrusters] + [thruster.max_thrusts[1] for thruster in thrusters])

    # Extra constraint for the original objective
    right_of_equality_mincurrent = np.zeros((0 if constraints is None else constraints.shape[0]) + 1)
    right_of_equality_mincurrent[0] = max_effort

    min_current_result = qpsolvers.solve_qp(
        P=2 * x_squared_coefficients,  # The solver minimizes 1/2 * Px^2 + qx, we need to cancel out the 1/2
        q=x_coefficients,
        A=left_of_equality_mincurrent,
        b=right_of_equality_mincurrent,
        lb=lower_bounds,
        ub=upper_bounds,
        solver="quadprog"
    )
    
    #sometimes min_current_result doesn't solve, I have no idea why. This generates ugly, erroneous graphs, but at least the program doesn't crash. 
    if(min_current_result is None):
        min_current_result = np.zeros(thruster_count * 2)

    # combine half-thrusters into full thrusters
    min_current_true_array = []
    for i in range(thruster_count):
        min_current_true_array.append(min_current_result[i] + min_current_result[i + thruster_count])

    current_quadratic = [0] * 3

    for i, thruster in enumerate(thrusters):
        thrust = min_current_true_array[i]
        if thrust >= 0:  # use the forward thrust coefficients
            current_quadratic[0] += thruster.fwd_current[0] * thrust ** 2  # a * t^2
            current_quadratic[1] += thruster.fwd_current[1] * thrust  # b * t
            current_quadratic[2] += thruster.fwd_current[2]  # c
        else:  # use the reverse thrust coefficients
            current_quadratic[0] += thruster.rev_current[0] * thrust ** 2
            current_quadratic[1] += thruster.rev_current[1] * thrust
            current_quadratic[2] += thruster.rev_current[2]

    current_quadratic[2] -= max_current  # ax^2 + bx + c = I -> ax^2 + bx + (c-I) = 0

    #sometimes quadroots is unsolvable, I have no idea why. This generates ugly, erroneous graphs, but at least the program doesn't crash. 
    quadroots = np.roots(current_quadratic)
    if(len(quadroots) == 0):
        quadroots = [0, 0]
    
    # solve quadratic, take the proper point, and clamp it to a maximum of 1.0
    effort_multiplier = min(1., max(quadroots))

    return max_effort * effort_multiplier


def setup_subplot(subplot, thrusters, axes_bounds):
    subplot.set_box_aspect((1, 1, 1))
    subplot.view_init(elev=30, azim=-150)
    subplot.set_xlim((axes_bounds, -axes_bounds))  # Invert x axis
    subplot.set_ylim((-axes_bounds, axes_bounds))
    subplot.set_zlim((axes_bounds, -axes_bounds))  # Invert z axis
    # Draw some "axes" so it's clear where (0, 0, 0) is
    subplot.plot((-axes_bounds, axes_bounds), (0, 0), (0, 0), c="black")
    subplot.plot((0, 0), (-axes_bounds, axes_bounds), (0, 0), c="black")
    subplot.plot((0, 0), (0, 0), (-axes_bounds, axes_bounds), c="black")
    # Plot the locations and orientations of the thrusters
    thrusterloc_x = [2 * thruster.pos[0] for thruster in thrusters]
    thrusterloc_y = [2 * thruster.pos[1] for thruster in thrusters]
    thrusterloc_z = [2 * thruster.pos[2] for thruster in thrusters]
    thrusterdir_x = [2 * thruster.orientation[0] for thruster in thrusters]
    thrusterdir_y = [2 * thruster.orientation[1] for thruster in thrusters]
    thrusterdir_z = [2 * thruster.orientation[2] for thruster in thrusters]
    subplot.quiver(thrusterloc_x, thrusterloc_y, thrusterloc_z, thrusterdir_x, thrusterdir_y, thrusterdir_z,
                   color="black")


def add_colorbar(plot, ax, color_index, norm=None, cmap=plt.cm.turbo):
    norm = norm or matplotlib.colors.Normalize(vmin=color_index.min(), vmax=color_index.max())
    color_range = norm.vmax - norm.vmin

    m = cm.ScalarMappable(cmap=cmap, norm=norm)
    plot.colorbar(m, ticks=[
        norm.vmin,
        norm.vmin + color_range * 1 / 4,
        norm.vmin + color_range * 2 / 4,
        norm.vmin + color_range * 3 / 4,
        norm.vmax
    ], ax=ax, fraction=0.1, shrink=0.5)


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='=', print_end="\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    # Print New Line on Complete
    if iteration == total:
        print()

def plot_effort_surface(plot, ax, thrusters: t.List[Thruster3D], effort_vectors: np.ndarray,
                        extra_constraints: np.ndarray, resolution: int, max_current: float):
    # Determine whether the the set of possible efforts is a solid, surface, or line

    # Solve the constraints matrix
    constraints_rref = rref(np.copy(extra_constraints), tol=1e-10)[0]

    # Find the pivot columns
    pivot_columns = []
    for row in constraints_rref:
        for j, val in enumerate(row):
            if abs(val) > 1e-10:
                pivot_columns.append(j)
                break

    # Find the vectors that span the solution set to extra_constraints * x = 0
    thruster_value_bases = []
    for i in range(constraints_rref.shape[1]):
        if i not in pivot_columns:
            new_basis = np.empty(constraints_rref.shape[1])
            for j in range(constraints_rref.shape[1]):
                if j in pivot_columns:
                    new_basis[j] = -constraints_rref[pivot_columns.index(j)][i]
                else:
                    new_basis[j] = int(i == j)
            thruster_value_bases.append(new_basis)

    thruster_bases_matrix = np.matrix.round(np.array(thruster_value_bases).transpose(), decimals=10)

    if thruster_bases_matrix.shape[0] == 0:
        # The thrusters cannot produce effort in any direction under the constraints
        return

    # Find the span of the effort vectors under the constraints
    effort_bases_matrix = effort_vectors.dot(thruster_bases_matrix)
    effort_span = np.matrix.round(get_column_span(effort_bases_matrix), decimals=10)

    if effort_span.shape[1] == 3:
        # The output space is a 3d solid
        # I have no idea what np.meshgrid does
        u, v = np.mgrid[0:2 * np.pi:resolution * 1j, 0:np.pi: resolution / 2 * 1j]
        mesh_x = np.empty(np.shape(u))
        mesh_y = np.empty(np.shape(u))
        mesh_z = np.empty(np.shape(u))
        color_index = np.empty(np.shape(u))

        k = 0
        print_progress_bar(0, np.size(u), prefix='Progress:', suffix='Complete', length=25)

        # Iterate over each vertex and calculate the max effort in that direction
        max_effort = 0
        for i in range(np.shape(u)[0]):
            for j in range(np.shape(u)[1]):
                z = np.cos(u[i][j]) * np.sin(v[i][j])
                y = np.sin(u[i][j]) * np.sin(v[i][j])
                x = np.cos(v[i][j])

                transformed_effort_vectors = rotate_to_vector(effort_vectors, np.array([x, y, z]))

                effort = get_max_effort(thrusters, transformed_effort_vectors[0],
                                        np.row_stack((transformed_effort_vectors[1:], extra_constraints)), max_current)

                mesh_x[i][j] = x * effort
                mesh_y[i][j] = y * effort
                mesh_z[i][j] = z * effort
                color_index[i][j] = effort

                max_effort = max(max_effort, effort)

                k = k + 1
                print_progress_bar(k, np.size(u), prefix='Progress:', suffix='Complete', length=25)

        # Adjust each color so that the min and max values correspond to the min and max colors
        color_index_modified = (color_index - color_index.min()) / (color_index.max() - color_index.min())

        setup_subplot(ax, thrusters, np.ceil(max_effort))

        ax.plot_surface(
            mesh_x, mesh_y, mesh_z, alpha=0.75, facecolors=cm.turbo(color_index_modified), edgecolors='w', linewidth=0
        )

        # Create a legend mapping the colors of the each plot to its values
        add_colorbar(plot, ax, color_index)

    elif effort_span.shape[1] == 2:
        # The output space is confined to a plane

        # Switch to equivalent perpendicular bases
        normal = np.cross(effort_span[..., 0], effort_span[..., 1])
        if normal[1] != 0 or normal[2] != 0:
            first_basis = np.cross(normal, np.array([1, 0, 0]))
        else:
            first_basis = np.cross(normal, np.array([0, 1, 0]))
        effort_span[..., 0] = first_basis / np.linalg.norm(first_basis)  # Convert to unit vector

        second_basis = np.cross(normal, first_basis)
        effort_span[..., 1] = second_basis / np.linalg.norm(second_basis)

        effort_inv_transform = np.linalg.pinv(effort_span)
        transformed_efforts = effort_inv_transform.dot(effort_vectors)

        theta_space = np.linspace(0, np.pi * 2, num=resolution * 2)
        curve = np.empty((2, theta_space.size))
        color_index = np.empty(theta_space.size)

        max_effort = 0
        for i, theta in enumerate(theta_space):
            u = np.cos(theta)
            v = np.sin(theta)

            rotation_mat = np.array([[u, -v], [v, u]])
            rotated_efforts = rotation_mat.dot(transformed_efforts)

            effort = get_max_effort(thrusters, rotated_efforts[0], np.array([rotated_efforts[1]]), max_current)

            curve[0, i] = u * effort
            curve[1, i] = v * effort
            color_index[i] = effort

            max_effort = max(max_effort, effort)

        # Transform the 2d output space back into 3d
        curve_3d = effort_span.dot(curve)

        setup_subplot(ax, thrusters, np.ceil(max_effort))

        points = curve_3d.T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = Line3DCollection(segments, cmap='turbo', linewidths=2.5)
        lc.set_array(color_index)

        ax.add_collection(lc)

        add_colorbar(plt, ax, color_index)

    elif effort_span.shape[1] == 1:
        # The output space is confined to a line
        effort_span[..., 0] = effort_span[..., 0] / np.linalg.norm(effort_span[..., 0])  # Normalize the basis vector

        effort_inv_transform = np.linalg.pinv(effort_span)
        transformed_efforts = effort_inv_transform.dot(effort_vectors)

        pos_effort = get_max_effort(thrusters, transformed_efforts, None, max_current)
        neg_effort = get_max_effort(thrusters, -transformed_efforts, None, max_current)

        efforts = np.zeros((3, 2))
        efforts[..., 0] = effort_span.transpose() * pos_effort
        efforts[..., 1] = effort_span.transpose() * -neg_effort

        setup_subplot(ax, thrusters, max(np.linalg.norm(efforts[..., 0]), np.linalg.norm(efforts[..., 1])))

        average_effort = (pos_effort + neg_effort) / 2
        norm = matplotlib.colors.Normalize(average_effort / 2, average_effort * 3 / 2)

        custom_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("rbu_cmap", ["blue", "gray", "red"])

        a1 = ax.quiver(0, 0, 0, efforts[0][0], efforts[1][0], efforts[2][0], cmap=custom_cmap, norm=norm)
        a1.set_array(np.array([pos_effort]))

        a2 = ax.quiver(0, 0, 0, efforts[0][1], efforts[1][1], efforts[2][1], cmap=custom_cmap, norm=norm)
        a2.set_array(np.array([neg_effort]))

        add_colorbar(plt, ax, None, norm=norm, cmap=custom_cmap)

    else:
        raise ValueError("The span of the effort vectors had an unexpected dimension")


# The main entry point of the program
# All the Click decorators define various options that can be passed in on the command line
@click.command()
@click.option("--thrusters", "-t", default="thrusters.json", help="file containing thruster specifications")
@click.option("--resolution", "-r",
              default=DEFAULT_RESOLUTION,
              help="resolution of the thrust calculation, runtime is O(n^2) with respect to this!"
              )
@click.option("--max-current", "-c", default=DEFAULT_MAX_CURRENT, help="maximum thruster current draw in amps")
def main(thrusters, resolution: int, max_current: float):
    # This doc comment becomes the description text for the --help menu
    """
    tau - the thruster arrangement utility
    """

    # Read the thruster transforms input JSON file
    # Wrap this in a try-except FileNotFoundError block to print a nicer error message
    with open(thrusters) as f:  # `with` blocks allow you to open files safely without risking corrupting them on crash
        thrusters_raw = json.load(f)

    # Convert loaded JSON data into Thruster3D objects
    thrusters: t.List[Thruster3D] = [
        Thruster3D(
            thruster_raw['x'],
            thruster_raw['y'],
            thruster_raw['z'],
            thruster_raw['theta'],
            thruster_raw['phi'],
            # Optional thruster parameters: dict.get is used to provide a default value if the key doesn't exist
            thruster_raw.get("max_thrusts", DEFAULT_MAX_THRUSTS),
            thruster_raw.get("fwd_current", DEFAULT_FWD_CURRENT),
            thruster_raw.get("rev_current", DEFAULT_REV_CURRENT)
        )
        for thruster_raw in thrusters_raw
    ]

    # Format the orientation and torque of the thrusters to be used as constraints
    thruster_orientations = np.array([thruster.orientation for thruster in thrusters]).transpose()
    thruster_torques = np.array([thruster.torque() for thruster in thrusters]).transpose()

    # Set up matplotlib window
    matplotlib.use('TkAgg')
    fig = plt.figure(num="TAU", figsize=(12, 6))  # Window size, in inches for some reason

    # Set up plot: 3d orthographic plot with ROV axis orientation
    ax_thrust = fig.add_subplot(121, projection='3d', proj_type='ortho')
    ax_torque = fig.add_subplot(122, projection='3d', proj_type='ortho')

    # Plot thrust surface
    print("Plotting thrust...")
    plot_effort_surface(plt, ax_thrust, thrusters, thruster_orientations, thruster_torques, resolution, max_current)
    # Plot torque surface
    print("Plotting torque...")
    plot_effort_surface(plt, ax_torque, thrusters, thruster_torques, thruster_orientations, resolution, max_current)

    ax_thrust.title.set_text('Thrust')
    ax_thrust.set_xlabel('X (Surge)')
    ax_thrust.set_ylabel('Y (Sway)')
    ax_thrust.set_zlabel('Z (Heave)')

    ax_torque.title.set_text('Torque')
    ax_torque.set_xlabel('X (Roll)')
    ax_torque.set_ylabel('Y (Pitch)')
    ax_torque.set_zlabel('Z (Yaw)')

    # Synchronize the rotation and zoom of both subplots
    def on_plot_move(event):
        if event.inaxes is None:
            return
        ax = event.inaxes
        ax2 = ax_thrust if event.inaxes == ax_torque else ax_torque
        try:
            button_pressed = ax.button_pressed
        except AttributeError:
            return
        if button_pressed in ax._rotate_btn:
            ax2.view_init(elev=ax.elev, azim=ax.azim)
        elif button_pressed in ax._zoom_btn:
            ax2.set_xlim3d(ax.get_xlim3d())
            ax2.set_ylim3d(ax.get_ylim3d())
            ax2.set_zlim3d(ax.get_zlim3d())
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_plot_move)

    # Show plot
    plt.show()


if __name__ == "__main__":  # Only run the main function the program is being run directly, not imported
    main()  # Click autofills the parameters to this based on the program's command-line arguments
