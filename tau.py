import numpy as np
from scipy.optimize import linprog
import json
import matplotlib
from matplotlib import cm
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
DEFAULT_REV_CURRENT = [1.36, -2.04, -.231]  # reverse direction
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


def get_max_effort(thrusters: t.List[Thruster3D], objective: np.ndarray, constraints: np.ndarray,
                   max_current: int):
    # First Simplex run. Find the maximum thrust in the desired direction
    bounds = [thruster.max_thrusts for thruster in thrusters]

    right_of_equality = np.zeros(5)  # There are 5 constraints that must all be 0

    max_effort_result = linprog(c=-objective, A_ub=None, b_ub=None, A_eq=constraints, b_eq=right_of_equality,
                                bounds=bounds, method="highs")

    max_effort = -.999 * max_effort_result.fun  # some sort of precision/numerical error makes this bullshit necessary

    if max_effort < 0.00000001:
        # The thruster layout is incapable of producing effort in the target direction
        return 0.0

    # Second Simplex run. Find the minimum current that produces the same effort as the first result

    # Each thruster is split into reverse and forwards, so there are double the elements in the objective
    # Minimize the sum of the absolute value of each effort
    objective_mincurrent = np.concatenate((-np.ones(len(thrusters)), np.ones(len(thrusters))))

    # All 6 degrees of freedom are constrained
    thruster_constraints_mincurrent = np.row_stack((objective, constraints))
    # Each thruster is split in two, the constraints are the same for each half of a thruster
    left_of_equality_mincurrent = np.column_stack((thruster_constraints_mincurrent, thruster_constraints_mincurrent))

    bounds_mincurrent = [(DEFAULT_MAX_THRUSTS[0], 0.0) for _ in range(len(thrusters))] + \
                        [(0.0, DEFAULT_MAX_THRUSTS[1]) for _ in range(len(thrusters))]

    right_of_equality_mincurrent = [
        max_effort,  # x thrust constrained to previous maximum
        0,  # y thrust
        0,  # z thrust
        0,  # x torque
        0,  # y torque
        0,  # z torque
    ]

    min_current_result = linprog(c=objective_mincurrent, A_ub=None, b_ub=None, A_eq=left_of_equality_mincurrent,
                                 b_eq=right_of_equality_mincurrent, bounds=bounds_mincurrent, method="highs")

    min_current_duplicated_array = min_current_result.x

    min_current_true_array = []
    for i in range(0, len(thrusters)):
        min_current_true_array.append(min_current_duplicated_array[i] + min_current_duplicated_array[
            i + len(thrusters)])  # combine half-thrusters into full thrusters

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

    # solve quadratic, take the proper point, and clamp it to a maximum of 1.0
    effort_multiplier = min(1., max(np.roots(current_quadratic)))

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


def add_colorbar(plot, ax, color_index):
    color_range = color_index.max() - color_index.min()
    norm = matplotlib.colors.Normalize(vmin=color_index.min(), vmax=color_index.max())
    m = cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
    plot.colorbar(m, ticks=[
        color_index.min(),
        color_index.min() + color_range * 1 / 4,
        color_index.min() + color_range * 2 / 4,
        color_index.min() + color_range * 3 / 4,
        color_index.max()
    ], ax=ax, fraction=0.1, shrink=0.5)

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '=', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

# The main entry point of the program
# All the Click decorators define various options that can be passed in on the command line
@click.command()
@click.option("--thrusters", "-t", default="thrusters.json", help="file containing thruster specifications")
@click.option("--resolution", "-r",
              default=DEFAULT_RESOLUTION,
              help="resolution of the thrust calculation, runtime is O(n^2) with respect to this!"
              )
@click.option("--max-current", "-c", default=DEFAULT_MAX_CURRENT, help="maximum thruster current draw in amps")
def main(thrusters, resolution: int, max_current: int):
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

    # I have no idea what np.meshgrid does
    u, v = np.mgrid[0:2 * np.pi:resolution * 1j, 0:np.pi: resolution / 2 * 1j]
    thrust_mesh_x = np.empty(np.shape(u))
    thrust_mesh_y = np.empty(np.shape(u))
    thrust_mesh_z = np.empty(np.shape(u))
    thrust_color_index = np.empty(np.shape(u))
    torque_mesh_x = np.empty(np.shape(u))
    torque_mesh_y = np.empty(np.shape(u))
    torque_mesh_z = np.empty(np.shape(u))
    torque_color_index = np.empty(np.shape(u))

    # Iterate over each vertex and calculate the max thrust in that direction
    # Note: Should probably be its own function, then it can be optimized more (i.e. Numba)
    max_thrust = 0
    max_torque = 0
    
    k = 0
    printProgressBar(0, np.size(u), prefix = 'Progress:', suffix = 'Complete', length = 25)
    
    for i in range(np.shape(u)[0]):
        for j in range(np.shape(u)[1]):
            z = np.cos(u[i][j]) * np.sin(v[i][j])
            y = np.sin(u[i][j]) * np.sin(v[i][j])
            x = np.cos(v[i][j])
            transformed_orientations = rotate_to_vector(thruster_orientations, np.array([x, y, z]))
            transformed_torques = rotate_to_vector(thruster_torques, np.array([x, y, z]))

            thrust = get_max_effort(thrusters, transformed_orientations[0],
                                    np.row_stack((transformed_orientations[1:], thruster_torques)), max_current)
            torque = get_max_effort(thrusters, transformed_torques[0],
                                    np.row_stack((transformed_torques[1:], thruster_orientations)), max_current)

            thrust_mesh_x[i][j] = x * thrust
            thrust_mesh_y[i][j] = y * thrust
            thrust_mesh_z[i][j] = z * thrust
            thrust_color_index[i][j] = thrust

            torque_mesh_x[i][j] = x * torque
            torque_mesh_y[i][j] = y * torque
            torque_mesh_z[i][j] = z * torque
            torque_color_index[i][j] = torque

            max_thrust = max(max_thrust, thrust)
            max_torque = max(max_torque, torque)
            
            k = k + 1
            printProgressBar(k, np.size(u), prefix = 'Progress:', suffix = 'Complete', length = 25)
            
    # Start plotting results
    matplotlib.use('TkAgg')
    fig = plt.figure(figsize=(12, 6))  # Window size, in inches for some reason

    # Set up plot: 3d orthographic plot with ROV axis orientation
    ax_thrust = fig.add_subplot(121, projection='3d', proj_type='ortho')
    ax_torque = fig.add_subplot(122, projection='3d', proj_type='ortho')

    setup_subplot(ax_thrust, thrusters, np.ceil(max_thrust))
    ax_thrust.set_xlabel('X (Surge)')
    ax_thrust.set_ylabel('Y (Sway)')
    ax_thrust.set_zlabel('Z (Heave)')
    ax_thrust.title.set_text('Thrust')

    setup_subplot(ax_torque, thrusters, np.ceil(max_torque))
    ax_torque.set_xlabel('X (Roll)')
    ax_torque.set_ylabel('Y (Pitch)')
    ax_torque.set_zlabel('Z (Yaw)')
    ax_torque.title.set_text('Torque')

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

    # Adjust each color so that the min and max values correspond to the min and max colors
    thrust_color_index_modified = (thrust_color_index - thrust_color_index.min()) / \
                                  (thrust_color_index.max() - thrust_color_index.min())
    torque_color_index_modified = (torque_color_index - torque_color_index.min()) / \
                                  (torque_color_index.max() - torque_color_index.min())

    # Plot the surfaces on their respective subplots
    ax_thrust.plot_surface(
        thrust_mesh_x, thrust_mesh_y, thrust_mesh_z,
        alpha=0.6, facecolors=cm.jet(thrust_color_index_modified), edgecolors='w', linewidth=0
    )
    ax_torque.plot_surface(
        torque_mesh_x, torque_mesh_y, torque_mesh_z,
        alpha=0.6, facecolors=cm.jet(torque_color_index_modified), edgecolors='w', linewidth=0
    )

    # Create a legend mapping the colors of the thrust plot to thrust values
    add_colorbar(plt, ax_thrust, thrust_color_index)
    add_colorbar(plt, ax_torque, torque_color_index)

    # Show plot
    plt.show()


if __name__ == "__main__":  # Only run the main function the program is being run directly, not imported
    main()  # Click autofills the parameters to this based on the program's command-line arguments
