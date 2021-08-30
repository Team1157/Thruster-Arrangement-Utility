import numpy as np
from scipy.optimize import linprog
import json
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import typing as t
import click

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


def transform_orientations(thrusters: t.List[Thruster3D], target_dir: np.ndarray):
    """
    Calculate the maximum force achievable in (exclusively) the given direction by the given set of constraints
    :param thrusters: A list of Thruster3D objects representing the available thrusters
    :param target_dir: A 3d vector in the target direction
    :return: The maximum thrust force in kgf
    """
    target_dir = target_dir / np.linalg.norm(target_dir)  # Make target_dir a unit vector

    orientations = np.empty((3, len(thrusters)))  # Create an empty 2d array to hold the orientation of each thruster
    for i in range(len(thrusters)):
        thruster = thrusters[i]
        orientations[..., i] = thruster.orientation

    new_bases = np.empty((3, 3))  # Create an empty 3x3 change of basis matrix
    new_bases[..., 0] = target_dir  # The first basis is our target direction

    if not (target_dir[1] == 0 and target_dir[2] == 0):  # Make sure the cross product computed below isn't 0
        second_basis = np.cross(target_dir, np.array([1, 0, 0]))  # Choose a second basis parallel the first
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
    transformed_orientations = inverse_transform.dot(orientations)

    return transformed_orientations


def get_max_thrust(thrusters, transformed_orientations, torques: np.ndarray, max_current: int):
    thruster_count = len(torques)

    # First Simplex run. Find the maximum thrust in the desired direction
    objective = -transformed_orientations[0]  # Get the max thrust in the transformed x axis (the target direction)
    # All other thrusts and torques must be 0
    torques_transpose = torques.transpose()
    left_of_equality = np.row_stack((torques_transpose, transformed_orientations[1:]))

    bounds = [thruster.max_thrusts for thruster in thrusters]

    right_of_equality = np.zeros(5)  # There are 5 constraints that must all be 0

    max_thrust_result = linprog(c=objective, A_ub=None, b_ub=None, A_eq=left_of_equality, b_eq=right_of_equality,
                                bounds=bounds, method="highs")

    max_thrust = -.999 * max_thrust_result.fun  # some sort of precision/numerical error makes this bullshit necessary

    # Second Simplex run. Find the minimum current that produces the same thrust as the first result

    # Each thruster is split into reverse and forwards, so there are double the elements in the objective
    # Minimize the sum of the absolute value of each thrust
    objective_mincurrent = np.concatenate((-np.ones(thruster_count), np.ones(thruster_count)))

    # All 6 degrees of freedom are constrained
    thruster_constraints_mincurrent = np.row_stack((transformed_orientations, torques_transpose))
    # Each thruster is split in two, the constraints are the same for each half of a thruster
    left_of_equality_mincurrent = np.column_stack((thruster_constraints_mincurrent, thruster_constraints_mincurrent))

    bounds_mincurrent = [(DEFAULT_MAX_THRUSTS[0], 0.0) for _ in range(thruster_count)] + \
                        [(0.0, DEFAULT_MAX_THRUSTS[1]) for _ in range(thruster_count)]

    right_of_equality_mincurrent = [
        max_thrust,  # x thrust constrained to previous maximum
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
    for i in range(0, thruster_count):
        min_current_true_array.append(min_current_duplicated_array[i] + min_current_duplicated_array[
            i + thruster_count])  # combine half-thrusters into full thrusters

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
    thrust_multiplier = min(1., max(np.roots(current_quadratic)))

    return max_thrust * thrust_multiplier


#####################################
# Yaw, pitch, roll code
#####################################
def calc_max_yaw_pitch_roll(thrusters, thruster_torques):
    torque_x = []
    torque_y = []
    torque_z = []
    orientation = []
    constraint3 = []
    constraint4 = []
    constraint5 = []
    right_equalities = [0, 0, 0, 0, 0]
    bounds = []

    for i in range(len(thruster_torques)):
        torque_x.append(thruster_torques[i][0])
        torque_y.append(thruster_torques[i][1])
        torque_z.append(thruster_torques[i][2])
        thruster = thrusters[i]
        orientation.append(thruster.orientation)
        constraint3.append(thruster.orientation[0])
        constraint4.append(thruster.orientation[1])
        constraint5.append(thruster.orientation[2])
        bounds.append(thrusters[i].max_thrusts)

    for i in range(3):
        torques = [torque_x, torque_y, torque_z]
        objective = [torques[i]]

        torques.pop(i)
        constraint1 = torques[0]
        constraint2 = torques[1]

        left_equalities = [constraint1, constraint2, constraint3, constraint4, constraint5]

        res = linprog(c=objective, A_ub=None, b_ub=None, A_eq=left_equalities, b_eq=right_equalities,
                      bounds=bounds, method="highs")

        print(res)


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

    # Calculate the torque constrains which will apply to every iteration
    thruster_torques = np.array([thruster.torque() for thruster in thrusters])

    # I have no idea what np.meshgrid does
    u, v = np.mgrid[0:2 * np.pi:resolution * 1j, 0:np.pi: resolution / 2 * 1j]
    np.empty(np.shape(u))
    mesh_x = np.empty(np.shape(u))
    mesh_y = np.empty(np.shape(u))
    mesh_z = np.empty(np.shape(u))
    color_index = np.empty(np.shape(u))

    # Iterate over each vertex and calculate the max thrust in that direction
    # Note: Should probably be its own function, then it can be optimized more (i.e. Numba)
    max_rho = 0
    for i in range(np.shape(u)[0]):
        for j in range(np.shape(u)[1]):
            z = np.cos(u[i][j]) * np.sin(v[i][j])
            y = np.sin(u[i][j]) * np.sin(v[i][j])
            x = np.cos(v[i][j])
            transformed_orientations = transform_orientations(thrusters, np.array([x, y, z]))
            rho = get_max_thrust(thrusters, transformed_orientations, thruster_torques, max_current)
            mesh_x[i][j] = x * rho
            mesh_y[i][j] = y * rho
            mesh_z[i][j] = z * rho
            color_index[i][j] = rho
            max_rho = max(max_rho, rho)

    max_rho = np.ceil(max_rho)

    norm = matplotlib.colors.Normalize(vmin=color_index.min(), vmax=color_index.max())

    # Start plotting results
    matplotlib.use('TkAgg')
    fig = plt.figure()

    # Set up plot: 3d orthographic plot with ROV axis orientation
    ax = fig.add_subplot(111, projection='3d', proj_type='ortho')

    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=30, azim=-150)

    ax.set_xlim((max_rho, -max_rho))  # Invert x axis
    ax.set_ylim((-max_rho, max_rho))
    ax.set_zlim((max_rho, -max_rho))  # Invert z axis

    ax.set_xlabel('X (Surge)')
    ax.set_ylabel('Y (Sway)')
    ax.set_zlabel('Z (Heave)')

    # Draw some "axes" so it's clear where (0, 0, 0) is
    ax.plot((-max_rho, max_rho), (0, 0), (0, 0), c="black")
    ax.plot((0, 0), (-max_rho, max_rho), (0, 0), c="black")
    ax.plot((0, 0), (0, 0), (-max_rho, max_rho), c="black")

    # Plot the locations and orientations of the thrusters
    # NOTE: Consider merging this all into the function call below to avoid creating all these unwieldy variable names?
    thrusterloc_x = [2 * thruster.pos[0] for thruster in thrusters]
    thrusterloc_y = [2 * thruster.pos[1] for thruster in thrusters]
    thrusterloc_z = [2 * thruster.pos[2] for thruster in thrusters]

    thrusterdir_x = [2 * thruster.orientation[0] for thruster in thrusters]
    thrusterdir_y = [2 * thruster.orientation[1] for thruster in thrusters]
    thrusterdir_z = [2 * thruster.orientation[2] for thruster in thrusters]

    ax.quiver(thrusterloc_x, thrusterloc_y, thrusterloc_z, thrusterdir_x, thrusterdir_y, thrusterdir_z, color="black")

    # Plot the zero-torque maximum thrust in each direction
    color_index_modified = (color_index - color_index.min()) / (color_index.max() - color_index.min())
    ax.plot_surface(
        mesh_x, mesh_y, mesh_z,
        alpha=0.6, facecolors=cm.jet(color_index_modified), edgecolors='w', linewidth=0
    )

    # Create a legend mapping the colors of the thrust plot to thrust values
    color_range = color_index.max() - color_index.min()
    m = cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
    plt.colorbar(m, ticks=[
        color_index.min(),
        color_index.min() + color_range / 4,
        color_index.min() + color_range / 2,
        color_index.min() + 3 * color_range / 4,
        color_index.max()
    ])

    # Show plot
    plt.show()

    # Print max yaw, pitch, and roll
    calc_max_yaw_pitch_roll(thrusters, thruster_torques)


if __name__ == "__main__":  # Only run the main function the program is being run directly, not imported
    main()  # Click autofills the parameters to this based on the program's command-line arguments
