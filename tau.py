import math
import numpy as np
from scipy.optimize import linprog
import json
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import typing as t


RESOLUTION = 100  # Runtime is O(n^2) with respect to resolution!
MAX_THRUSTER_FORCE = [-2.9, 3.71]  # Lifted from the BlueRobotics public performance data (kgf)


class Thruster3D:
    def __init__(self, x, y, z, theta, phi):
        self.pos = np.array([x, y, z])

        # Calculate the unit vector in the direction specified by theta and phi
        theta = math.radians(theta)
        phi = math.radians(phi)
        self.orientation = np.array([
            math.sin(phi) * math.cos(theta),
            math.sin(phi) * math.sin(theta),
            math.cos(phi)
        ])
    def torque(self):
        return np.cross(self.pos, self.orientation)

def get_max_thrust(thrusters: t.List[Thruster3D], target_dir: np.ndarray, t_constraints: np.ndarray) -> float:
    """
    Calculate the maximum force achievable in (exclusively) the given direction by the given set of constraints
    :param thrusters: A list of Thruster3D objects representing the available thrusters
    :param target_dir: A 3d vector in the target direction
    :param t_constraints: Any additional constraints (such as torque constraints which are the same for every iteration)
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
    transformed_orientations = inverse_transform.dot(orientations).transpose()

    #START OF SIMPLEX CODE
    objective = []
    left_of_equality = []
    
    thrusts_y = []
    thrusts_z = []
    
    torques_x = []
    torques_y = []
    torques_z = []
    
    bounds = []
    
    for i, orientation in enumerate(transformed_orientations, start = 0):
        objective.append(orientation[0])
        
        thrusts_y.append(orientation[1])
        thrusts_z.append(orientation[2])
        
        torques_x.append(t_constraints[i][0])
        torques_y.append(t_constraints[i][1])
        torques_z.append(t_constraints[i][2])
        
        bounds.append(MAX_THRUSTER_FORCE)
        
    left_of_equality.append(thrusts_y)
    left_of_equality.append(thrusts_z)
    
    left_of_equality.append(torques_x)
    left_of_equality.append(torques_y)
    left_of_equality.append(torques_z)
    
    right_of_equality = [
        0, #y thrust
        0, #z thrust
        0, #x torque
        0, #y torque
        0, #z torque
    ]
    
    optimized_result = linprog(c=objective, A_ub = None, b_ub = None, A_eq = left_of_equality, b_eq = right_of_equality, bounds=bounds, method="revised simplex")

    return optimized_result.fun

# Load the thruster data from a json file
with open("thrusters.json") as input_file:
    input_transforms = json.loads(input_file.read())

thrusters = []
for transform in input_transforms:
    thrusters.append(Thruster3D(transform['x'], transform['y'], transform['z'], transform['theta'], transform['phi']))

# Calculate the torque constrains which will apply to every iteration
torque_constraints = []

for thruster in thrusters:
    torque_constraints.append(thruster.torque())

#  I have no idea what np.meshgrid does
u, v = np.mgrid[0:2*np.pi:RESOLUTION * 1j, 0:np.pi: RESOLUTION / 2 * 1j]
np.empty(np.shape(u))
mesh_x = np.empty(np.shape(u))
mesh_y = np.empty(np.shape(u))
mesh_z = np.empty(np.shape(u))

# Iterate over each vertex and calculate the max thrust in that direction
max_rho = 0
for i in range(np.shape(u)[0]):
    for j in range(np.shape(u)[1]):
        x = math.cos(u[i][j]) * math.sin(v[i][j])
        y = math.sin(u[i][j]) * math.sin(v[i][j])
        z = math.cos(v[i][j])
        rho = -get_max_thrust(thrusters, np.array([x, y, z]), torque_constraints)
        mesh_x[i][j] = x * rho
        mesh_y[i][j] = y * rho
        mesh_z[i][j] = z * rho
        max_rho = max(max_rho, rho)

# Display the result
matplotlib.use('TkAgg')
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
ax.set_xlim((max_rho, -max_rho))  # Invert x axis
ax.set_ylim((-max_rho, max_rho))
ax.set_zlim((max_rho, -max_rho))  # Invert y axis

# Draw some "axes" so it's clear where (0, 0, 0)
ax.plot((-max_rho, max_rho), (0, 0), (0, 0), c="black")
ax.plot((0, 0), (-max_rho, max_rho), (0, 0), c="black")
ax.plot((0, 0), (0, 0), (-max_rho, max_rho), c="black")

ax.plot_surface(mesh_x, mesh_y, mesh_z, alpha=0.5)
ax.view_init(elev=30, azim=-150)

ax.set_xlabel('X (Surge)')
ax.set_ylabel('Y (Sway)')
ax.set_zlabel('Z (Heave)')

plt.show()