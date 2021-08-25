import math
import numpy as np
from scipy.optimize import linprog
import json
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import typing as t
import sys
from pathlib import Path

RESOLUTION = 100 # Runtime is O(n^2) with respect to resolution!
MAX_THRUSTER_FORCE = [-2.9, 3.71] # Lifted from the BlueRobotics public performance data (kgf)
T_I_QUAD_COEF_FWD = [.741, 1.89, -.278] #coefficiants of the quadratic approximating current draw as a function of thrust in the forward direction in the form ax^2 + bx + c
T_I_QUAD_COEF_REV = [1.36, 2.04, -.231] #reverse direction
I_LIMIT = 22 #maximum allowable current

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

    #First Simplex run. Find the maximum thrust in the desired direction
    objective = []
    left_of_equality = []
    
    thrusts_y = []
    thrusts_z = []
    
    torques_x = []
    torques_y = []
    torques_z = []
    
    bounds = []
    
    for i, orientation in enumerate(transformed_orientations, start = 0):
        objective.append(-orientation[0]) #Algorithm minimizes only, so the objective function needs to be negated.
        
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
    
    max_thrust_result = linprog(c=objective, A_ub = None, b_ub = None, A_eq = left_of_equality, b_eq = right_of_equality, bounds=bounds, method="highs")

    max_thrust = -.999 * max_thrust_result.fun #some sort of precision/numerical error makes this bullshit necessary

    #Second Simplex run. Find the minimum current that produces the same thrust as the first result
    objective_mincurrent = []
    left_of_equality_mincurrent = []
    
    thrusts_x_mincurrent = []
    thrusts_y_mincurrent = []
    thrusts_z_mincurrent = []
    
    torques_x_mincurrent = []
    torques_y_mincurrent = []
    torques_z_mincurrent = []
    
    bounds_mincurrent = []

    for i, orientation in enumerate(transformed_orientations, start = 0):
        #duplicate each thruster into a forward and a reverse half-thruster
        objective_mincurrent.append(1) #minimize the thrust of all thrusters weighted equally
        objective_mincurrent.append(1)
        
        thrusts_x_mincurrent.append(orientation[0])
        thrusts_x_mincurrent.append(-orientation[0]) #duplicate, reversed thruster
        
        thrusts_y_mincurrent.append(orientation[1])
        thrusts_y_mincurrent.append(-orientation[1])
        
        thrusts_z_mincurrent.append(orientation[2])
        thrusts_z_mincurrent.append(-orientation[2])
        
        
        torques_x_mincurrent.append(t_constraints[i][0])
        torques_x_mincurrent.append(-t_constraints[i][0])
        
        torques_y_mincurrent.append(t_constraints[i][1])
        torques_y_mincurrent.append(-t_constraints[i][1])
        
        torques_z_mincurrent.append(t_constraints[i][2])
        torques_z_mincurrent.append(-t_constraints[i][2])
        
        
        bounds_mincurrent.append((0, 3.71))
        bounds_mincurrent.append((0, 2.90))
        
    left_of_equality_mincurrent.append(thrusts_x_mincurrent)
    left_of_equality_mincurrent.append(thrusts_y_mincurrent)
    left_of_equality_mincurrent.append(thrusts_z_mincurrent)
    
    left_of_equality_mincurrent.append(torques_x_mincurrent)
    left_of_equality_mincurrent.append(torques_y_mincurrent)
    left_of_equality_mincurrent.append(torques_z_mincurrent)
    
    right_of_equality_mincurrent = [
        max_thrust, #x thrust constrained to previous maximum
        0, #y thrust
        0, #z thrust
        0, #x torque
        0, #y torque
        0, #z torque
    ]
    
    min_current_result = linprog(c=objective_mincurrent, A_ub = None, b_ub = None, A_eq = left_of_equality_mincurrent, b_eq = right_of_equality_mincurrent, bounds=bounds_mincurrent, method="highs")    

    min_current_duplicated_array = min_current_result.x
    
    min_current_true_array = []
    for i in range(0, len(min_current_duplicated_array)-1, 2):
        min_current_true_array.append(min_current_duplicated_array[i] - min_current_duplicated_array[i + 1]) #combine half-thrusters into full thrusters

    current_quadratic = [0] * 3
    
    for thrust in min_current_true_array:
        if thrust >= 0: #use the forward thrust coefficiants
            current_quadratic[0] += T_I_QUAD_COEF_FWD[0] * thrust**2 #a * t^2
            current_quadratic[1] += T_I_QUAD_COEF_FWD[1] * thrust    #b * t
            current_quadratic[2] += T_I_QUAD_COEF_FWD[2]             #c
        else: #use the reverse thrust coefficiants
            current_quadratic[0] += T_I_QUAD_COEF_REV[0] * (-thrust)**2
            current_quadratic[1] += T_I_QUAD_COEF_REV[1] * (-thrust)
            current_quadratic[2] += T_I_QUAD_COEF_REV[2] 

    current_quadratic[2] -= I_LIMIT #ax^2 + bx + c = I -> ax^2 + bx + (c-I) = 0

    thrust_multiplier = min(1., max(np.roots(current_quadratic))) #solve quadratic, take the proper point, and clamp it to a maximum of 1.0
    
    thrust_value = 0
    for i in range(0, len(min_current_true_array)):
        thrust_value += min_current_true_array[i] * transformed_orientations[i][0] #get total thrust in target direction
    
    return thrust_value * thrust_multiplier

in_file = ""

if(len(sys.argv) >= 2):
    in_file = sys.argv[1]

if Path(in_file).is_file() == False:
    in_file = "thrusters.json"  
    print("invalid file name, using \"thrusters.json\"")

# Load the thruster data from a json file
with open(in_file) as input_file:
    input_transforms = json.loads(input_file.read())

thrusters = []
for transform in input_transforms:
    thrusters.append(Thruster3D(transform['x'], transform['y'], transform['z'], transform['theta'], transform['phi']))

# Calculate the torque constrains which will apply to every iteration
torque_constraints = []

for thruster in thrusters:
    torque_constraints.append(thruster.torque())

#get_max_thrust(thrusters, np.array([1, 0, 0]), torque_constraints)

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
        rho = get_max_thrust(thrusters, np.array([x, y, z]), torque_constraints)
        mesh_x[i][j] = x * rho
        mesh_y[i][j] = y * rho
        mesh_z[i][j] = z * rho
        max_rho = max(max_rho, rho)
max_rho = np.ceil(max_rho)

color_index = np.sqrt(mesh_x**2 + mesh_y**2 + mesh_z**2,)

norm = matplotlib.colors.Normalize(vmin = color_index.min(), vmax = color_index.max())

color_index_modified = (color_index - color_index.min())/(color_index.max() - color_index.min())

# Display the result
matplotlib.use('TkAgg')
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
ax.set_xlim((max_rho, -max_rho))  # Invert x axis
ax.set_ylim((-max_rho, max_rho))
ax.set_zlim((max_rho, -max_rho))  # Invert y axis
ax.set_box_aspect((1, 1, 1))

# Draw some "axes" so it's clear where (0, 0, 0)
ax.plot((-max_rho, max_rho), (0, 0), (0, 0), c="black")
ax.plot((0, 0), (-max_rho, max_rho), (0, 0), c="black")
ax.plot((0, 0), (0, 0), (-max_rho, max_rho), c="black")

thrusterloc_x = []
thrusterloc_y = []
thrusterloc_z = []

thrusterdir_x = []
thrusterdir_y = []
thrusterdir_z = []

for thruster in thrusters:
    thrusterloc_x.append(2*thruster.pos[0])
    thrusterloc_y.append(2*thruster.pos[1])
    thrusterloc_z.append(2*thruster.pos[2])
    
    thrusterdir_x.append(2*thruster.orientation[0])
    thrusterdir_y.append(2*thruster.orientation[1])
    thrusterdir_z.append(2*thruster.orientation[2])

ax.quiver(thrusterloc_x, thrusterloc_y, thrusterloc_z, thrusterdir_x, thrusterdir_y, thrusterdir_z)

ax.plot_surface(mesh_x, mesh_y, mesh_z, alpha=0.5, facecolors=cm.jet(color_index_modified), edgecolors='w', linewidth=.1)
ax.view_init(elev=30, azim=-150)

m = cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
m.set_array([])
plt.colorbar(m, ticks=[color_index.min(), color_index.max()])

ax.set_xlabel('X (Surge)')
ax.set_ylabel('Y (Sway)')
ax.set_zlabel('Z (Heave)')

plt.show()