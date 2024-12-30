#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import warnings
from ergodic_control import models, utilities
import json
import os
from scipy.stats import multivariate_normal as mvn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt


warnings.filterwarnings("ignore")

np.random.seed(0) # Seed for reproducibility

def generate_repulsive_source(grid, max_distance=10, repulsion_strength=10):
    """
    Generate a repulsive source field for the walls based on the distance transform.
    """
    # Compute the distance transform (distance to nearest wall)
    distances = distance_transform_edt(grid)
    distances = np.clip(distances, 0, max_distance)  # Clip the distances to a maximum value
    repulsive_source = repulsion_strength / (distances + 1)  # Repulsion strength inversely proportional to distance
    return repulsive_source

"""
Load the map
"""
# Set the map
map_name = 'simpleMap_augmented'
# Load the map
map_path = os.path.join(os.getcwd(), 'example_maps/', map_name + '.npy')
map = np.load(map_path)

# Extend the map with 1 cell to avoid index out of bounds
padded_map = np.pad(map, 1, 'constant', constant_values=0)
occ_map = utilities.get_occupied_polygon(padded_map)
# Remove the padding
occ_map = occ_map - 1

"""
===============================
Parameters
===============================
"""
param_file = os.path.dirname(os.path.realpath(__file__)) + '/params/' + 'hedac_params_map.json'

with open(param_file, 'r') as f:
    param_data = json.load(f)

param = lambda: None

for key, value in param_data.items():
    setattr(param, key, value)

param.max_dtheta = np.pi / 18 # Maximum angular velocity (rad/s)
param.nbResX = map.shape[0] # Number of cells in the x-direction
param.nbResY = map.shape[1] # Number of cells in the y-direction
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

"""
===============================
Initialize Agents
===============================
"""
max_diffusion = np.max(param.alpha)
param.dt = min(
    0.2, (param.dx * param.dx) / (4.0 * max_diffusion)
)  # for the stability of implicit integration of Heat Equation

agents = []
x0_array = np.array([[180,180],
               [20,80],
               [80,80]])
for i in range(param.nbAgents):
    # x0 = np.random.uniform(0, param.nbResX, 2)
    x0 = x0_array[i]
    theta0 = np.random.uniform(0, 2 * np.pi)
    agent = models.SecondOrderAgentWithHeading(x=x0,
                                                theta=theta0,
                                                max_dx=param.max_dx,
                                                max_ddx=param.max_ddx,
                                                max_dtheta=param.max_dtheta,
                                                dt=param.dx,
                                                id=i)
    agents.append(agent)

"""
===============================
Goal Density
===============================
"""
free_cells = np.array(np.where(map == 0)).T  # Replace with your actual free cell array
_, density_map = utilities.generate_gmm_on_map(map,
                                             free_cells,
                                             param.nbGaussian,
                                             param.nbParticles,
                                             param.nbVar,
                                             random_state=param.random_seed)

# Compute the area of the map
cell_area = param.dx * param.dx
param.area = np.sum(map == 0) * cell_area

"""
Smoothing/Spreading RBF Approximation 
Probably not the best way to do it, but it works, also not sure if this is useful...
"""
def kernel_matrix(X, sigma=1):
    """Compute the kernel (RBF) matrix."""
    eps = 1 / (2 * sigma ** 2)
    return np.exp(-eps * np.linalg.norm(X[:, np.newaxis] - X[np.newaxis, :], axis=2) ** 2)

def fit(G, d):
    """Fit the RBF weights."""
    return np.dot(np.linalg.pinv(G), d)

def predict(inputs, X_train, m, sigma=1):
    """Predict values using the RBF approximation."""
    eps = 1 / (2 * sigma ** 2)
    pairwise_distances = np.linalg.norm(inputs[:, np.newaxis, :] - X_train[np.newaxis, :, :], axis=2)
    K = np.exp(-eps * pairwise_distances ** 2)
    return K @ m

# Sampling from the density map
N = 100 # Number of basis functions to use
sampled_indices = np.random.choice(len(free_cells), size=N, replace=False)

# Collect sampled values and their corresponding densities
samples = free_cells[sampled_indices]
d = np.array([density_map[tuple(cell)] for cell in samples]).reshape(-1, 1)

# RBF shape parameter
sigma = 20

# Compute the kernel matrix and fit RBF weights
G = kernel_matrix(samples, sigma)
m = fit(G, d)

# Predict the goal density using RBFs
S = predict(free_cells, samples, m, sigma)

# Reshape the result to the map's shape
S_map = np.full(map.shape, np.nan)  # Initialize map with NaN
S_map[free_cells[:, 0], free_cells[:, 1]] = S.flatten()
S_map = np.abs(S_map)
S_map = np.nan_to_num(S_map)
# Normalize and finalize the goal density
S_map = S_map.astype(np.float128) / np.sum(S_map, dtype=np.float128)

goal_density = np.zeros_like(map)
goal_density[free_cells[:, 0], free_cells[:, 1]] = density_map[free_cells[:, 0], free_cells[:, 1]]

repulsive_source = generate_repulsive_source(map, max_distance=1, repulsion_strength=10)
reuplsiive_source = 0

"""
===============================
Initialize heat equation related parameters
===============================
"""
param.height, param.width = goal_density.shape

param.beta = param.beta / param.area # Eq. 17 - Beta normalized 
param.local_cooling = param.local_cooling / param.area # Eq. 16 - Local cooling normalized

coverage_density = np.zeros_like(goal_density) # The coverage density
heat = np.array(goal_density) # The heat is the goal density

# max_diffusion = np.max(param.alpha)
param.dt = min(
    0.2, (param.dx * param.dx) / (4.0 * param.alpha)
)  # for the stability of implicit integration of Heat Equation

fov_edges = []
# Initialize the Field of View (FOV) and move it to the robot position
for agent in agents:
    tmp = utilities.init_fov(param.fov_deg, param.fov_depth)
    fov_edges.append(utilities.rotate_and_translate(tmp, agent.x, agent.theta))

coverage_block = utilities.agent_block(param.nbVar, param.min_kernel_val, param.agent_radius)
safety_dist = 3
param.kernel_size = coverage_block.shape[0] + safety_dist

"""
===============================
Main Loop
===============================
"""
last_heading = np.empty((0, 1))
last_position = np.empty((0, 2))

for agent in agents:
    last_heading = np.vstack((last_heading, agent.theta))
    last_position = np.vstack((last_position, agent.x))

collisions = []
for t in range(param.nbDataPoints):
    print(f'\nStep: {t}/{param.nbDataPoints}')

    local_cooling = np.zeros_like(goal_density)
    for agent in agents:
        if param.use_fov:
            if t == 7189:
                print(f'Agent {agent.id} hit the wall')
                plt.close('all')
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111)
                ax.fill(occ_map[:, 0], occ_map[:, 1], 'k')
                ax.scatter(agent.x[0], agent.x[1], c='red', s=100, marker='x')
                ax.plot(agent.x_hist[:, 0], agent.x_hist[:, 1], c='blue', lw=2)
                # Plot the kernel block
                block_min = agent.x - param.kernel_size // 2
                block_max = agent.x + param.kernel_size // 2
                ax.add_patch(plt.Rectangle(block_min, block_max[0] - block_min[0], block_max[1] - block_min[1], fill=None, edgecolor='green', lw=2))
                plt.show()
                 
            """ Field of View (FOV) """
            fov_edges_moved = utilities.relative_move_fov(fov_edges[agent.id], 
                                                          last_position[agent.id], 
                                                          last_heading[agent.id], 
                                                          agent.x, 
                                                          agent.theta).squeeze()
            fov_edges_clipped = utilities.clip_polygon_no_convex(agent.x, fov_edges_moved, occ_map)
            fov_points = utilities.insidepolygon(fov_edges_clipped, grid_step=param.dx).astype(int)

            # Delete points outside the box
            fov_points = fov_points[
                (fov_points[:, 0] >= 0)
                & (fov_points[:, 0] < param.width)
                & (fov_points[:, 1] >= 0)
                & (fov_points[:, 1] < param.height)
            ]
            fov_probs = utilities.fov_coverage_block(fov_points, fov_edges_clipped, param.fov_depth)
            fov_probs = fov_probs / np.sum(fov_probs) # Normalize the probabilities

            coverage_density[fov_points[:, 0], fov_points[:, 1]] += fov_probs # Eq. 3 - Coverage density
            fov_edges[agent.id] = fov_edges_moved.squeeze()
        else:
            """ Agent block """
            # adjusted_position = agent.x / param.dx
            adjusted_position = agent.x
            col, row = adjusted_position.astype(int)
            row_indices, row_start_kernel, num_kernel_rows = utilities.clamp_kernel_1d(
                row, 0, param.height, param.kernel_size
            )
            col_indices, col_start_kernel, num_kernel_cols = utilities.clamp_kernel_1d(
                col, 0, param.width, param.kernel_size
            )

            coverage_density[row_indices, col_indices] += coverage_block[
                row_start_kernel : row_start_kernel + num_kernel_rows,
                col_start_kernel : col_start_kernel + num_kernel_cols,
            ] # Eq. 3 - Coverage density

        # """ Local cooling """
        # adjusted_position = agent.x
        # col, row = adjusted_position.astype(int)
        # row_indices, row_start_kernel, num_kernel_rows = utilities.clamp_kernel_1d(
        #     row, 0, param.height, param.kernel_size, map, axis="row"
        # )
        # col_indices, col_start_kernel, num_kernel_cols = utilities.clamp_kernel_1d(
        #     col, 0, param.width, param.kernel_size, map, axis="col"
        # )

        # local_cooling[row_indices, col_indices] += coverage_block[
        #     row_start_kernel : row_start_kernel + num_kernel_rows,
        #     col_start_kernel : col_start_kernel + num_kernel_cols,
        # ]

    local_cooling = utilities.normalize_mat(local_cooling) * param.area # Eq. 15 - Local cooling normalized

    coverage = utilities.normalize_mat(coverage_density) # Eq. 4 - Coverage density normalized

    diff = goal_density - coverage # Eq. 6 - Difference between the goal density and the coverage density
    # # Min Max
    # diff = (diff - np.min(diff)) / (np.max(diff) - np.min(diff))
    # repulsive_source = (repulsive_source - np.min(repulsive_source)) / (np.max(repulsive_source) - np.min(repulsive_source))
    # diff = diff + repulsive_source # Eq. 12 - Repulsive source added to the difference
    source = np.maximum(diff, 0) ** 2 # Eq. 13 - Source term
    source = utilities.normalize_mat(source) * param.area # Eq. 14 - Source term scaled

    heat = utilities.update_heat(heat,
                                source,
                                local_cooling,
                                map,
                                param).astype(np.float32)

    masked_heat = np.ma.array(heat, mask=(map != 0))

    gradient_y, gradient_x = np.gradient(masked_heat.T, 1, 1) # Gradient of the heat

    gradient_x = gradient_x / np.linalg.norm(gradient_x)
    gradient_y = gradient_y / np.linalg.norm(gradient_y)

    # Check collisions
    # for agent in agents:
    #     for other_agent in agents:
    #         if agent.id != other_agent.id:
    #             if np.linalg.norm(agent.x - other_agent.x) < 1e-2:
    #                 print(f'Collision between agent {agent.id} and agent {other_agent.id}')
    #                 collisions.append([t, agent.x])

    for agent in agents:
        # Store the last position and heading
        last_heading[agent.id] = agent.theta
        last_position[agent.id] = agent.x
        # Update the agent
        grad = utilities.calculate_gradient_map(
            param, agent, gradient_x, gradient_y, map
        )
        agent.update(grad)

    if t == param.nbDataPoints - 1:
    # if t % 100 == 0:
        plt.close("all")
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        time_array = np.linspace(0, param.nbDataPoints, param.nbDataPoints)
        # Plot the agents
        ax[0].cla()
        ax[1].cla()

        ax[0].contourf(goal_density.T, cmap='Reds', levels=10)
        ax[0].fill(occ_map[:, 0], occ_map[:, 1], 'k')
        for agent in agents:
            lines = utilities.colored_line(agent.x_hist[:, 0],
                                            agent.x_hist[:, 1],
                                            time_array,
                                            ax[0],
                                            cmap='Greys',
                                            linewidth=2)
        # FOV
        if param.use_fov:
            for agent in agents:
                fov_edges_clipped = utilities.clip_polygon_no_convex(agent.x, fov_edges[agent.id], occ_map)
                ax[0].fill(fov_edges_clipped[:, 0], fov_edges_clipped[:, 1], color='blue', alpha=0.3)
                # Heading
                ax[0].quiver(agent.x[0], agent.x[1], np.cos(agent.theta), np.sin(agent.theta), scale = 2, scale_units='inches')
                ax[0].scatter(agent.x_hist[0, 0], agent.x_hist[0, 1], s=100, facecolors='none', edgecolors='green', lw=2)
                ax[0].scatter(agent.x[0], agent.x[1], c='k', s=100, marker='o', zorder=10)
                # Plot the kernel block
                block_min = agent.x - param.kernel_size // 2
                block_max = agent.x + param.kernel_size // 2
                ax[0].add_patch(plt.Rectangle(block_min, block_max[0] - block_min[0], block_max[1] - block_min[1], fill=None, edgecolor='green', lw=2))
                ax[1].scatter(agent.x[0], agent.x[1], c='black', s=100, marker='o')
                # Plot the heading
                ax[1].quiver(agent.x[0], agent.x[1], np.cos(agent.theta), np.sin(agent.theta), scale = 2, scale_units='inches')

        # Plot the collisions
        for t in range(len(collisions)):
            if collisions[t][0] == t:
                ax[0].scatter(collisions[t][1][0], collisions[t][1][1], c='red', s=100, marker='x')
        # Heat
        ax[1].contourf(heat.T, cmap='Blues', levels=10)
        delta_quiver = 5
        grid_x = np.arange(0, heat.shape[0], delta_quiver)
        grid_y = np.arange(0, heat.shape[1], delta_quiver)
        gradient_x = gradient_x[::delta_quiver, ::delta_quiver] / np.linalg.norm(gradient_x[::delta_quiver, ::delta_quiver])
        gradient_y = gradient_y[::delta_quiver, ::delta_quiver] / np.linalg.norm(gradient_y[::delta_quiver, ::delta_quiver])
        ax[1].quiver(grid_x, grid_y, gradient_x, gradient_y, scale=1, scale_units='inches')
        ax[1].fill(occ_map[:, 0], occ_map[:, 1], 'k')
        plt.suptitle(f'Timestep: {t}')
        # plt.pause(0.0001)
        plt.show()