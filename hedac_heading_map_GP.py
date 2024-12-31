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
import time

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

# Grid for the map
x_min, x_max = 0, map.shape[0]
y_min, y_max = 0, map.shape[1]
grid_x, grid_y = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max), indexing='ij')
grid = np.vstack([grid_x.flatten(), grid_y.flatten()]).T

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
                                                dt=0.1,
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

# Normalize the density map (mind the NaN values)
norm_density_map = density_map[~np.isnan(density_map)]
norm_density_map = utilities.min_max_normalize(norm_density_map)

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
goal_density[map == 0] = norm_density_map

# fig = plt.figure(figsize=(15, 5))
# ax1 = fig.add_subplot(111)
# ax1.set_aspect('equal')
# ax1.set_title("Goal Density")
# ax1.contourf(grid_x, grid_y, goal_density, cmap='Greens')
# plt.show()


"""
===============================
Initialize heat equation related parameters
===============================
"""
param.width = map.shape[0]
param.height = map.shape[1]

param.beta = param.beta / param.area # Eq. 17 - Beta normalized 
param.local_cooling = param.local_cooling / param.area # Eq. 16 - Local cooling normalized

coverage_density = np.zeros_like(goal_density) # The coverage density
coverage_density_hist = np.empty((param.width, param.height, 0))
# heat = np.array(goal_density) # The heat is the goal density

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
param.kernel_size = coverage_block.shape[0] + param.safety_dist

"""
===============================
Gaussian process initialization
===============================
"""
# For the moment, no noise is added to the kernel
noise = 0
lengthscale = 1.0
constant = 1.0
kernel = C(1.0) * RBF(1.0) # + WhiteKernel(1e-3, (1e-8, 1e8))
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, normalize_y=False)#, alpha=noise**2)
pooled_dataset = np.empty((0, 3))
subset = np.empty((0, 3))
epsilon = 0.1 # 5% around the mean
decay = 0
rng = np.random.RandomState(param.random_seed)
std_pred_test = np.zeros_like(goal_density)

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
            """ Field of View (FOV) """
            fov_edges_moved = utilities.relative_move_fov(fov_edges[agent.id], 
                                                          last_position[agent.id], 
                                                          last_heading[agent.id], 
                                                          agent.x, 
                                                          agent.theta).squeeze()
            fov_edges_clipped = utilities.clip_polygon_no_convex(agent.x, fov_edges_moved, occ_map)
            fov_points = utilities.insidepolygon(fov_edges_clipped).astype(int)

            # Delete points outside the box
            fov_points = fov_points[
                (fov_points[:, 0] >= 0)
                & (fov_points[:, 0] < param.width)
                & (fov_points[:, 1] >= 0)
                & (fov_points[:, 1] < param.height)
            ]
            fov_probs = utilities.fov_coverage_block(fov_points, fov_edges_clipped, param.fov_depth)
            fov_probs = fov_probs / np.sum(fov_probs) # Normalize the probabilities

            # Update coverage density
            coverage_density = np.zeros_like(goal_density)
            coverage_density[fov_points[:, 0], fov_points[:, 1]] += fov_probs  # Eq. 3 - Coverage density

            # Append to history
            coverage_density_hist = np.dstack((coverage_density_hist, coverage_density))

            # Apply decay and sum
            decay_array = np.exp(-decay * (t - np.arange(t + 1)))[:, None, None].T
            coverage_density = (coverage_density_hist * decay_array).sum(axis=2)

            fov_edges[agent.id] = fov_edges_moved.squeeze()

            """ Goal density sampling """
            # Sample the goal density with noise
            samples = np.hstack((fov_points, (goal_density[fov_points[:, 0], fov_points[:, 1]] + rng.normal(0, noise, len(fov_points))).reshape(-1, 1)))

            # ============================= RAL Filter Mantovani2024
            if t > 0:
                std_test = std_pred_test[samples[:, 0].astype(int), samples[:, 1].astype(int)]
                indices = np.where(std_test > epsilon / 1.96)[0]  # Only epsilon because it's minMAX normalized
                print(f"Number of samples above the threshold: {len(indices)}/{len(samples)}")
                samples = samples[indices]

            # Pool and remove duplicates
            pooled_dataset = np.unique(np.vstack((pooled_dataset, utilities.max_pooling(samples, 5))), axis=0, return_index=False)

            # ============================= ICRA 2025
            K = utilities.rbf_white_kernel(pooled_dataset[:, :2], pooled_dataset[:, :2], 
                                            lengthscale=lengthscale, sigma_f=constant, sigma_n=noise)

            if t > 0:
                # Eigen decomposition and sorting
                eigvals, eigvecs = np.linalg.eigh(K)
                eigvals, eigvecs = eigvals[np.argsort(eigvals)[::-1]], eigvecs[:, np.argsort(eigvals)[::-1]]

                # Cumulative variance and selecting top eigenvectors explaining 95% variance
                n_top = np.searchsorted(np.cumsum(eigvals) / np.sum(eigvals), 0.95) + 1
                top_eigvecs = eigvecs[:, :n_top]

                # Identify influential points
                influential_points = np.unique(np.where(np.abs(top_eigvecs) > 0.99 * np.max(np.abs(top_eigvecs), axis=0))[0])

                # Extract subset of the dataset
                X_subset, y_subset = pooled_dataset[influential_points, :2], pooled_dataset[influential_points, 2].reshape(-1, 1)
            else:
                # First iteration, no filtering
                X_subset, y_subset = pooled_dataset[:, :2], pooled_dataset[:, 2].reshape(-1, 1)

            # Combine X_subset and y_subset
            subset = np.hstack((X_subset, y_subset))
            print("Pooled dataset shape:", pooled_dataset.shape)
            print("Subset shape:", subset.shape)

            # Fit the Gaussian Process
            gpr.fit(subset[:, :2], subset[:, 2])
            lengthscale = gpr.kernel_.get_params()['k2__length_scale']
            constant = np.sqrt(gpr.kernel_.get_params()['k1__constant_value'])

            # Prediction
            sTime = time.time()
            mu_pred, std_pred = gpr.predict(grid, return_std=True)
            print(f"Prediction time: {time.time() - sTime:.6f} s")
            mu_pred = mu_pred.reshape(map.shape)
            std_pred = std_pred.reshape(map.shape)

            mu_pred = utilities.min_max_normalize(mu_pred)
            std_pred = utilities.min_max_normalize(std_pred)

            std_pred_test = np.copy(std_pred)

            # Smooth
            mu_pred = np.exp(mu_pred)
            std_pred = np.exp(std_pred)
            # mu_pred = 10**mu_pred
            # std_pred = 10**std_pred

            combo = mu_pred + std_pred
            combo_density = np.where(map == 0, combo, 0)

            if t == 0:
                heat = np.array(utilities.normalize_mat(combo_density))
        else:
            """ Agent block # SKIP FOR THE MOMENT!! """
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

    coverage_density = utilities.min_max_normalize(coverage_density)
    combo_density = utilities.min_max_normalize(combo_density)

    diff = combo_density - coverage_density # Difference between goal and coverage densities

    source = np.maximum(diff, 0) # Source term
    source = utilities.normalize_mat(source) * param.area # Eq. 14 - Source term scaled

    heat = utilities.update_heat(heat,
                                source,
                                local_cooling,
                                map,
                                param).astype(np.float32)

    masked_heat = np.ma.array(heat, mask=(map != 0))

    gradient_y, gradient_x = np.gradient(heat.T, 1, 1)

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
        plt.close("all")
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        time_array = np.linspace(0, param.nbDataPoints, param.nbDataPoints)
        # Plot the agents
        ax[0].cla()
        ax[1].cla()

        ax[0].contourf(grid_x, grid_y, goal_density, cmap='coolwarm', levels=10)
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
        ax[1].contourf(grid_x, grid_y, heat, cmap='Blues', levels=10)
        delta_quiver = 5
        grad_x = gradient_x[::delta_quiver, ::delta_quiver] / np.linalg.norm(gradient_x[::delta_quiver, ::delta_quiver])
        grad_y = gradient_y[::delta_quiver, ::delta_quiver] / np.linalg.norm(gradient_y[::delta_quiver, ::delta_quiver])
        qx = np.arange(0, map.shape[0], delta_quiver)
        qy = np.arange(0, map.shape[1], delta_quiver)        
        q = ax[1].quiver(qx, qy, grad_x, grad_y, scale=1, scale_units='inches')
        ax[1].fill(occ_map[:, 0], occ_map[:, 1], 'k')
        plt.suptitle(f'Timestep: {t}')
        # plt.pause(0.0001)
        plt.show()