#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import warnings
from ergodic_control import models, utilities
import json
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
warnings.filterwarnings("ignore")

"""
Load the map
"""
# Set the map
map_name = 'simpleMap_05'
# Load the map
map_path = os.path.join(os.getcwd(), 'example_maps/', map_name + '.npy')
map = np.load(map_path)
# Check if the map is closed or open
closed_map = True
# Extend the map with 1 cell to avoid index out of bounds
padded_map = np.pad(map, 1, 'constant', constant_values=1)
occ_map = utilities.get_occupied_polygon(padded_map)
occ_map = occ_map - 1

x_min, x_max = 0, map.shape[0]
y_min, y_max = 0, map.shape[1]
grid_x, grid_y = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max), indexing='ij')
grid = np.vstack([grid_x.flatten(), grid_y.flatten()]).T
"""
===============================
Parameters
===============================
"""
param_file = os.path.dirname(os.path.realpath(__file__)) + '/params/' + 'hedac_params_map_05.json'

with open(param_file, 'r') as f:
    param_data = json.load(f)

param = lambda: None

for key, value in param_data.items():
    setattr(param, key, value)

param.max_dtheta = np.pi / param.max_dtheta # Maximum angular velocity (rad/s)
param.nbResX = map.shape[0] # Number of cells in the x-direction
param.nbResY = map.shape[1] # Number of cells in the y-direction

"""
===============================
Initialize Agents
===============================
"""
np.random.seed(param.random_seed)

max_diffusion = np.max(param.alpha)
# CFL condition
param.dt = min(
    0.2, (param.dx ** 2) / (4.0 * max_diffusion)
)  # for the stability of implicit integration of Heat Equation

agents = []
x0_array = np.array([[10.5, 25],
                [30, 25]])

for i in range(param.nbAgents):
    x0 = x0_array[i]
    theta0 = np.random.uniform(0, 2 * np.pi)
    agent = models.SecondOrderAgentWithHeading(x=x0,
                                                theta=theta0,
                                                max_dx=param.max_dx,
                                                max_ddx=param.max_ddx,
                                                max_dtheta=param.max_dtheta,
                                                dt=param.dt_agent,
                                                id=i)
    agent.sens_range = param.sens_range
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

goal_density = np.zeros_like(map)
goal_density[map == 0] = norm_density_map

"""
===============================
Initialize heat equation related parameters
===============================
"""
param.width = map.shape[0]
param.height = map.shape[1]

param.beta = param.beta / param.area # Eq. 17 - Beta normalized 
param.local_cooling = param.local_cooling / param.area # Eq. 16 - Local cooling normalized

local_cooling = np.zeros_like(goal_density) # The local cooling
coverage_density = np.zeros_like(goal_density) # The coverage density

# heat = np.array(goal_density) # The heat is the goal density
for agent in agents:
    tmp = utilities.init_fov(param.fov_deg, param.fov_depth)
    agent.fov_edges = utilities.rotate_and_translate(tmp, agent.x, agent.theta)

coverage_block = utilities.agent_block(param.nbVar, param.min_kernel_val, param.agent_radius)
param.kernel_size = coverage_block.shape[0] + param.safety_dist

"""
===============================
Gaussian process and decay
===============================
"""
# For the moment, no noise is added to the kernel
noise = 0
kernel = C(1.0) * RBF(1.0) # + WhiteKernel(1e-3, (1e-8, 1e8))
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, normalize_y=False)
pooled_dataset = np.empty((0, 3))
subset = np.empty((0, 3))

# decay_array = np.exp(-param.decay *  np.arange(param.nbDataPoints))

# memmap for faster simulations
# coverage_density_hist = np.memmap('coverage_density_hist.dat', dtype='int32', mode='w+', shape=(15, 2, param.nbDataPoints, param.nbAgents))
# coverage_density_prob_hist = np.memmap('coverage_density_prob_hist.dat', dtype='float32', mode='w+', shape=(15, param.nbDataPoints, param.nbAgents))
# decay_weights = np.memmap('decay_array.dat', dtype='float32', mode='w+', shape=(param.nbDataPoints,))
coverage_density_hist = np.zeros((15, 2, param.nbDataPoints, param.nbAgents), dtype=int)
coverage_density_prob_hist = np.zeros((15, param.nbDataPoints, param.nbAgents), dtype=float)

std_pred_test = np.zeros_like(goal_density)

# Precompute the GPR for faster simulations
preSamplesN = 500
preSamples = np.random.randint(0, len(free_cells), preSamplesN)
preSamples = np.hstack((free_cells[preSamples], goal_density[free_cells[preSamples][:, 0], free_cells[preSamples][:, 1]].reshape(-1, 1)))
gpr.fit(preSamples[:, :2], preSamples[:, 2])
constant = np.sqrt(gpr.kernel_.get_params()['k1__constant_value'])
lengthscale = gpr.kernel_.get_params()['k2__length_scale']
print("Pre-training finished.")
print(f"Lengthscale: {lengthscale}, Constant: {constant}\n\n")
subset_hash_old = 0
ergodic_metric = np.zeros((param.nbDataPoints, param.nbAgents))

"""
Debug & animation
"""
grad = 0
_heat_hist = []

"""
===============================
Main Loop
===============================
"""
for agent in agents:
    agent.last_heading = agent.theta
    agent.last_position = agent.x
    agent.samples = np.empty((0, 3))
    agent.pooled_dataset = np.empty((0, 3))
    agent.subset = np.empty((0, 3))
    # Stack 100 of the preSamples
    # agent.pooled_dataset = np.vstack((agent.pooled_dataset, preSamples[:300]))
    agent.neighbors = []

# Create a matrix with ones on the diagonal
adjacency_matrix = np.eye(param.nbAgents)
obstacles = np.zeros((param.nbAgents, param.nbAgents))

# Example of chunked processing for memmap arrays
chunk_size = param.nbDataPoints // 10
num_chunks = param.nbDataPoints // chunk_size

for chunk in range(num_chunks):
    start_idx = chunk * chunk_size
    end_idx = min((chunk + 1) * chunk_size, param.nbDataPoints)
    print(f"Processing chunk {chunk + 1}/{num_chunks} (steps {start_idx} to {end_idx})")
        
    for t in range(start_idx, end_idx):
        print(f'\nStep: {t}/{param.nbDataPoints}')
        if param.nbAgents > 1 & t > 0:
            utilities.share_samples(agents, map, param.sens_range, adjacency_matrix)
        for agent in agents:
            if param.use_fov:                
                """ Field of View (FOV) """
                # Probably we can avoid clipping if we handle the collected samples for the GP?
                fov_edges_moved = utilities.relative_move_fov(agent.fov_edges,
                                                            agent.last_position,
                                                            agent.last_heading, 
                                                            agent.x, 
                                                            agent.theta).squeeze()
                fov_edges_clipped = utilities.clip_polygon_no_convex(agent.x, fov_edges_moved, occ_map, closed_map)
                fov_points = utilities.insidepolygon(fov_edges_clipped).astype(int)

                # Delete points outside the box
                fov_probs = utilities.fov_coverage_block(fov_points, fov_edges_clipped, param.fov_depth)
                fov_probs = utilities.normalize_mat(fov_probs)

                coverage_density_hist[:len(fov_points), :, t, agent.id] = fov_points # Save the points for the decay
                coverage_density_prob_hist[:len(fov_points), t, agent.id] = fov_probs # Save the probabilities for the decay

                agent.coverage_density = np.zeros_like(goal_density)
                utilities.apply_decay(agent.coverage_density, 
                                    coverage_density_hist[:, :, :, agent.id], 
                                    coverage_density_prob_hist[:, :, agent.id],
                                    t, 
                                    500,
                                    param.decay)
    
                agent.fov_edges = fov_edges_moved

                """ Goal density sampling """
                # Sample the goal density with noise
                samples = np.hstack((fov_points, (goal_density[fov_points[:, 0], fov_points[:, 1]] + np.random.normal(0, noise, len(fov_points))).reshape(-1, 1)))
                agent.samples = np.vstack((agent.samples, samples))

                # ============================= RAL Filter Mantovani2024
                if t > 0:
                    std_test = agent.std_pred_test[agent.samples[:, 0].astype(int), agent.samples[:, 1].astype(int)]
                    agent.samples = agent.samples[np.where(std_test > 0.75)[0]]

                    if len(agent.samples) != 0:
                        agent.pooled_dataset = np.unique(np.vstack((agent.pooled_dataset, utilities.max_pooling(agent.samples, 5))), axis=0, return_index=False)
                        if len(agent.pooled_dataset) != 0:
                            agent.subset = np.hstack((agent.pooled_dataset[:, :2], agent.pooled_dataset[:, 2].reshape(-1, 1)))

                            # gpr.fit(agent.subset[:, :2], agent.subset[:, 2]) # Skipping for simulation speed

                            agent.combo_density, agent.mu, agent.std, agent.std_pred_test = utilities.compute_combo(agent.subset, grid, map, gpr.kernel_)
                            agent.std[agent.std < 0.3] = 0
                else:
                    agent.pooled_dataset = np.unique(np.vstack((agent.pooled_dataset, utilities.max_pooling(agent.samples, 5))), axis=0, return_index=False)
                    agent.subset = np.hstack((agent.pooled_dataset[:, :2], agent.pooled_dataset[:, 2].reshape(-1, 1)))
                    agent.combo_density, agent.mu, agent.std, agent.std_pred_test = utilities.compute_combo(agent.subset, grid, map, gpr.kernel_)
                    agent.std[agent.std < 0.3] = 0
                print(f"agent.subset shape: {agent.subset.shape}")

                if t == 0:
                    agent.heat = np.array(utilities.normalize_mat(agent.combo_density))
            else:
                """ Agent block # SKIP FOR THE MOMENT!! """
                adjusted_position = agent.x
                x, y = adjusted_position.astype(int)
                # Don't care if hits walls cause handled in heat eq.
                x_indices, x_start_kernel, num_kernel_dx = utilities.clamp_kernel_1d(
                    x, 0, param.width, param.kernel_size
                )
                y_indices, y_start_kernel, num_kernel_dy = utilities.clamp_kernel_1d(
                    y, 0, param.height, param.kernel_size
                )

                local_cooling[x_indices, y_indices] += coverage_block[
                    x_start_kernel : x_start_kernel + num_kernel_dx,
                    y_start_kernel : y_start_kernel + num_kernel_dy,
                ]

            diff = utilities.normalize_mat(agent.combo_density) - utilities.normalize_mat(agent.coverage_density)
            # source = np.maximum(diff, 0)**2 # Source term
            # diff = utilities.min_max_normalize(agent.combo_density) - utilities.min_max_normalize(agent.coverage_density)
            # L2 norm
            ergodic_metric[t, agent.id] = np.linalg.norm(diff)
            # source = np.maximum(diff, 0) # RElu function
            # source = np.exp(diff)
            source = np.maximum(diff, 0)**2 # RElu function
            # source = np.exp(source) - 1

            agent.source = utilities.normalize_mat(source) * param.area # Eq. 14 - Source term scaled

            agent.heat = utilities.update_heat_optimized(agent.heat,
                                        agent.source,
                                        map,
                                        param.dt,
                                        # param.laplacian_kernel,
                                        param.alpha,
                                        param.source_strength,
                                        param.beta,
                                        param.dx).astype(np.float32)
            
            gradient_y, gradient_x = np.gradient(agent.heat.T, 1, 1)

            gradient_x /= np.linalg.norm(gradient_x)
            gradient_y /= np.linalg.norm(gradient_y)

            # Store the last position and heading
            agent.last_heading = agent.theta
            agent.last_position = agent.x
            # Update the agent
            grad = utilities.calculate_gradient_map(
                param, agent, gradient_x, gradient_y, map
            )
            agent.update(grad)

            _heat_hist.append(agent.heat)

plt.close("all")
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
time_array = np.linspace(0, param.nbDataPoints, param.nbDataPoints)
# Plot the agents
ax[0].cla()
ax[1].cla()

ax[0].set_aspect('equal')
ax[1].set_aspect('equal')

ax[0].contourf(grid_x, grid_y, goal_density, cmap='Greys', levels=10)
ax[0].pcolormesh(grid_x, grid_y, np.where(map == 0, np.nan, map), cmap='gray')

collisions = []
for i in range(len(agents)):
    for j in range(i + 1, len(agents)):
            if np.any(np.all(np.linalg.norm(agents[i].x_hist - agents[j].x_hist, axis=1) < 0.5, axis=0)):
                collisions.append((i, j))

# Plot the collisions
for i, j in collisions:
    ax[0].plot(agents[i].x_hist[:, 0], agents[i].x_hist[:, 1], color='red', alpha=0.5)
    ax[0].plot(agents[j].x_hist[:, 0], agents[j].x_hist[:, 1], color='blue', alpha=0.5)

cmaps = ['Blues', 'Reds', 'Purples', 'Greens', 'Oranges']
for agent in agents:
    lines = utilities.colored_line(agent.x_hist[:, 0],
                                    agent.x_hist[:, 1],
                                    time_array,
                                    ax[0],
                                    cmap=cmaps[agent.id],
                                    linewidth=2)
# FOV
if param.use_fov:
    for agent in agents:
        fov_edges_clipped = utilities.clip_polygon_no_convex(agent.x, agent.fov_edges, occ_map, closed_map)
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


for i in range(param.nbAgents):
    for j in range(i + 1, param.nbAgents):
        if adjacency_matrix[i, j] == 1:
            # Plot a line between the two agents
            ax[0].plot([agents[i].x[0], agents[j].x[0]], [agents[i].x[1], agents[j].x[1]], color='orange', alpha=1, lw=2, linestyle='--')
            

# Heat
ax[1].contourf(grid_x, grid_y, agents[0].heat, cmap='Blues', levels=10)
ax[1].pcolormesh(grid_x, grid_y, np.where(map == 0, np.nan, map), cmap='gray')
delta_quiver = 2
grad_x = gradient_x[::delta_quiver, ::delta_quiver] / np.linalg.norm(gradient_x[::delta_quiver, ::delta_quiver])
grad_y = gradient_y[::delta_quiver, ::delta_quiver] / np.linalg.norm(gradient_y[::delta_quiver, ::delta_quiver])
qx = np.arange(0, map.shape[0], delta_quiver)
qy = np.arange(0, map.shape[1], delta_quiver)        
q = ax[1].quiver(qx, qy, grad_x, grad_y, scale=1, scale_units='inches')
plt.suptitle(f'Timestep: {t}')
plt.tight_layout()
# plt.pause(0.0001)
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
# Plot the mean, std
ax[0].cla()
ax[1].cla()
ax[0].set_aspect('equal')
ax[1].set_aspect('equal')
ax[0].contourf(grid_x, grid_y, agents[0].mu, cmap='Blues')
ax[1].contourf(grid_x, grid_y, agents[0].std, cmap='Reds')
bar = plt.colorbar(ax[0].contourf(grid_x, grid_y, agents[0].mu, cmap='Blues'), ax=ax[0])
ax[0].set_title("Mean")
bar = plt.colorbar(ax[1].contourf(grid_x, grid_y, agents[0].std, cmap='Reds'), ax=ax[1])
ax[1].set_title("Std")
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
# Plot the coverage density and the source
ax[0].cla()
ax[1].cla()
ax[0].set_aspect('equal')
ax[1].set_aspect('equal')
# ax[0].contourf(grid_x, grid_y, utilities.min_max_normalize(agents[0].coverage_density), cmap='Greens')
bar = plt.colorbar(ax[0].contourf(grid_x, grid_y, utilities.min_max_normalize(agents[0].coverage_density), cmap='Greens'), ax=ax[0])
ax[0].set_title("Coverage Density")
ax[1].contourf(grid_x, grid_y, agents[0].source, cmap='Oranges')
bar = plt.colorbar(ax[1].contourf(grid_x, grid_y, agents[0].source, cmap='Oranges'), ax=ax[1])
ax[1].set_title("Source")
plt.show()

fig = plt.figure(figsize=(15, 5))
# Plot the ergodic metric over time
ax = fig.add_subplot(111)
ax.set_title("Ergodic Metric")
ax.plot(time_array, ergodic_metric)
ax.set_xlabel("Time")
ax.set_ylabel("Ergodic Metric")
plt.show()