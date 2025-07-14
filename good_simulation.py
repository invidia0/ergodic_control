import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from ergodic_control import models, utilities
import json

import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
import warnings
warnings.filterwarnings("ignore")
import time

"""
Load the map
"""
# Set the map
map_name = 'simpleMap_05'
# Load the map
map_path = os.path.join(os.getcwd(), 'example_maps/', map_name + '.npy')
map = np.load(map_path)

free_cells = np.array(np.where(map == 0)).T

# Create a map  with just the border
# border_map = np.ones_like(map)
# border_map[1:-1, 1:-1] = 0
# map = border_map

# Is the map closed?
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
param_file = os.path.dirname(os.path.realpath(__file__)) + '/params/' + 'doubleintegrator.json'

with open(param_file, 'r') as f:
    param_data = json.load(f)

param = lambda: None

for key, value in param_data.items():
    setattr(param, key, value)

np.random.seed(param.random_seed)

# param.max_dtheta = np.pi / param.max_dtheta # Maximum angular velocity (rad/s)
param.nbResX = map.shape[0] # Number of cells in the x-direction
param.nbResY = map.shape[1] # Number of cells in the y-direction

"""
===============================
Initialize Agents
===============================
"""

# CFL (Courant-Friedrichs-Lewy) condition for implicit integration and heat equation stability
param.dt = min(
    1.0, (param.dx ** 2) / (4.0 * np.max(param.alpha))
)

agents = []

# Fixed
x0_array = np.array([[9, 7],
                     [15, 20],
                     [5, 40],
                     [40, 40]])

# Random
# x0_array = free_cells[np.random.choice(free_cells.shape[0], param.nbAgents, replace=False)]

print(f"Initial positions: {x0_array}")

for i in range(param.nbAgents):
    x0 = x0_array[i]
    theta0 = np.random.uniform(0, 2 * np.pi)

    agent = models.DoubleIntegratorAgent(
        x=x0,
        theta=theta0,
        max_dx=param.max_dx,
        max_ddx=param.max_ddx,
        max_dtheta=param.max_dtheta,
        max_ddtheta=param.max_ddtheta,
        dt=param.dt_agent,
        id=i
    )

    agent.sens_range = param.sens_range
    agents.append(agent)

"""
===============================
Goal Density
===============================
"""
free_cells = np.array(np.where(map == 0)).T  # Replace with your actual free cell array

means = np.array([[30, 8], [20, 45], [8, 8]])
cov = np.array([[[10, 0], [0, 10]], [[10, 0], [0, 10]], [[10, 0], [0, 10]]])
density_map = utilities.gauss_pdf(grid, means[0], cov[0])
density_map = utilities.min_max_normalize(density_map)

norm_density_map = utilities.min_max_normalize(density_map).reshape(map.shape)
norm_density_map = density_map.reshape(map.shape)

# Compute the area of the map
cell_area = param.dx * param.dx
param.area = np.sum(map == 0) * cell_area

goal_density = np.zeros_like(map)
goal_density[free_cells[:, 0], free_cells[:, 1]] = norm_density_map[free_cells[:, 0], free_cells[:, 1]]
# Min-max normalize the goal density
goal_density = np.abs(goal_density)
goal_density = utilities.min_max_normalize(goal_density) # remember to normalize
goal_density_norm = utilities.normalize_mat(goal_density)

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(111)
ax.set_aspect('equal')
# Plot the map and the ground truth goal density
cont = plt.contourf(grid_x, grid_y, goal_density, cmap='RdPu', levels=10)
plt.pcolormesh(grid_x, grid_y, np.where(map == 0, np.nan, map), cmap='gray')
plt.colorbar(cont, ax=ax, label='Density')
plt.show()

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

for agent in agents:
    tmp = utilities.init_fov(param.fov_deg, param.fov_depth)
    agent.fov_edges = utilities.rotate_and_translate(tmp, agent.x, agent.theta)

coverage_block = utilities.agent_block(param.nbVar, param.min_kernel_val, param.agent_radius)
param.kernel_size = coverage_block.shape[0]

"""
================
Gaussian process
================
"""
# # For the moment, no noise is added to the kernel
noise = 0.0001
# kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e5)) # + WhiteKernel(noise_level=noise, noise_level_bounds=(1e-5, 1e1))
# gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=False, alpha=1e-5)
# # dataset = np.empty((0, 3))

# std_pred_test = np.zeros_like(goal_density)

kernel = (
    C(1.0, constant_value_bounds=(1e-3, 1e3))
    * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))  # space
    + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e1))
)

gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, alpha=1e-5, normalize_y=True)

gpr.kernel_ = kernel

"""
=========
Main Loop
=========
"""
buffer_size = 500  # Size of the circular buffer for coverage density
for agent in agents:
    agent.heat = np.empty_like(goal_density)
    agent.samples = np.empty((0, 4))
    agent.dataset = np.empty((0, 4))
    # agent.coverage_updates = []  # List to store (fov_points, fov_probs, step) tuples
    # agent.max_coverage_history = 500
    agent.coverage_buffer = np.zeros((buffer_size, *goal_density.shape))  # Circular buffer
    agent.buffer_index = 0
    agent.buffer_full = False

    agent.neighbors = []
    agent.last_neighbors = []
    agent.local_cooling = np.zeros_like(goal_density)
    agent.coverage_density = np.zeros_like(goal_density)
    agent.cov_dens = np.zeros_like(goal_density)  # Coverage density for each agent

    # Stack the precomputed samples
    # agent.dataset = np.vstack((agent.dataset, preSamples))

adjacency_matrix = np.eye(param.nbAgents)

# Chunk processing for speed
chunk_size = param.nbDataPoints // 10
num_chunks = param.nbDataPoints // chunk_size

violations = []
min_safe_range = 1

plt.close('all')  # Close all previous plots
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(111)

spatial_decay = 100
temporal_decay = 100
takeTh = 0.90
removeTh = 0.75  # 0.80 good
# Start time
time_samples_array = np.array([[]])

# Create a matrix of blocks of sens_range x sens_range that will be updated with the coverage density around of each agent current position
local_shared_block = np.zeros((param.nbAgents, param.sens_range, param.sens_range), dtype=float)

ergodic_metric = np.zeros((param.nbDataPoints, param.nbAgents), dtype=float)

PLOT = False

save_debug = True

if save_debug:
    debug_mu = np.zeros((param.nbDataPoints, *map.shape), dtype=float)
    debug_std = np.zeros((param.nbDataPoints, *map.shape), dtype=float)
    debug_combo_density = np.zeros((param.nbDataPoints, *map.shape), dtype=float)


for step in range(param.nbDataPoints):
    if step // 10:
        print(f"Step - {step}")

    if step == 5000:
        means = np.array([[10, 10]])
        cov = np.array([[[10, 0], [0, 10]]])
        density_map = utilities.gauss_pdf(grid, means[0], cov[0])
        density_map = utilities.min_max_normalize(density_map)

        norm_density_map = utilities.min_max_normalize(density_map).reshape(map.shape)
        norm_density_map = density_map.reshape(map.shape)

        goal_density = np.zeros_like(map)
        goal_density[free_cells[:, 0], free_cells[:, 1]] = norm_density_map[free_cells[:, 0], free_cells[:, 1]]
        # Min-max normalize the goal density
        goal_density = np.abs(goal_density)
        goal_density = utilities.min_max_normalize(goal_density) # remember to normalize
        goal_density_norm = utilities.normalize_mat(goal_density)

                
    if param.nbAgents > 1 & step > 0:
        # Implement DAC + Kalman update on estimates?
        adjacency_matrix = utilities.share_samples(agents, map, param.sens_range, adjacency_matrix)

    # Check if two agents are too close to each other (just for DEBUG)
    for i, agent in enumerate(agents):
        for j, other_agent in enumerate(agents):
            if i != j and np.linalg.norm(agent.x - other_agent.x) <= min_safe_range:
                violations.append((i, j))
                print(f"Agents {i} and {j} are too close at step {step}")

    for agent in agents:
        # Collision check
        if map[agent.x[0].astype(int), agent.x[1].astype(int)] == 1:
            print(f"Agent {agent.id} collided with an obstacle")
            raise ValueError("Agent collided with an obstacle")

        agent.local_cooling = np.zeros_like(goal_density)

        fov_edges_moved = utilities.rotate_and_translate(tmp, agent.x, agent.theta)
        fov_edges_clipped = utilities.clip_polygon_no_convex(agent.x, fov_edges_moved, occ_map, closed_map)
        fov_points = utilities.insidepolygon(fov_edges_clipped).astype(int)

        # Delete points outside the box
        fov_probs = utilities.fov_coverage_block(fov_points, fov_edges_clipped, param.fov_depth)
        # fov_probs = utilities.min_max_normalize(fov_probs)

        # Clear the current buffer slot and add new coverage
        current_coverage = np.zeros_like(goal_density)
        current_coverage[fov_points[:, 0], fov_points[:, 1]] = fov_probs
        
        # Store in circular buffer
        agent.coverage_buffer[agent.buffer_index] = current_coverage
        agent.buffer_index = (agent.buffer_index + 1) % buffer_size
        if agent.buffer_index == 0:
            agent.buffer_full = True
        
        # Compute total coverage density from buffer
        if agent.buffer_full:
            agent.coverage_density = np.sum(agent.coverage_buffer, axis=0)
        else:
            agent.coverage_density = np.sum(agent.coverage_buffer[:agent.buffer_index], axis=0)

        coverage_density = agent.coverage_density

        coverage_density = agent.coverage_density
        # Incorporate the shared coverage density from neighbors
        if len(agent.neighbors) > 0:
            for neighbor_id in agent.neighbors:
                neighbor_agent = agents[neighbor_id]
                # Get the local shared block for the neighbor
                x_idx = int(neighbor_agent.x[0] // param.dx)
                y_idx = int(neighbor_agent.x[1] // param.dx)

                x_start = max(0, x_idx - param.sens_range // 2)
                x_end = min(map.shape[0], x_idx + param.sens_range // 2 + 1)
                y_start = max(0, y_idx - param.sens_range // 2)
                y_end = min(map.shape[1], y_idx + param.sens_range // 2 + 1)

                # Update the local cooling with the neighbor's coverage density
                coverage_density[x_start:x_end, y_start:y_end] += neighbor_agent.coverage_density[x_start:x_end, y_start:y_end]

        agent.fov_edges = fov_edges_moved

        """ Goal density sampling """
        y = goal_density[fov_points[:, 0], fov_points[:, 1]]  # + np.random.normal(0, noise, len(fov_points))
        agent.samples = np.hstack((fov_points, step * np.ones((fov_points.shape[0], 1), dtype=int).reshape(-1, 1), y.reshape(-1, 1)))

        agent.testSet = np.empty((0, 4)) # Reset the test set for each step
        pooled_samples = utilities.max_pooling(agent.samples, 5)
        agent.testSet = np.unique(np.vstack((agent.testSet, pooled_samples)), axis=0)

        # Share the samples with neighbors no pooling here
        if agent.neighbors:
            all_samples = np.vstack([agents[neighbor_id].dataset for neighbor_id in agent.neighbors])
            agent.testSet = np.unique(np.vstack([agent.testSet, all_samples]), axis=0)

        new_samples = agent.testSet
        # Mantovani et al. 2024 ======================================================================
        if step > 0:
            # Filter samples based on standard deviation threshold
            std_test = agent.std[agent.testSet[:, 0].astype(int), agent.testSet[:, 1].astype(int)]
            new_samples = agent.testSet[std_test > takeTh]
            # Print in green color the number of added samples
            if len(new_samples) > 0:
                print(f"\033[92mAdded {len(new_samples)} samples to agent {agent.id}'s dataset.\033[0m")
        
        agent.dataset = np.vstack((agent.dataset, new_samples))

        # Only proceed if there are samples to process
        if len(new_samples) != 0:
            gpr.fit(agent.dataset[:, :2], agent.dataset[:, 3])

        # agent.samples = np.vstack((agent.samples, subset))
        agent.dataset = agent.dataset[np.argsort(agent.dataset[:, 2])]

        # Compute decay matrices
        D, d = utilities.compute_spatio_decay_matrix(agent.dataset[:, :2], spatial_decay, agent.x)
        T, t = utilities.compute_temporal_decay_matrix(agent.dataset[:, 2], step, temporal_decay)

        # Compute combo density and update agent state
        agent.combo_density, agent.mu, agent.std = utilities.compute_combo(
            agent.dataset[:, :2],
            agent.dataset[:, 3],
            grid,
            map,
            gpr.kernel_,
            D,
            T,
            d,
            t
        )

        print(f"Agent {agent.id} dataset: {len(agent.dataset)}")
        # Pratissoli et al. 2025 =====================================================================
        if step > 0:
            # Keep only the samples with uncertainty low enough
            std_test = agent.std[agent.dataset[:, 0].astype(int), agent.dataset[:, 1].astype(int)]
            agent.dataset = agent.dataset[std_test < removeTh]
            agent.dataset = agent.dataset[np.argsort(agent.dataset[:, 2])]
            # Print in red color the number of samples removed
            if len(std_test[std_test >= removeTh]) > 0:
                print(f"\033[91mRemoved {len(std_test[std_test >= removeTh])} samples from agent {agent.id}'s dataset.\033[0m")

        if step == 0:
            agent.heat = np.array(utilities.normalize_mat(agent.combo_density))
        
        diff = utilities.normalize_mat(agent.combo_density) - utilities.normalize_mat(coverage_density)

        source = np.maximum(diff, 0) ** 2 # Eq. 13 - Source term
        source = np.where(map == 0, source, 0)
        agent.source = utilities.normalize_mat(source) * param.area # Eq. 14 - Source term scaled

        ergodic_metric[step, agent.id] = np.linalg.norm(
            goal_density - utilities.normalize_mat(coverage_density)
        )

        agent.heat = utilities.update_heat_optimized(agent.heat,
                                    agent.source,
                                    map,
                                    agent.local_cooling,
                                    param.dt,
                                    param.alpha,
                                    param.source_strength,
                                    param.beta,
                                    param.local_cooling,
                                    param.dx).astype(np.float32)

        gradient_y, gradient_x = np.gradient(agent.heat.T, 1, 1)

        # Only normalize if gradients are extremely large
        grad_mag = np.sqrt(np.mean(gradient_x**2) + np.mean(gradient_y**2))

        gradient_x /= grad_mag
        gradient_y /= grad_mag

        # Update the agent
        agent.grad = utilities.calculate_gradient_map_v2(
            param, agent, gradient_x, gradient_y, map
        )

        # Reduce collision avoidance aggressiveness
        if len(agent.neighbors) > 0:
            for neighbor in agent.neighbors:
                neighbor_agent = agents[neighbor]
                q_ij = np.linalg.norm(agent.x - neighbor_agent.x)**2
                p = 4 * (param.sens_range**2 - min_safe_range**2) * (q_ij - param.sens_range**2) * (agent.x - neighbor_agent.x) / \
                    (q_ij - min_safe_range**2)**3
                # Control law, we want to move away from the neighbor
                agent.grad -= p

        k_target = 2  # Reduce from 1.0 to make movement less aggressive
        v_target = k_target * agent.grad  # Use gradient directly

        theta_target = np.atan2(agent.grad[1], agent.grad[0])

        agent.track_velocity_and_heading(v_target, theta_target, penalize_lateral=True)

        if save_debug and agent.id == 0:
            debug_mu[step] = agent.mu.reshape(map.shape)
            debug_std[step] = agent.std.reshape(map.shape)
            debug_combo_density[step] = agent.combo_density.reshape(map.shape)
    
    if step % 1 == 0 and PLOT:
        ax.clear()
        ax.set_aspect('equal')
        # Plot the map and the ground truth goal density
        ax.contourf(grid_x, grid_y, agents[0].combo_density, cmap='RdPu', levels=10)
        ax.pcolormesh(grid_x, grid_y, np.where(map == 0, np.nan, map), cmap='gray')
        # Plot the agent position and fov
        for agent in agents:
            # Plot the agents' paths
            ax.plot(agent.x_hist[:, 0], agent.x_hist[:, 1], color=f'C{agent.id}', alpha=0.5, lw=2)
            # Plot the agents' current positions
            ax.scatter(agent.x[0], agent.x[1], c=f'C{agent.id}', s=100, marker='x', label=f'Agent {agent.id} End')
            # Plot the heading
            ax.quiver(agent.x[0], agent.x[1], np.cos(agent.theta), np.sin(agent.theta), scale=3, scale_units='inches', color=f'C{agent.id}')
            # Circle the min safe range
            circle = plt.Circle((agent.x[0], agent.x[1]), min_safe_range, color=f'C{agent.id}', fill=False, linestyle='--', label=f'Min Safe Range Agent {agent.id}')
            ax.add_artist(circle)
            # Circle the sens range
            circle = plt.Circle((agent.x[0], agent.x[1]), param.sens_range, color=f'C{agent.id}', fill=False, linestyle=':', label=f'Sens Range Agent {agent.id}')
            ax.add_artist(circle)
            # Plot the fov center
            ax.scatter(agent.fov_center[0], agent.fov_center[1], c=f'C{agent.id}', s=50, marker='o', label=f'FOV Center Agent {agent.id}')

            # Add a coverage block around the fov center
            x, y = agent.fov_center.astype(int)
            x_indices, x_start_kernel, num_kernel_x = utilities.clamp_kernel_1d(x, 0, param.width, param.kernel_size)
            y_indices, y_start_kernel, num_kernel_y = utilities.clamp_kernel_1d(y, 0, param.height, param.kernel_size)
            agent.cov_dens[x_indices, y_indices] += coverage_block[
                        x_start_kernel : x_start_kernel + num_kernel_x,
                        y_start_kernel : y_start_kernel + num_kernel_y,
                    ]
            
        # Plot the coverage density
        ax.pcolormesh(grid_x, grid_y, agent.cov_dens, cmap='viridis', alpha=0.5)

        # Plot a dashed line if two agents are neighbors using the adjacency matrix
        for i in range(param.nbAgents):
            for j in range(param.nbAgents):
                if adjacency_matrix[i, j] == 1 and i != j:
                    ax.plot(
                        [agents[i].x[0], agents[j].x[0]],
                        [agents[i].x[1], agents[j].x[1]],
                        color='black',
                        linestyle='--',
                        alpha=0.5
                    )
        ax.set_title(f'Step {step + 1}/{param.nbDataPoints}')

        plt.pause(0.01)

# Smooth the ergodic metric
import scipy.ndimage as ndimage
ergodic_metric = ndimage.gaussian_filter1d(ergodic_metric, sigma=4, axis=0)

if save_debug:
    np.save('video_ergodic_metric.npy', ergodic_metric)
    # Save the agents' data
    np.save('video_agents_data.npy', [agent.x_hist for agent in agents])
    # Save the debug data
    np.save('video_mu.npy', debug_mu)
    np.save('video_std.npy', debug_std)
    np.save('video_combo_density.npy', debug_combo_density)

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(111)
ax.set_aspect('equal')
# Plot the map and the ground truth goal density
ax.contourf(grid_x, grid_y, goal_density, cmap='RdPu', levels=10)
ax.pcolormesh(grid_x, grid_y, np.where(map == 0, np.nan, map), cmap='gray')
for agent in agents:
    # Plot initial position
    ax.scatter(agent.x_hist[0, 0], agent.x_hist[0, 1], c=f'C{agent.id}', s=100, marker='o', label=f'Agent {agent.id} Start')
    # Plot the agents' paths
    ax.plot(agent.x_hist[:, 0], agent.x_hist[:, 1], color=f'C{agent.id}', alpha=0.5, lw=2)
    # Plot the agents' current positions
    ax.scatter(agent.x_hist[-1, 0], agent.x_hist[-1, 1], c=f'C{agent.id}', s=100, marker='x', label=f'Agent {agent.id} End')
    # Plot the heading
    ax.quiver(agent.x_hist[-1, 0], agent.x_hist[-1, 1], np.cos(agent.x_hist[-1, 2]), np.sin(agent.x_hist[-1, 2]), scale=2, scale_units='inches', color=f'C{agent.id}')
    # Draw the FOV
    # Circle the min safe range
    circle = plt.Circle((agent.x_hist[-1, 0], agent.x_hist[-1, 1]), min_safe_range, color=f'C{agent.id}', fill=False, linestyle='--', label=f'Min Safe Range Agent {agent.id}')
    ax.add_artist(circle)
    # Circle the sens range
    circle = plt.Circle((agent.x_hist[-1, 0], agent.x_hist[-1, 1]), param.sens_range, color=f'C{agent.id}', fill=False, linestyle=':', label=f'Sens Range Agent {agent.id}')
    ax.add_artist(circle)

ax.set_title('Final Agent Positions and Paths')
# Scatter violations
ax.scatter(
    [agents[i].x_hist[-1, 0] for i, j in violations],
    [agents[j].x_hist[-1, 1] for i, j in violations],
    c='red', s=100, marker='x', label='Violations'
)

plt.legend()
plt.show(block=True)

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(111)
# Plot the ergodic metric for each agent
for i in range(param.nbAgents):
    ax.plot(np.arange(param.nbDataPoints), ergodic_metric[:, i], label=f'Agent {i}', alpha=0.7)
ax.set_xlabel('Time Step')
ax.set_ylabel('Ergodic Metric Value')
ax.set_title('Ergodic Metric Over Time')
ax.grid()
ax.legend()
plt.show(block=True)

# # Plot the ergodic metric
# plt.figure(figsize=(10, 5))
# plt.plot(np.arange(param.nbDataPoints), ergodic_metric.sum(axis=1), label='Ergodic Metric', color='blue')
# plt.xlabel('Time Step')
# plt.ylabel('Ergodic Metric Value')
# plt.title('Ergodic Metric Over Time')
# plt.grid()
# plt.legend()
# plt.show(block=True)

# Plot mean and uncertainty of the GPR predictions
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(131)
ax.set_aspect('equal')
ax.contourf(grid_x, grid_y, goal_density, cmap='RdPu', levels=10)
ax.pcolormesh(grid_x, grid_y, np.where(map == 0, np.nan, map), cmap='gray')
ax.set_title('Ground Truth Goal Density')

ax = fig.add_subplot(132)
ax.set_aspect('equal')
ax.contourf(grid_x, grid_y, agents[0].mu.reshape(map.shape), cmap='RdPu', levels=10)
ax.pcolormesh(grid_x, grid_y, np.where(map == 0, np.nan, map), cmap='gray')
ax.set_title('GPR Mean Prediction')

ax = fig.add_subplot(133)
ax.set_aspect('equal')
ax.contourf(grid_x, grid_y, agents[0].std.reshape(map.shape), cmap='RdPu', levels=10)
ax.pcolormesh(grid_x, grid_y, np.where(map == 0, np.nan, map), cmap='gray')
ax.set_title('GPR Uncertainty (Std Dev)')

plt.show(block=True)

# Create video of the agents moving
import matplotlib.animation as animation
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(111)
ax.set_aspect('equal')
def update(frame):
    print(f"Updating frame {frame + 1}/{param.nbDataPoints}")
    ax.clear()
    ax.set_aspect('equal')
    ax.contourf(grid_x, grid_y, goal_density, cmap='RdPu', levels=10)
    ax.pcolormesh(grid_x, grid_y, np.where(map == 0, np.nan, map), cmap='gray')

    for agent in agents:
        # Plot the agents' paths
        ax.plot(agent.x_hist[:frame+1, 0], agent.x_hist[:frame+1, 1], color=f'C{agent.id}', alpha=0.5, lw=2)
        # Plot the agents' current positions
        ax.scatter(agent.x_hist[frame, 0], agent.x_hist[frame, 1], c=f'C{agent.id}', s=100, marker='x', label=f'Agent {agent.id} End')
        # Plot the heading
        ax.quiver(agent.x_hist[frame, 0], agent.x_hist[frame, 1], np.cos(agent.x_hist[frame, 2]), np.sin(agent.x_hist[frame, 2]), scale=3, scale_units='inches', color=f'C{agent.id}')
        # Draw the FOV
        # fov_edges_clipped = utilities.clip_polygon_no_convex(agent.x_hist[frame], agent.fov_edges, occ_map, closed_map=True)
        # ax.fill(fov_edges_clipped[:, 0], fov_edges_clipped[:, 1], color=f'C{agent.id}', alpha=0.3)

    ax.set_title(f'Frame {frame + 1}')

    return ax,

ani = animation.FuncAnimation(fig, update, frames=np.arange(param.nbDataPoints, step=10), repeat=False)
date = np.datetime64('now').astype(str).replace(':', '-').replace(' ', '_')
ani.save('agents_simulation_' + date + '.mp4', writer='ffmpeg', fps=30)