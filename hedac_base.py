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
param_file = os.path.dirname(os.path.realpath(__file__)) + '/params/' + 'hedac_base.json'

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
                     [30, 30]])

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

means = np.array([[40, 8], [20, 45], [8, 8], [40, 40]])
cov = np.array([[[10, 0], [0, 10]], [[10, 0], [0, 10]], [[10, 0], [0, 10]], [[10, 0], [0, 10]]])
density_map = utilities.gauss_pdf(grid, means[1], cov[1])
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

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(111)
ax.set_aspect('equal')

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
=========
Main Loop
=========
"""
ergodic_metric = np.zeros((param.nbDataPoints), dtype=float)

PLOT = param.plot

for step in range(param.nbDataPoints):
    if step // 10:
        print(f"Step - {step}")

    if step == 5000:
        means = np.array([[40, 8], [20, 45], [8, 8], [40, 40]])
        cov = np.array([[[10, 0], [0, 10]], [[10, 0], [0, 10]], [[10, 0], [0, 10]], [[10, 0], [0, 10]]])
        density_map = utilities.gauss_pdf(grid, means[2], cov[2])
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

    local_cooling = np.zeros((param.height, param.width))
    for agent in agents:
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
        try:
            # Update the local cooling and coverage density
            local_cooling[x_indices, y_indices] += coverage_block[
                x_start_kernel : x_start_kernel + num_kernel_dx,
                y_start_kernel : y_start_kernel + num_kernel_dy,
            ]
            coverage_density[x_indices, y_indices] += coverage_block[
                x_start_kernel : x_start_kernel + num_kernel_dx,
                y_start_kernel : y_start_kernel + num_kernel_dy,
            ]  # Eq. 3 - Coverage density
        except ValueError as e:
            print(f"ValueError at step {step} for agent {agent.id}: {e}")
            continue

        local_cooling = utilities.normalize_mat(local_cooling) * param.area # Eq. 15 - Local cooling normalized


    if step == 0:
        heat = np.array(utilities.normalize_mat(coverage_density))

    diff = utilities.normalize_mat(goal_density) - utilities.normalize_mat(coverage_density)

    source = np.maximum(diff, 0) ** 2 # Eq. 13 - Source term
    source = np.where(map == 0, source, 0)
    source = utilities.normalize_mat(source) * param.area # Eq. 14 - Source term scaled

    ergodic_metric[step] = np.linalg.norm(
        goal_density - utilities.normalize_mat(coverage_density)
    )

    heat = utilities.update_heat_optimized(heat,
                                source,
                                map,
                                local_cooling,
                                param.dt,
                                param.alpha,
                                param.source_strength,
                                param.beta,
                                param.local_cooling,
                                param.dx).astype(np.float32)

    gradient_y, gradient_x = np.gradient(heat.T, 1, 1)


    # Only normalize if gradients are extremely large
    grad_mag = np.sqrt(np.mean(gradient_x**2) + np.mean(gradient_y**2))

    gradient_x /= grad_mag
    gradient_y /= grad_mag

    for agent in agents:

        # Update the agent
        grad = utilities.calculate_gradient_map(
            param, agent, gradient_x, gradient_y, map
        )

        # Use gradient directly without excessive scaling
        k_target = 1  # Reduce from 1.0 to make movement less aggressive
        v_target = k_target * grad  # Use gradient directly

        theta_target = np.atan2(grad[1], grad[0])

        agent.track_velocity_and_heading(v_target, theta_target, penalize_lateral=False)

    if step % 10 == 0 and PLOT:
        ax.clear()
        ax.set_aspect('equal')
        # Plot the map and the ground truth goal density
        ax.contourf(grid_x, grid_y, goal_density, cmap='RdPu', levels=10)
        ax.pcolormesh(grid_x, grid_y, np.where(map == 0, np.nan, map), cmap='gray')
        # Plot the agent position and fov
        for agent in agents:
            # Plot the agents' paths
            ax.plot(agent.x_hist[:, 0], agent.x_hist[:, 1], color=f'C{agent.id}', alpha=0.5, lw=2)
            # Plot the agents' current positions
            ax.scatter(agent.x[0], agent.x[1], c=f'C{agent.id}', s=100, marker='x', label=f'Agent {agent.id} End')
            # Plot the kernel block
            adjusted_position = agent.x
            x, y = adjusted_position.astype(int)
            x_indices, x_start_kernel, num_kernel_dx = utilities.clamp_kernel_1d(
                x, 0, param.width, param.kernel_size
            )
            y_indices, y_start_kernel, num_kernel_dy = utilities.clamp_kernel_1d(
                y, 0, param.height, param.kernel_size
            )
            # ax.plot(x_indices, y_indices, color=f'C{agent.id}', alpha=0.5, lw=2)

        ax.set_title(f'Step {step + 1}/{param.nbDataPoints}')

        plt.pause(0.01)

# Smooth the ergodic metric
import scipy.ndimage as ndimage
ergodic_metric = ndimage.gaussian_filter1d(ergodic_metric, sigma=4, axis=0)


# np.save('ergodic_metric_base_change_3.npy', ergodic_metric)
# # Save the agents' data
# np.save('agents_data_base_change_3.npy', [agent.x_hist for agent in agents])

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

ax.set_title('Final Agent Positions and Paths')

plt.legend()
plt.show(block=True)

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(111)
# Plot the ergodic metric for each agent
ax.plot(np.arange(param.nbDataPoints), ergodic_metric, alpha=0.7)
ax.set_xlabel('Time Step')
ax.set_ylabel('Ergodic Metric Value')
ax.set_title('Ergodic Metric Over Time')
ax.grid()
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