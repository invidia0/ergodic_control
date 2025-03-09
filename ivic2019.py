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
# Create a map  with just the border
border_map = np.ones_like(map)
border_map[1:-1, 1:-1] = 0
map = border_map
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
param_file = os.path.dirname(os.path.realpath(__file__)) + '/params/' + 'ivic2019.json'

with open(param_file, 'r') as f:
    param_data = json.load(f)

param = lambda: None

for key, value in param_data.items():
    setattr(param, key, value)

# param.max_dtheta = np.pi / param.max_dtheta # Maximum angular velocity (rad/s)
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
                [30, 15],
                [10.5, 10],
                [45, 45],])

free_cells = np.array(np.where(map == 0)).T

x0_array = free_cells[np.random.choice(free_cells.shape[0], param.nbAgents, replace=False)]

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
_, density_map = utilities.generate_gmm_on_map(map,
                                             free_cells,
                                             param.nbGaussian,
                                             param.nbParticles,
                                             param.nbVar,
                                             random_state=param.random_seed)

means = np.array([[40, 40], [20, 45], [8, 8]])
cov = np.array([[[20, 0], [0, 20]], [[10, 0], [0, 10]], [[10, 0], [0, 10]]])
density_map = utilities.gauss_pdf(grid, means[0], cov[0]) #+ \
            # utilities.gauss_pdf(grid, means[1], cov[1]) 
                # utilities.gauss_pdf(grid, means[2], cov[2])

norm_density_map = utilities.min_max_normalize(density_map).reshape(map.shape)

# Compute the area of the map
cell_area = param.dx * param.dx
param.area = np.sum(map == 0) * cell_area

goal_density = np.zeros_like(map)
goal_density[free_cells[:, 0], free_cells[:, 1]] = norm_density_map[free_cells[:, 0], free_cells[:, 1]]


"""
===============================
Initialize heat equation related parameters
===============================
"""
param.width = map.shape[0]
param.height = map.shape[1]

param.beta = param.beta / param.area # Eq. 17 - Beta normalized 
param.local_cooling = param.local_cooling / param.area # Eq. 16 - Local cooling normalized

# goal_density = np.ones_like(map) # The goal density
local_cooling = np.zeros_like(goal_density) # The local cooling
coverage_density = np.zeros_like(goal_density) # The coverage density
heat = np.array(goal_density, dtype=np.float32) # The heat map

for agent in agents:
    tmp = utilities.init_fov(param.fov_deg, param.fov_depth)
    agent.fov_edges = utilities.rotate_and_translate(tmp, agent.x, agent.theta)

coverage_block = utilities.agent_block(param.nbVar, param.min_kernel_val, param.agent_radius)
param.kernel_size = coverage_block.shape[0]

ergodic_metric = np.zeros((param.nbDataPoints))

"""
Debug & animation
"""
grad = 0
if param.save_data:
    data_storage_path = os.path.join(os.getcwd(), '_datastorage/')
    _heat_hist = np.memmap(data_storage_path + "heat_hist.dat", dtype=np.float64, mode='w+', 
                        shape=(goal_density.shape[0], goal_density.shape[1], param.nbDataPoints))
    _source_hist = np.memmap(data_storage_path + "source_hist.dat", dtype=np.float64, mode='w+', 
                            shape=(goal_density.shape[0], goal_density.shape[1], param.nbDataPoints))
    _path_hist = np.memmap(data_storage_path + "path_hist.dat", dtype=np.float64, mode='w+', 
                            shape=(3, param.nbDataPoints, param.nbAgents))

"""
===============================
Main Loop
===============================
"""
# Create a matrix with ones on the diagonal
adjacency_matrix = np.eye(param.nbAgents)
obstacles = np.zeros((param.nbAgents, param.nbAgents))

# Example of chunked processing for memmap arrays
chunk_size = param.nbDataPoints // 10
num_chunks = param.nbDataPoints // chunk_size

for agent in agents:
    agent.last_heading = agent.theta
    agent.last_position = agent.x
    agent.heat = np.empty_like(goal_density)
    agent.samples = np.empty((0, 3))
    agent.pooled_dataset = np.empty((0, 3))
    agent.subset = np.empty((0, 3))
    agent.local_cooling = np.zeros_like(goal_density)
    agent.angular_error = 0

for chunk in range(num_chunks):
    start_idx = chunk * chunk_size
    end_idx = min((chunk + 1) * chunk_size, param.nbDataPoints)
    print(f"Processing chunk {chunk + 1}/{num_chunks} (steps {start_idx} to {end_idx})")
        
    for t in range(start_idx, end_idx):
        print(f'\nStep: {t}/{param.nbDataPoints}')

        local_cooling = np.zeros_like(goal_density)
        coverage = np.zeros_like(goal_density)
        for agent in agents:
            x, y = agent.x.astype(int)
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

            coverage_density[x_indices, y_indices] += np.pow(1/param.agent_radius, -2) * utilities.normalize_mat(coverage_block[
                x_start_kernel : x_start_kernel + num_kernel_dx,
                y_start_kernel : y_start_kernel + num_kernel_dy,
            ]) # Eq. 3 - Coverage density
            # # coverage += utilities.norma)lize_mat(coverage_density)

        coverage = coverage_density / (param.nbAgents * t+1)
        # coverage = utilities.normalize_mat(coverage_density) 

        # source = goal_density * np.exp(-coverage_density) # Eq. 15 - Source ter
        # diff = utilities.min_max_normalize(goal_density) - utilities.min_max_normalize(coverage_density)
        diff = utilities.normalize_mat(goal_density) - coverage # Eq. 6 - Difference between the goal density and the coverage density
        source = np.maximum(diff, 0) ** 2 # Eq. 13 - Source term
        source = np.where(map == 0, source, 0)
        _em_diff = np.sum(source/np.linalg.norm(source)) * param.dx * param.dx
        ergodic_metric[t] = _em_diff

        local_cooling = utilities.normalize_mat(local_cooling) * param.area # Eq. 16 - Local cooling scaled
        source = utilities.normalize_mat(source) * param.area # Eq. 14 - Source term scaled

        # _em_diff = utilities.min_max_normalize(source)
        # _em_diff = source / np.linalg.norm(source)

        # current_heat = np.zeros((param.width, param.height))
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

        gradient_x /= np.linalg.norm(gradient_x)
        gradient_y /= np.linalg.norm(gradient_y)

        for agent in agents:
            # Store the last position and heading
            agent.last_heading = agent.theta
            agent.last_position = agent.x
            # Update the agent
            agent.grad = utilities.calculate_gradient_map(
                param, agent, gradient_x, gradient_y, map
            )

            u = np.array([agent.grad[0], agent.grad[1]])
            # Compute the desired heading based on the vector field
            desired_theta = np.arctan2(u[1], u[0])  # u = [u_x, u_y]

            # Compute angular error
            angular_error = desired_theta - agent.theta
            angular_error = (angular_error + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-pi, pi]
            k_theta = 1 # Proportional gain
            d_theta = 2 * np.sqrt(k_theta) # Derivative gain
            u_theta = k_theta * angular_error + d_theta * (angular_error - agent.angular_error) / param.dt
            u_theta = np.clip(u_theta, -param.max_ddtheta, param.max_ddtheta)
            agent.angular_error = angular_error
            u = np.array([u[0], u[1], u_theta])
            agent.update(u)

        # Simulations
        # Combine the data collection in a single loop for better performance
        if param.save_data:
            _heat_hist[:, :, t] = heat
            _source_hist[:, :, t] = source
            for idx, agent in enumerate(agents):
                _path_hist[:, t, idx] = agent.x_hist[-1, :]

filepath = os.path.join(os.getcwd(), 'free_env/hedac_sim3_r1/')
hist_array = np.array([agent.x_hist for agent in agents])
np.save(filepath + 'hist_array.npy', hist_array)
np.save(filepath + 'ergodic_metrics.npy', ergodic_metric)

if param.save_data:
    if not os.path.exists(data_storage_path):
        os.makedirs(data_storage_path)

    _heat_hist.flush()
    _source_hist.flush()
    _path_hist.flush()

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

cmaps = ['Blues', 'Reds', 'Purples', 'Greens', 'Oranges']
for agent in agents:
    lines = utilities.colored_line(agent.x_hist[:, 0],
                                    agent.x_hist[:, 1],
                                    time_array,
                                    ax[0],
                                    cmap=cmaps[agent.id],
                                    linewidth=2)

for agent in agents:
    fov_edges_clipped = utilities.clip_polygon_no_convex(agent.x, agent.fov_edges, occ_map, closed_map)
    ax[0].fill(fov_edges_clipped[:, 0], fov_edges_clipped[:, 1], color='blue', alpha=0.3)
    # Heading
    ax[0].quiver(agent.x[0], agent.x[1], np.cos(agent.theta), np.sin(agent.theta), scale = 2, scale_units='inches')
    ax[0].scatter(agent.x_hist[0, 0], agent.x_hist[0, 1], s=100, facecolors='none', edgecolors='green', lw=2)
    ax[0].scatter(agent.x[0], agent.x[1], c='k', s=100, marker='o', zorder=10)
    ax[0].quiver(agent.x[0], agent.x[1], agent.grad[0], agent.grad[1], scale = None, scale_units='inches', color='red')
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
ax[1].contourf(grid_x, grid_y, heat, cmap='Blues', levels=10)
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
# Plot the coverage density and the source
ax[0].cla()
ax[1].cla()
ax[0].set_aspect('equal')
ax[1].set_aspect('equal')
bar = plt.colorbar(ax[0].contourf(grid_x, grid_y, coverage_density, cmap='Greens'), ax=ax[0])
ax[0].set_title("Coverage Density")
ax[1].contourf(grid_x, grid_y, source, cmap='Oranges')
bar = plt.colorbar(ax[1].contourf(grid_x, grid_y, source, cmap='Oranges'), ax=ax[1])
ax[1].set_title("Source")
plt.show()

# Plot the ergodic metric
plt.figure()
plt.plot(ergodic_metric)
plt.title("Ergodic Metric")
plt.show()


# Generate a showcase of the coverage density
while True:
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    for t in range(0, param.nbDataPoints, 10):
        ax.cla()
        goal = ax.pcolormesh(grid_x, grid_y, np.where(map == 0, np.nan, map), cmap='gray')
        ax.contourf(grid_x, grid_y, goal_density, cmap='Greys', levels=10)
        for i, agent in enumerate(agents):
            ax.plot(agent.x_hist[t - 1:t + 1, 0], agent.x_hist[t - 1:t + 1, 1], color=f'C{i}', alpha=1, lw=2, linestyle='--')
            ax.scatter(agent.x_hist[t, 0], agent.x_hist[t, 1], c='black', s=100, marker='o')
            ax.plot(agent.x_hist[:t, 0], agent.x_hist[:t, 1], color=f'C{i}', alpha=1, lw=2)
        
            # Plot the fov
            edges = utilities.rotate_and_translate(tmp, agent.x_hist[t, :2], agent.x_hist[t, 2])
            fov_edges_clipped = utilities.clip_polygon_no_convex(agent.x_hist[t, :2], edges, occ_map, closed_map)
            ax.fill(fov_edges_clipped[:, 0], fov_edges_clipped[:, 1], color='blue', alpha=0.3)
        plt.pause(0.05)
    plt.show()
    if input("Continue? (y/n)") == 'n':
        break