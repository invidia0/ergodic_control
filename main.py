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
                     [30, 30],
                     [12, 15]])

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
_, density_map = utilities.generate_gmm_on_map(map,
                                             free_cells,
                                             param.nbGaussian,
                                             param.nbParticles,
                                             param.nbVar,
                                             random_state=param.random_seed)
# means = np.array([[40, 40], [20, 45], [8, 8]])
# cov = np.array([[[20, 0], [0, 20]], [[10, 0], [0, 10]], [[10, 0], [0, 10]]])
# density_map = utilities.gauss_pdf(grid, means[0], cov[0]) #+ \
#             # utilities.gauss_pdf(grid, means[1], cov[1])
#                 # utilities.gauss_pdf(grid, means[2], cov[2])

# norm_density_map = utilities.min_max_normalize(density_map).reshape(map.shape)
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
noise = 0.005
# kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e5)) # + WhiteKernel(noise_level=noise, noise_level_bounds=(1e-5, 1e1))
# gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=False, alpha=1e-5)
# # subset = np.empty((0, 3))

# std_pred_test = np.zeros_like(goal_density)

kernel = (
    C(1.0, constant_value_bounds=(1e-3, 1e3))
    * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))  # space
    + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e1))
)

gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-5, normalize_y=False)

gpr.kernel_ = kernel
# # Precompute the GPR for faster simulations
# preSamplesN = 500
# preSamples = np.random.randint(0, len(free_cells), preSamplesN)
# preSamples = np.hstack((free_cells[preSamples], goal_density[free_cells[preSamples][:, 0], free_cells[preSamples][:, 1]].reshape(-1, 1)))
# preSamples = np.unique(preSamples, axis=0, return_index=False)

# gpr.fit(preSamples[:, :2], preSamples[:, 2])

# mu, std = gpr.predict(grid, return_std=True)

# fig = plt.figure(figsize=(12, 5))
# ax = fig.add_subplot(131)
# ax.set_aspect('equal')
# # Plot the mean
# ax.contourf(grid_x, grid_y, mu.reshape(map.shape), cmap='RdPu', levels=10)
# ax = fig.add_subplot(132)
# ax.set_aspect('equal')
# # Plot the std
# ax.contourf(grid_x, grid_y, std.reshape(map.shape), cmap='binary', levels=10)
# ax = fig.add_subplot(133)
# ax.set_aspect('equal')
# # Plot the goal density
# ax.contourf(grid_x, grid_y, goal_density, cmap='RdPu', levels=10)
# plt.show(block=True)

"""
=========
Main Loop
=========
"""
for agent in agents:
    agent.heat = np.empty_like(goal_density)
    agent.samples = np.empty((0, 4))
    agent.subset = np.empty((0, 4))
    agent.coverage_density_hist = np.zeros((50, 2, param.nbDataPoints), dtype=int)
    agent.coverage_density_prob_hist = np.zeros((50, param.nbDataPoints), dtype=float)

    agent.neighbors = []
    agent.last_neighbors = []
    agent.local_cooling = np.zeros_like(goal_density)
    agent.coverage_density = np.zeros_like(goal_density)

    # Stack the precomputed samples
    # agent.subset = np.vstack((agent.subset, preSamples))

adjacency_matrix = np.eye(param.nbAgents)

# Chunk processing for speed
chunk_size = param.nbDataPoints // 10
num_chunks = param.nbDataPoints // chunk_size

violations = []
min_safe_range = 1

plt.close('all')  # Close all previous plots
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(111)

spatial_decay = 1000
temporal_decay = 1000
# Start time

for step in range(param.nbDataPoints):
    if step // 10:
        print(f"Step - {step}")

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

        agent.coverage_density[fov_points[:, 0], fov_points[:, 1]] += fov_probs

        agent.fov_edges = fov_edges_moved

        """ Goal density sampling """
        y = goal_density[fov_points[:, 0], fov_points[:, 1]]  # + np.random.normal(0, noise, len(fov_points))
        dataset = np.hstack((fov_points, time.time() * np.ones((fov_points.shape[0], 1), dtype=int).reshape(-1, 1), y.reshape(-1, 1)))

        agent.samples = np.vstack((agent.samples, dataset))
        agent.subset = agent.subset[np.argsort(agent.subset[:, 2])]

        # Mantovani et al. 2024 ======================================================================
        if step > 0:
            # Filter samples based on standard deviation threshold
            std_test = agent.std[agent.samples[:, 0].astype(int), agent.samples[:, 1].astype(int)]
            agent.samples = agent.samples[std_test > 0.75]

        # Only proceed if there are samples to process
        if len(agent.samples) != 0:
            pooled_samples = utilities.max_pooling(agent.samples, 5)
            agent.subset = np.unique(np.vstack((agent.subset, pooled_samples)), axis=0)

            if step > 0:
                gpr.fit(agent.subset[:, :2], agent.subset[:, 3])

        # Compute decay matrices
        D, d = utilities.compute_spatio_decay_matrix(agent.subset[:, :2], spatial_decay, agent.x)
        T, t = utilities.compute_temporal_decay_matrix(agent.subset[:, 2], time.time(), temporal_decay)

        # Compute combo density and update agent state
        agent.combo_density, agent.mu, agent.std = utilities.compute_combo(
            agent.subset[:, :2],
            agent.subset[:, 3],
            grid,
            map,
            gpr.kernel_,
            D,
            T,
            d,
            t
        )

        # Clear low standard deviation values
        # agent.std[agent.std < 0.3] = 0

        print(f"Agent {agent.id} subset: {len(agent.subset)}")
        # Pratissoli et al. 2025 =====================================================================
        if step > 0:
            # Keep only the samples with uncertainty low enough
            std_test = agent.std[agent.subset[:, 0].astype(int), agent.subset[:, 1].astype(int)]
            agent.subset = agent.subset[std_test < 0.75]
            agent.subset = agent.subset[np.argsort(agent.subset[:, 2])]


        if step == 0:
            agent.heat = np.array(utilities.normalize_mat(agent.combo_density))

        diff = utilities.normalize_mat(agent.combo_density) - utilities.normalize_mat(agent.coverage_density)

        source = np.maximum(diff, 0) ** 2 # Eq. 13 - Source term
        source = np.where(map == 0, source, 0)
        agent.source = utilities.normalize_mat(source) * param.area # Eq. 14 - Source term scaled

        # ergodic_metric[step, agent.id] = np.linalg.norm(agent.source) * param.dt # Eq. 15 - Ergodic metric

        current_heat = utilities.update_heat_optimized(
            agent.heat,
            agent.source,
            map,
            agent.local_cooling,
            param.dt,
            param.alpha,
            param.source_strength,
            param.beta,
            param.local_cooling,
            param.dx
        )

        agent.heat = current_heat.astype(np.float32)

        gradient_y, gradient_x = np.gradient(agent.heat.T, 1, 1)

        gradient_x /= np.linalg.norm(gradient_x) + 1e-6
        gradient_y /= np.linalg.norm(gradient_y) + 1e-6

        # Update the agent
        agent.grad = utilities.calculate_gradient_map(
            param, agent, gradient_x, gradient_y, map
        )

        if len(agent.neighbors) > 0:
            for neighbor in agent.neighbors:
                neighbor_agent = agents[neighbor]
                q_ij = np.linalg.norm(agent.x - neighbor_agent.x)**2
                p = 4 * (param.sens_range**2 - min_safe_range**2) * (q_ij - param.sens_range**2) * (agent.x - neighbor_agent.x) / \
                    (q_ij - min_safe_range**2)**3
                # Control law, we want to move away from the neighbor
                agent.grad -= p

        # v_target and theta_target
        k_target = 1
        v_target = k_target * np.array([agent.grad[0], agent.grad[1]])

        theta_target = np.atan2(agent.grad[1], agent.grad[0])

        agent.track_velocity_and_heading(v_target, theta_target, penalize_lateral=True)

        ax.clear()
        ax.set_aspect('equal')
        # Plot the map and the ground truth goal density
        ax.contourf(grid_x, grid_y, agent.std, cmap='RdPu', levels=10)
        ax.pcolormesh(grid_x, grid_y, np.where(map == 0, np.nan, map), cmap='gray')
        # Plot the agent position and fov
        ax.scatter(agent.x[0], agent.x[1], c=f'C{agent.id}', s=100, marker='o', label=f'Agent {agent.id} Start')
        # Plot the FOV
        fov_edges_clipped = utilities.clip_polygon_no_convex(agent.x, agent.fov_edges, occ_map, closed_map=True)
        ax.fill(fov_edges_clipped[:, 0], fov_edges_clipped[:, 1], color=f'C{agent.id}', alpha=0.3, label=f'FOV Agent {agent.id}')
        plt.pause(0.01)


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
    # fov_edges_clipped = utilities.clip_polygon_no_convex(agent.x, agent.fov_edges, occ_map, closed_map=True)
    # ax.fill(fov_edges_clipped[:, 0], fov_edges_clipped[:, 1], color=f'C{agent.id}', alpha=0.3, label=f'FOV Agent {agent.id}')
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