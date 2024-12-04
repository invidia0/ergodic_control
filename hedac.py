#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import warnings
from ergodic_control import models, utilities
import json

warnings.filterwarnings("ignore")

np.random.seed(3) # Seed for reproducibility

"""
===============================
Parameters
===============================
"""
param_file = 'hedac_params.json'

with open(param_file, 'r') as f:
    param_data = json.load(f)

param = lambda: None

for key, value in param_data.items():
    setattr(param, key, value)

param.L = (param.xlim[1] - param.xlim[0]) * 2  # Size of [-xlim(2),xlim(2)]
param.omega = 2 * np.pi / param.L  # Frequency of the domain
param.alpha = np.array(param.alpha) * param.diffusion

fig, ax = plt.subplots(1, 3, figsize=(12, 5))

for a in ax:
    a.set_xlim([0, param.xlim[1]])
    a.set_ylim([0, param.xlim[1]])
    a.set_aspect('equal')
    a.set_xticks([])
    a.set_yticks([])

# 3D Plot of the source term
ax[2] = fig.add_subplot(133, projection='3d')

"""
===============================
Goal Density
===============================
"""
param.Means, param.Covs, param.Weights = utilities.compute_gmm(param)

grids_x, grids_y = np.meshgrid(
    np.linspace(0, param.xlim[1], 100),
    np.linspace(0, param.xlim[1], 100)
)
grids = np.array([grids_x.ravel(), grids_y.ravel()]).T

# Ground truth density function
pdf_gt = utilities.gmm_eval(grids, param.Means, param.Covs, param.Weights).reshape(param.nbResX, param.nbResY)
pdf_gt = np.abs(pdf_gt)
pdf_gt = utilities.normalize_mat(pdf_gt)

# Discretize the GMM into a grid, using Fourier basis functions.
G = utilities.discrete_gmm_original(param).reshape(param.nbResX, param.nbResY)
G = np.abs(G)
G = utilities.normalize_mat(G)
goal_density = G

"""
===============================
Initialize Agents
===============================
"""
agents = []
for i in range(param.nbAgents):
    # x0 = np.random.uniform(0, param.nbResX, 2)
    x0 = np.array([50, 50])
    agent = models.SecondOrderAgent(x=x0, 
                                    max_dx=param.max_dx,
                                    max_ddx=param.max_ddx,
                                    dt=param.dt)
    rgb = np.random.uniform(0, 1, 3)
    agent.color = np.concatenate((rgb, [1.0]))
    agents.append(agent)

"""
===============================
Initialize heat equation related fields
===============================
"""
coverage_arr = np.zeros((param.nbResX, param.nbResY, param.nbDataPoints))
heat_arr = np.zeros((param.nbResX, param.nbResY, param.nbDataPoints))
local_arr = np.zeros((param.nbResX, param.nbResY, param.nbDataPoints))
goal_arr = np.zeros((param.nbResX, param.nbResY, param.nbDataPoints))

param.height, param.width = G.shape

param.area = param.dx * param.width * param.dx * param.height

param.beta = param.beta / param.area # Eq. 17 - Beta normalized 
param.local_cooling = param.local_cooling / param.area # Eq. 16 - Local cooling normalized

coverage_density = np.zeros((param.height, param.width))
heat = np.array(G) # The heat is the goal density

max_diffusion = np.max(param.alpha)
param.dt = min(
    0.2, (param.dx * param.dx) / (4.0 * max_diffusion)
)  # for the stability of implicit integration of Heat Equation
coverage_block = utilities.agent_block(param.eps, param.nbVarX, param.min_kernel_val, param.agent_radius)
# coverage_block = np.ones_like(coverage_block)
# cooling_block = utilities.agent_block(param.nbVarX, param.min_kernel_val, param.cooling_radius)
param.kernel_size = coverage_block.shape[0]

"""
===============================
Main Loop
===============================
"""
for t in range(param.nbDataPoints):
    # cooling of all the agents for a single timestep used for collision avoidance
    local_cooling = np.zeros((param.height, param.width))
    for agent in agents:
        p = agent.x
        adjusted_position = p / param.dx
        col, row = adjusted_position.astype(int)

        row_indices, row_start_kernel, num_kernel_rows = utilities.clamp_kernel_1d(
            row, 0, param.height, param.kernel_size
        )
        col_indices, col_start_kernel, num_kernel_cols = utilities.clamp_kernel_1d(
            col, 0, param.width, param.kernel_size
        )

        # effect of the agent on the coverage density
        coverage_density[row_indices, col_indices] += coverage_block[
            row_start_kernel : row_start_kernel + num_kernel_rows,
            col_start_kernel : col_start_kernel + num_kernel_cols,
        ]

        if param.local_cooling != 0:
            local_cooling[row_indices, col_indices] += coverage_block[
                row_start_kernel : row_start_kernel + num_kernel_rows,
                col_start_kernel : col_start_kernel + num_kernel_cols,
            ]
        local_cooling = utilities.normalize_mat(local_cooling) # Eq. 15 - Local cooling normalized
  
    coverage = utilities.normalize_mat(coverage_density) # Eq. 4 - Coverage density normalized

    diff = G - coverage # Eq. 6 - Difference between the goal density and the coverage density
    source = np.maximum(diff, 0) ** 2 # Eq. 13 - Source term
    source = utilities.normalize_mat(source) * param.area # Eq. 14 - Source term scaled

    current_heat = np.zeros((param.height, param.width))

    """
    Finite difference method to solve the heat equation
    https://en.wikipedia.org/wiki/Five-point_stencil
    https://www.youtube.com/watch?v=YotrBNLFen0
    """
    current_heat[1:-1, 1:-1] = param.dt * (
        (
            + param.alpha[0] * utilities.offset(heat, 1, 0)
            + param.alpha[0] * utilities.offset(heat, -1, 0)
            + param.alpha[1] * utilities.offset(heat, 0, 1)
            + param.alpha[1] * utilities.offset(heat, 0, -1)
            - 4.0 * utilities.offset(heat, 0, 0)
        )
        / (param.dx * param.dx)
        + param.source_strength * utilities.offset(source, 0, 0)
        - param.local_cooling * utilities.offset(local_cooling, 0, 0)
        - param.beta * utilities.offset(heat, 0, 0)
    )  + utilities.offset(heat, 0, 0)

    heat = current_heat.astype(np.float32)
    
    gradient_y, gradient_x = np.gradient(heat, 1, 1) # Gradient of the heat

    for agent in agents:
        grad = utilities.calculate_gradient(
            param,
            agent,
            gradient_x,
            gradient_y,
        )
        agent.update(grad)

    # Plot the agents
    ax[0].cla()
    ax[1].cla()
    ax[2].cla()
    # ax[0].imshow(pdf_gt, cmap='Reds', origin='lower')
    # ax[0].scatter(agents[0].x[0], agents[0].x[1], c='tab:blue', s=100, marker='o')
    # ax[0].plot(agents[0].x_hist[:, 0], agents[0].x_hist[:, 1], c='black', alpha=1, lw=2)

    strength = np.sqrt(gradient_x ** 2 + gradient_y ** 2) # Quiver strength (magnitude of the gradient) for plotting 
    Q = ax[1].quiver(grids_x*100, grids_y*100, gradient_x, gradient_y, strength, cmap='plasma', scale = 100, scale_units='inches')
    ax[1].scatter(agents[0].x[0], agents[0].x[1], c='tab:blue', s=100, marker='o')
    cbar = plt.colorbar(Q, ax=ax[1], fraction=0.046, pad=0.04)
    # Draw the agent block
    ax[1].add_patch(plt.Rectangle((agents[0].x[0] - param.agent_radius, 
                                   agents[0].x[1] - param.agent_radius), 
                                   2 * param.agent_radius, 2 * param.agent_radius, 
                                   fill=None, edgecolor='tab:blue', lw=2))
    # 3D Plot of the source term
    ax[2].plot_surface(grids_x, grids_y, heat.reshape(param.nbResX, param.nbResY), cmap='viridis')
    plt.suptitle(f'Timestep: {t}')
    plt.pause(0.0001) 
    cbar.remove() 