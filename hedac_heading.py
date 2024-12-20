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

warnings.filterwarnings("ignore")

np.random.seed(0) # Seed for reproducibility

""" Toy example of a 3-component Gaussian Mixture Model """
mean1 = np.array([0.35, 0.38])
cov1 = np.array([
    [0.01, 0.004],
    [0.004, 0.01]
])
w1 = 0.5

mean2 = np.array([0.68, 0.25])
cov2 = np.array([
    [0.005, -0.003],
    [-0.003, 0.005]
])
w2 = 0.2

mean3 = np.array([0.56, 0.64])
cov3 = np.array([
    [0.008, 0.0],
    [0.0, 0.004]
])
w3 = 0.3

def pdf(x):
    return w1 * mvn.pdf(x, mean1, cov1) + \
           w2 * mvn.pdf(x, mean2, cov2) + \
           w3 * mvn.pdf(x, mean3, cov3)

"""
===============================
Parameters
===============================
"""

param_file = os.path.dirname(os.path.realpath(__file__)) + '/params/' + 'hedac_params.json'

with open(param_file, 'r') as f:
    param_data = json.load(f)

param = lambda: None

for key, value in param_data.items():
    setattr(param, key, value)

param.alpha = np.array(param.alpha) * param.diffusion
param.max_dtheta = np.pi / 12 # Maximum angular velocity (rad/s)
param.box = np.array([[0, 0], 
                    [param.nbResX, 0],
                    [param.nbResX, param.nbResY],
                    [0, param.nbResY]]) # Box constraints

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

for a in ax:
    a.set_xlim([0, param.xlim[1]])
    a.set_ylim([0, param.xlim[1]])
    a.set_aspect('equal')
    a.set_xticks([])
    a.set_yticks([])

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
for i in range(param.nbAgents):
    # x0 = np.random.uniform(0, param.nbResX, 2)
    x0 = np.array([20, 20])
    theta0 = np.random.uniform(0, 2 * np.pi)
    agent = models.SecondOrderAgentWithHeading(x=x0,
                                                theta=theta0,
                                                max_dx=param.max_dx,
                                                max_ddx=param.max_ddx,
                                                max_dtheta=param.max_dtheta,
                                                dt=param.dt)
    agents.append(agent)

"""
===============================
Goal Density
===============================
"""
grids_x, grids_y = np.meshgrid(
    np.linspace(0, param.xlim[1], 100),
    np.linspace(0, param.xlim[1], 100)
)
grids = np.array([grids_x.ravel(), grids_y.ravel()]).T

""" Gaussian Mixture Model and FFT approximation """
# param.Means, param.Covs, param.Weights = utilities.compute_gmm(param)

# Ground truth density function
# pdf_gt = utilities.gmm_eval(grids, param.Means, param.Covs, param.Weights).reshape(param.nbResX, param.nbResY)
# pdf_gt = np.abs(pdf_gt)
# pdf_gt = utilities.normalize_mat(pdf_gt)
# Discretize the GMM into a grid, using Fourier basis functions.

# G = utilities.discrete_gmm_original(param).reshape(param.nbResX, param.nbResY)
# G = np.abs(G)
# G = utilities.normalize_mat(G)
# goal_density = G

""" Smoothing/Spreading RBF Approximation """
def rbf_func(mean, x, eps=1):
    return np.exp(-eps * np.linalg.norm(mean - x)**2)

def kernel_matrix(x, sigma=1):
    return [[rbf_func(x1, x2, sigma) for x2 in x] for x1 in x]

def fit(G, d):
    return np.dot(np.linalg.pinv(G), d)

def normalize_mat(mat):
    return mat / (np.sum(mat) + 1e-10)

def predict(inputs, X_train, m, eps=1):
    pairwise_distances = np.linalg.norm(inputs[:, np.newaxis, :] - X_train[np.newaxis, :, :], axis=2)
    K = np.exp(-eps * pairwise_distances**2)
    return K @ m

N = 100 # Number of basis functions to use to approximate each point
samples = np.random.rand(N, 2)
d = pdf(samples).reshape(-1, 1)
eps = 1 / (2 * 0.1 ** 2) # Shape parameter

G = kernel_matrix(samples, eps) # Kernel matrix
m = fit(G, d) # Weights for the RBF basis functions

S = predict(grids, samples, m, eps).reshape(param.nbResX, param.nbResY)
S = np.abs(S)
S = utilities.normalize_mat(S)

goal_density = S

"""
===============================
Initialize heat equation related parameters
===============================
"""
param.height, param.width = goal_density.shape

param.area = param.dx * param.width * param.dx * param.height

param.beta = param.beta / param.area # Eq. 17 - Beta normalized 
param.local_cooling = param.local_cooling / param.area # Eq. 16 - Local cooling normalized

coverage_density = np.zeros((param.height, param.width))
heat = np.array(goal_density) # The heat is the goal density

max_diffusion = np.max(param.alpha)
param.dt = min(
    0.2, (param.dx * param.dx) / (4.0 * max_diffusion)
)  # for the stability of implicit integration of Heat Equation

if param.use_fov:
    # Initialize the Field of View (FOV)
    fov_edges = utilities.init_fov(param.fov_deg, param.fov_depth)
    fov_edges = utilities.rotate_and_translate(fov_edges, agents[0].x, agents[0].theta)
    param.kernel_size = param.fov_depth
else:
    coverage_block = utilities.agent_block(param.nbVarX, param.min_kernel_val, param.agent_radius)
    param.kernel_size = coverage_block.shape[0]

"""
===============================
Main Loop
===============================
"""
last_heading = agents[0].theta
last_position = agents[0].x

for t in range(param.nbDataPoints):
    print(f'Timestep: {t}')

    local_cooling = np.zeros((param.height, param.width))
    for agent in agents:
        p = agent.x
        if param.use_fov:
            # fov_edges = utilities.init_fov(param.fov_deg, param.fov_depth)
            fov_edges = utilities.relative_move_fov(fov_edges, last_position, last_heading, agent.x, agent.theta)
            # fov_edges = utilities.rotate_and_translate(fov_edges, p, agent.theta)
            # Clip the FOV
            fov_edges_clipped = utilities.clip_polygon(fov_edges, param.box)
            fov_points = utilities.insidepolygon(fov_edges_clipped, grid_step=param.dx).T.astype(int)
            # Delete points outside the box
            fov_points = fov_points[
                (fov_points[:, 0] >= 0)
                & (fov_points[:, 0] < param.width)
                & (fov_points[:, 1] >= 0)
                & (fov_points[:, 1] < param.height)
            ]
            fov_probs = utilities.fov_coverage_block(fov_points, fov_edges_clipped, param.fov_depth)

            # # DEBUG
            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            # ax.scatter(p[0], p[1], c='black', s=100, marker='o')
            # for i in range(len(param.box)):
            #     ax.plot([param.box[i][0], param.box[(i + 1) % len(param.box)][0]], [param.box[i][1], param.box[(i + 1) % len(param.box)][1]], 'r-')
            # for i in range(len(fov_edges)):
            #     ax.plot([fov_edges[i][0], fov_edges[(i + 1) % len(fov_edges)][0]], [fov_edges[i][1], fov_edges[(i + 1) % len(fov_edges)][1]], 'b-')
            # ax.scatter(fov_points[:, 0], fov_points[:, 1], c=fov_probs, cmap='Reds', s=10, marker='o')
            # ax.set_xlim([-5, 105])
            # ax.set_ylim([-5, 105])
            # ax.set_aspect('equal')
            # ax.set_title('Field of View')
            # plt.show()

            coverage_density[fov_points[:, 1], fov_points[:, 0]] += fov_probs
        else:
            adjusted_position = p / param.dx
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

        # if param.local_cooling != 0:
        #     local_cooling[row_indices, col_indices] += coverage_block[
        #         row_start_kernel : row_start_kernel + num_kernel_rows,
        #         col_start_kernel : col_start_kernel + num_kernel_cols,
        #     ]
        # local_cooling = utilities.normalize_mat(local_cooling) * param.area # Eq. 15 - Local cooling normalized

    coverage = utilities.normalize_mat(coverage_density) # Eq. 4 - Coverage density normalized

    diff = goal_density - coverage # Eq. 6 - Difference between the goal density and the coverage density
    source = np.maximum(diff, 0) ** 2 # Eq. 13 - Source term
    source = utilities.normalize_mat(source) * param.area # Eq. 14 - Source term scaled

    current_heat = np.zeros((param.height, param.width))

    """
    Finite difference method to solve the heat equation
    https://en.wikipedia.org/wiki/Five-point_stencil
    https://www.youtube.com/watch?v=YotrBNLFen0
    """
    # This is the heat equation (non-stationary) with a source term
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
        # - param.local_cooling * utilities.offset(local_cooling, 0, 0)
        - param.beta * utilities.offset(heat, 0, 0)
    )  + utilities.offset(heat, 0, 0)

    heat = current_heat.astype(np.float32)

    gradient_y, gradient_x = np.gradient(heat, 1, 1) # Gradient of the heat

    gradient_x = gradient_x / np.linalg.norm(gradient_x)
    gradient_y = gradient_y / np.linalg.norm(gradient_y)

    last_heading = agents[0].theta
    last_position = agents[0].x

    for agent in agents:
        grad = utilities.calculate_gradient(
            param,
            agent,
            gradient_x,
            gradient_y,
        )
        agent.update(grad)

    if t == param.nbDataPoints - 1:
        # Plot the agents
        ax[0].cla()
        ax[1].cla()

        ax[0].contourf(grids_x*100, grids_y*100, goal_density, cmap='Reds', levels=10)
        # ax[0].plot(agents[0].x_hist[:, 0], agents[0].x_hist[:, 1], c='black', alpha=1, lw=2)
        time_array = np.linspace(0, param.nbDataPoints, param.nbDataPoints)
        lines = utilities.colored_line(agents[0].x_hist[:, 0], 
                                        agents[0].x_hist[:, 1],
                                        time_array,
                                        ax[0],
                                        cmap='Greys',
                                        linewidth=2)
        # FOV
        if param.use_fov:
            ax[0].fill(fov_edges[:, 0], fov_edges[:, 1], 'b', alpha=0.2)
        # Heading
        ax[0].quiver(agents[0].x[0], agents[0].x[1], np.cos(agents[0].theta), np.sin(agents[0].theta), scale = None, scale_units='inches')
        ax[0].scatter(agents[0].x_hist[0, 0], agents[0].x_hist[0, 1], s=100, facecolors='none', edgecolors='black', lw=2)
        ax[0].scatter(agents[0].x[0], agents[0].x[1], c='k', s=100, marker='o', zorder=10)

        ax[1].contourf(grids_x*100, grids_y*100, heat, cmap='coolwarm', levels=10)
        ax[1].quiver(grids_x*100, grids_y*100, gradient_x / np.linalg.norm(gradient_x), gradient_y / np.linalg.norm(gradient_y), scale = 1, scale_units='inches')
        ax[1].scatter(agents[0].x[0], agents[0].x[1], c='black', s=100, marker='o')
        # Plot the heading
        ax[1].quiver(agents[0].x[0], agents[0].x[1], np.cos(agents[0].theta), np.sin(agents[0].theta), scale = 2, scale_units='inches')

        plt.suptitle(f'Timestep: {t}')
        # plt.pause(0.0001)
        plt.show()