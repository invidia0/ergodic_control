"""
2D ergodic control formulated as Spectral Multiscale Coverage (SMC) objective,
with a spatial distribution described as a mixture of Gaussians.

Copyright (c) 2023 Idiap Research Institute <https://www.idiap.ch>
Written by Philip Abbet <philip.abbet@idiap.ch> and
Sylvain Calinon <https://calinon.ch>

This file is part of RCFS <https://rcfs.ch>
License: GPL-3.0-only
"""

import numpy as np
#from math import exp
import matplotlib.pyplot as plt
from ergodic_control import models, utilities
import json

# Import the parameters
########################
param_file = 'smc_params.json'

with open(param_file, 'r') as f:
    param_data = json.load(f)

param = lambda: None

for key, value in param_data.items():
    setattr(param, key, value)

L = (param.xlim[1] - param.xlim[0]) * 2  # Size of [-xlim(2),xlim(2)]
param.omega = 2 * np.pi / L
param.Means = np.array(param._mu)
param.Sigmas = np.array(param._sigmas)
param.Weights = np.array(param._weights)


agents = [models.FirstOrderAgent(
    x=np.random.uniform(0, param.L_list[0], 2),
    dt=param.dt,
    max_ut=param.ud
) for _ in range(param.nbAgents)]

grids_x, grids_y = np.meshgrid(
    np.linspace(0, param.xlim[1], 100),
    np.linspace(0, param.xlim[1], 100)
)
grids = np.array([grids_x.ravel(), grids_y.ravel()]).T

# Discretize the GMM into a grid, using Fourier basis functions.
g, w_hat, phim, xx, yy, rg, Lambda = utilities.discrete_gmm(param)
G = np.abs(g).reshape(param.nbResX, param.nbResY)

# Ergodic control
# ===============================
x = np.array([agents[0].x[0], agents[0].x[1]])

wt = np.zeros(param.nbFct**param.nbVar)

fig, axes = plt.subplots(1, 1, figsize=(9,5), dpi=70, tight_layout=True)
ax = axes
ax.set_aspect('equal')
ax.set_xlim(0.0, param.L_list[0])
ax.set_ylim(0.0, param.L_list[1])

x_history = np.empty((0, 2))

for t in range(param.tsteps):
    # Fourier basis functions and derivatives for each dimension
    # (only cosine part on [0,L/2] is computed since the signal
    # is even and real by construction)
    angle = x[:,np.newaxis] * rg * param.om
    phi1 = np.cos(angle) / L 
    dphi1 = -np.sin(angle) * np.tile(rg * param.om, (param.nbVar, 1)) / L

    # Gradient of basis functions
    phix = phi1[0, xx-1].flatten()
    phiy = phi1[1, yy-1].flatten()
    dphix = dphi1[0, xx-1].flatten()
    dphiy = dphi1[1, yy-1].flatten()

    dphi = np.vstack([[dphix * phiy], [phix * dphiy]]).T

    # w are the Fourier series coefficients along trajectory
    wt = wt + (phix * phiy).T
    w = wt / (t + 1)

    # Controller with constrained velocity norm
    u = -dphi.T @ (Lambda * (w - w_hat)) 
    u = u * param.ud / (np.linalg.norm(u) + param.u_norm_reg)  # Velocity command

    agents[0].update(u)  # Update agent state

    x_history = np.vstack((x_history, agents[0].x.copy()))


    # Visualization
    ax.cla()
    ax.set_title(f"t={t*param.dt:.2f}s")
    ax.contourf(grids_x, grids_y, G.reshape(100,100), cmap='Reds')
    ax.scatter(agents[0].x[0], agents[0].x[1], c='tab:blue', s=100, zorder=10)
    ax.plot(x_history[:,0], x_history[:,1], c='black', linewidth=3)

    plt.pause(0.001)