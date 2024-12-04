"""
Code for simulating the heat equation in 2D
"""
import numpy as np
import matplotlib.pyplot as plt
from ergodic_control import models, utilities

# Parameters
param = lambda: None
param.L = 1  # Length of the domain
param.nx = 100  # Number of grid points
param.dx = param.L / param.nx  # Grid spacing
param.dt = 0.01  # Time step
param.T = 1000  # Total time
param.diffusion = 0.1  # DIFFUSION
param.beta = 0.1  # CONVECTIVE TERM (global cooling)
param.alpha = np.array(param.alpha) * param.diffusion


