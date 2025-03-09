import numpy as np 
np.set_printoptions(precision=4)
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
from ergodic_control import models, utilities
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
np.random.seed(3) # Seed for reproducibility

scaling_factor = 50

mean1 = np.array([0.35, 0.38]) * scaling_factor
cov1 = np.array([
    [0.01, 0.004],
    [0.004, 0.01]
]) * (scaling_factor ** 2)

mean2 = np.array([0.68, 0.25]) * scaling_factor
cov2 = np.array([
    [0.005, -0.003],
    [-0.003, 0.005]
]) * (scaling_factor ** 2)

mean3 = np.array([0.56, 0.64]) * scaling_factor
cov3 = np.array([
    [0.008, 0.0],
    [0.0, 0.004]
]) * (scaling_factor ** 2)

w1, w2, w3 = 0.5, 0.2, 0.3

def pdf(x):
    return (w1 * mvn.pdf(x, mean1, cov1) + 
            w2 * mvn.pdf(x, mean2, cov2) + 
            w3 * mvn.pdf(x, mean3, cov3))

mean = np.array([40, 40])
cov = np.array([[20, 0], [0, 20]])

# Define a 1-by-1 2D search space
size = 50
L_list = np.array([size, size])  # boundaries for each dimension

# Discretize the search space into 100-by-100 mesh grids
grids_x, grids_y = np.meshgrid(
    np.linspace(0, L_list[0], size),
    np.linspace(0, L_list[1], size)
)
grids = np.array([grids_x.ravel(), grids_y.ravel()]).T
dx = L_list[0] / size
dy = L_list[1] / size

# Configure the index vectors
num_k_per_dim = 20
ks_dim1, ks_dim2 = np.meshgrid(
    np.arange(num_k_per_dim), np.arange(num_k_per_dim)
)
ks = np.array([ks_dim1.ravel(), ks_dim2.ravel()]).T

# Pre-processing lambda_k and h_k
lamk_list = np.power(1.0 + np.linalg.norm(ks, axis=1), -3/2.0)
hk_list = np.zeros(ks.shape[0])
for i, k_vec in enumerate(ks):
    fk_vals = np.prod(np.cos(np.pi * k_vec / L_list * grids), axis=1)  
    hk = np.sqrt(np.sum(np.square(fk_vals)) * dx * dy)
    hk_list[i] = hk

# compute the coefficients for the target distribution
phik_list = np.zeros(ks.shape[0])  
# pdf_vals = pdf(grids)
pdf_vals = utilities.gauss_pdf(grids, mean, cov)
for i, (k_vec, hk) in enumerate(zip(ks, hk_list)):
    fk_vals = np.prod(np.cos(np.pi * k_vec / L_list * grids), axis=1)  
    fk_vals /= hk

    phik = np.sum(fk_vals * pdf_vals) * dx * dy 
    phik_list[i] = phik

# Specify the dynamic system 
dt = 0.1
tsteps = 5000
ud = 1.0  # desired velocity 0.2 m/s

def dyn(xt, ut):
    xdot = ut 
    return ut

def step(xt, ut):
    xt_new = xt + dt * dyn(xt, ut)
    return xt_new 

# start SMC iteration
xt = np.random.uniform(low=0.2, high=0.8, size=(2,))
x_traj = np.zeros((tsteps, 2))
ck_list_update = np.zeros(ks.shape[0])  # trajectory coefficients (update over time, not normalized)
metric_log = []  # record ergodic metric at each step

# coverage_density = np.zeros_like(grids_x)
# width = grids_x.shape[0]
# height = grids_y.shape[1]
# nbVar = 2
# min_kernel_val = 1e-8
# radius = 10
# coverage_block = utilities.agent_block(nbVar, min_kernel_val, radius)
# kernel_size = coverage_block.shape[0]
ergodic_metric = np.zeros(tsteps)

np.random.seed(50)
num_robots = 4
# Initialize multiple robots
xt = np.random.uniform(low=0.1, high=50, size=(num_robots, 2))
x_traj = np.zeros((num_robots, tsteps, 2))
ck_list_update = np.zeros((num_robots, ks.shape[0]))  # Per robot
coverage_density = np.zeros_like(grids_x)
width, height = grids_x.shape[0], grids_y.shape[1]
coverage_block = utilities.agent_block(2, 1e-8, 10)

pdf_vals = pdf_vals.reshape(grids_x.shape)

# def mode_insertion_gradient(xt, global_ck):
#     """Compute control signal using mode insertion gradient."""
#     bt_all = np.zeros((num_robots, 2))
#     for i in range(num_robots):
#         dfk_xt_all = np.array([
#             -np.pi * ks[:, 0] / L_list[0] * np.sin(np.pi * ks[:, 0] / L_list[0] * xt[i, 0]) * np.cos(np.pi * ks[:, 1] / L_list[1] * xt[i, 1]),
#             -np.pi * ks[:, 1] / L_list[1] * np.cos(np.pi * ks[:, 0] / L_list[0] * xt[i, 0]) * np.sin(np.pi * ks[:, 1] / L_list[1] * xt[i, 1]),
#         ]) / hk_list
        
#         # Compute mode-insertion gradient control
#         bt = np.sum(lamk_list * (global_ck - phik_list) * dfk_xt_all, axis=1)
#         norm_bt = np.linalg.norm(bt)
#         bt_all[i] = -ud * bt / (norm_bt + 1e-8) if norm_bt > 1e-8 else np.zeros(2)
#     return bt_all

# # Multi-agent SMC with decentralized control
# for t in range(tsteps):
#     global_ck = np.zeros(ks.shape[0])  # Shared coefficient list
    
#     # Step 2: Each robot updates local ck and contributes to global ck
#     for i in range(num_robots):
#         fk_xt_all = np.prod(np.cos(np.pi * ks / L_list * xt[i]), axis=1) / hk_list
#         ck_list_update[i] += fk_xt_all * dt  # Update local ck
#         global_ck += ck_list_update[i] / num_robots  # Consensus step
    
#     # Step 4: Compute control signal using mode-insertion gradient method
#     ut_all = mode_insertion_gradient(xt, global_ck)
    
#     # Step 5: Move robots
#     for i in range(num_robots):
#         xt[i] = step(xt[i], ut_all[i])
#         x_traj[i, t] = xt[i].copy()
    
#     # Step 6: Update coverage density
#     for i in range(num_robots):
#         x, y = xt[i].astype(int)
#         x_indices, x_start, num_dx = utilities.clamp_kernel_1d(x, 0, width, coverage_block.shape[0])
#         y_indices, y_start, num_dy = utilities.clamp_kernel_1d(y, 0, height, coverage_block.shape[1])
        
#         coverage_density[x_indices, y_indices] += utilities.normalize_mat(
#             coverage_block[x_start:x_start + num_dx, y_start:y_start + num_dy]
#         )
    
#     # Compute ergodic metric
#     coverage = coverage_density / (1 * t + 1)
#     diff = utilities.normalize_mat(pdf_vals) - coverage
#     source = np.maximum(diff, 0) ** 2
#     ergodic_metric = np.sum(source / np.linalg.norm(source)) * dx * dy

# Multi-agent SMC Ergodic Control loop
for t in range(tsteps):
    global_ck = np.zeros(ks.shape[0])  # Shared coefficient list
    
    # Step 2: Each robot computes and shares its ck
    for i in range(num_robots):
        # Evaluate Fourier basis functions
        fk_xt_all = np.prod(np.cos(np.pi * ks / L_list * xt[i]), axis=1) / hk_list
        ck_list_update[i] += fk_xt_all * dt  # Update local ck

        # Aggregate global ck via averaging (or weighted consensus)
        global_ck += ck_list_update[i] / num_robots

    # Step 4: Compute control signal using the GLOBAL ck
    for i in range(num_robots):
        dfk_xt_all = np.array([
            -np.pi * ks[:,0] / L_list[0] * np.sin(np.pi * ks[:,0] / L_list[0] * xt[i, 0]) * np.cos(np.pi * ks[:,1] / L_list[1] * xt[i, 1]),
            -np.pi * ks[:,1] / L_list[1] * np.cos(np.pi * ks[:,0] / L_list[0] * xt[i, 0]) * np.sin(np.pi * ks[:,1] / L_list[1] * xt[i, 1]),
        ]) / hk_list
        
        # Use the GLOBAL ck for control instead of local ck_list_update[i]
        bt = np.sum(lamk_list * (global_ck / (t * dt + dt) - phik_list) * dfk_xt_all, axis=1)
        
        # Normalize control input
        ut = -ud * bt / np.linalg.norm(bt)
        
        # Execute control and move the robot
        xt[i] = step(xt[i], ut)
        x_traj[i, t] = xt[i].copy()

    # Update global coverage density (sum over all agents)
    for i in range(num_robots):
        x, y = xt[i].astype(int)
        x_indices, x_start, num_dx = utilities.clamp_kernel_1d(x, 0, width, coverage_block.shape[0])
        y_indices, y_start, num_dy = utilities.clamp_kernel_1d(y, 0, height, coverage_block.shape[1])
        
        coverage_density[x_indices, y_indices] += utilities.normalize_mat(
            coverage_block[x_start:x_start + num_dx, y_start:y_start + num_dy]
        )

    # Compute ergodic metric
    coverage = coverage_density / (1 * t + 1)
    diff = utilities.normalize_mat(pdf_vals) - coverage
    source = np.maximum(diff, 0) ** 2
    ergodic_metric[t] = np.sum(source / np.linalg.norm(source)) * dx * dy

    # Log ergodic metric
    metric_log.append(np.sum(lamk_list * np.square(phik_list - global_ck / (t * dt + dt))))

# # Save trajectories and ergodic metric
# filepath = os.path.join(os.path.dirname(__file__), 'free_env/smc_sim3_r4/')
# np.save(filepath + 'hist_array.npy', x_traj)
# np.save(filepath + 'ergodic_metric.npy', ergodic_metric)

# visualize the trajectory
fig = plt.figure(figsize=(9,5), dpi=70, tight_layout=True)

ax = fig.add_subplot(111)
ax.set_aspect('equal')
ax.set_xlim(0.0, L_list[0])
ax.set_ylim(0.0, L_list[1])
ax.set_title('Original PDF')
ax.contourf(grids_x, grids_y, pdf_vals.reshape(grids_x.shape), cmap='Reds')
for i in range(num_robots):
    ax.plot(x_traj[i,::10,0], x_traj[i,::10,1], linestyle='-', marker='', color=f'C{i}', alpha=0.5, label='Trajectory', linewidth=4)
    ax.plot(x_traj[i,0,0], x_traj[i,0,1], linestyle='', marker='o', markersize=10, color=f'C{i}', alpha=1.0, label='Initial state')
# ax.plot(x_traj[::10,0], x_traj[::10,1], linestyle='-', marker='', color='k', alpha=0.5, label='Trajectory', linewidth=4)
# ax.plot(x_traj[0,0], x_traj[0,1], linestyle='', marker='o', markersize=10, color='tab:blue', alpha=1.0, label='Initial state')
ax.legend(loc=1)

plt.show()
plt.close()

# Plot the ergodic metric
fig, ax = plt.subplots(1, 1, figsize=(9,5), dpi=70, tight_layout=True)
ax.plot(ergodic_metric, c='tab:blue')
ax.set_xlabel('Time steps')
ax.set_ylabel('Ergodic metric')
plt.show()