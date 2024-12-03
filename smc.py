import numpy as np 
np.set_printoptions(precision=4)
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
from ergodic_control import models, utilities

np.random.seed(3) # Seed for reproducibility

# Import the parameters
########################
param_file = 'smc_params.json'

with open(param_file, 'r') as f:
    param_data = json.load(f)

param = lambda: None

for key, value in param_data.items():
    setattr(param, key, value)

# Initialize the agent and the target distribution
##################################################

agents = [models.FirstOrderAgent(
    x=np.random.uniform(0, param.L_list[0], 2),
    dt=param.dt,
    max_ut=param.ud
) for _ in range(param.nbAgents)]

grids_x, grids_y = np.meshgrid(
    np.linspace(0, param.L_list[0], param.nbResX),
    np.linspace(0, param.L_list[1], param.nbResY)
)
grids = np.array([grids_x.ravel(), grids_y.ravel()]).T

pdf_vals = utilities.gmm_eval(grids, param._mu, param._sigmas, param._weights).reshape(param.nbResX, param.nbResY)

dx = param.dx / param.nbResX
dy = param.dy / param.nbResY

# Configure the index vectors
ks_dim1, ks_dim2 = np.meshgrid(
    np.arange(param.nbFct), np.arange(param.nbFct)
)
ks = np.array([ks_dim1.ravel(), ks_dim2.ravel()]).T

# Pre-processing lambda_k and h_k
lamk_list = np.power(1.0 + np.linalg.norm(ks, axis=1), -3/2.0)
hk_list = np.zeros(ks.shape[0])
for i, k_vec in enumerate(ks):
    fk_vals = np.prod(np.cos(np.pi * k_vec / param.L_list * grids), axis=1)  
    hk = np.sqrt(np.sum(np.square(fk_vals)) * dx * dy)
    hk_list[i] = hk

# compute the coefficients for the target distribution
phik_list = np.zeros(ks.shape[0])  
for i, (k_vec, hk) in enumerate(zip(ks, hk_list)):
    fk_vals = np.prod(np.cos(np.pi * k_vec / param.L_list * grids), axis=1)  
    fk_vals /= hk

    phik = np.sum(fk_vals * pdf_vals.reshape(-1)) * dx * dy 
    phik_list[i] = phik

fig, axes = plt.subplots(1, 1, figsize=(9,5), dpi=70, tight_layout=True)
ax = axes
ax.set_aspect('equal')
ax.set_xlim(0.0, param.L_list[0])
ax.set_ylim(0.0, param.L_list[1])

x_traj = np.zeros((param.tsteps, 2))
x_history = np.empty((0, 2))
ck_list_update = np.zeros(ks.shape[0])  # trajectory coefficients (update over time, not normalized)
metric_log = []  # record ergodic metric at each step

# Main loop
###########
for t in range(param.tsteps):
    # step 1: evaluate all the fourier basis functions at the current state
    fk_xt_all = np.prod(np.cos(np.pi * ks / param.L_list * agents[0].x), axis=1) / hk_list
    
    # step 2: update the coefficients
    ck_list_update += fk_xt_all * param.dt

    # step 3: compute the derivative of all basis functions at the current state
    dfk_xt_all = np.array([
        -np.pi * ks[:,0] / param.L_list[0] * np.sin(np.pi * ks[:,0] / param.L_list[0] * agents[0].x[0]) * np.cos(np.pi * ks[:,1] / param.L_list[1] * agents[0].x[1]),
        -np.pi * ks[:,1] / param.L_list[1] * np.cos(np.pi * ks[:,0] / param.L_list[0] * agents[0].x[0]) * np.sin(np.pi * ks[:,1] / param.L_list[1] * agents[0].x[1]),
    ]) / hk_list
    
    # step 4: compute control signal
    bt = np.sum(lamk_list * (ck_list_update / (t*param.dt+param.dt) - phik_list) * dfk_xt_all, axis=1)
    ut = -param.ud * bt / (np.linalg.norm(bt) + param.u_norm_reg)
    
    # step 5: execute the control, move on to the next iteration
    agents[0].update(ut)
    # xt = xt_new.copy()
    x_history = np.vstack((x_history, agents[0].x.copy()))
    
    # erg_metric = np.sum(lamk_list * np.square(phik_list - ck_list_update / (t*param.dt+param.dt)))
    # metric_log.append(erg_metric)

    # Plot the current state
    ax.cla()
    ax.set_title(f"t={t*param.dt:.2f}s")
    ax.contourf(grids_x, grids_y, pdf_vals.reshape(100,100), cmap='Reds')
    ax.scatter(agents[0].x[0], agents[0].x[1], c='tab:blue', s=100, zorder=10)
    ax.plot(x_history[:,0], x_history[:,1], c='black', linewidth=3)

    plt.pause(0.001)
    