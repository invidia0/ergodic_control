import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ergodic_control import models, utilities

# Setup matplotlib properties, remove type 3 fonts
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
# Use tex
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

# Load the data
hedac_r1 = []
hedac_r4 = []
smc_r1 = []
smc_r4 = []
proposed_r1 = []
proposed_r4 = []
voronoi_sim1 = []
voronoi_sim2 = []
voronoi_sim3 = []

nbSims = 3
# Load the data
for i in [1, 4]:
    for dir in ['hedac', 'smc', 'proposed', 'voronoi']:
        if dir == 'voronoi':
            for j in [1, 2, 3]:
                for r in range(8):
                    path = os.path.join(os.path.dirname(__file__), f'{dir}_sim{j}/robot_{r}.npy')
                    data = np.load(path)
                    if j == 1:
                        voronoi_sim1.append(data)
                    elif j == 2:
                        voronoi_sim2.append(data)
                    else:
                        voronoi_sim3.append(data)
        else:
            for j in [1, 2, 3]:
                path = os.path.join(os.path.dirname(__file__), f'{dir}_sim{j}_r{i}/hist_array.npy')
                data = np.load(path)
                if dir == 'hedac':
                    if i == 1:
                        hedac_r1.append(data)
                    else:
                        hedac_r4.append(data)
                elif dir == 'smc':
                    if i == 1:
                        smc_r1.append(data)
                    else:
                        smc_r4.append(data)
                else:
                    if i == 1:
                        proposed_r1.append(data)
                    else:
                        proposed_r4.append(data)
    
nbDataPoints = 5000

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

goal_density = utilities.gauss_pdf(grids, mean, cov)

# Plot the data
width = 7.16 # Full page width for double-column papers
golden_ratio = 0.618 # Golden ratio
height = width * golden_ratio
fontdict = {'weight': 'bold', 'size': 14, 'family': 'serif'}

# fig = plt.figure(figsize=(width, height))
fig = plt.figure()
ax = fig.add_subplot(111)

fontsize = 16
ax.set_xlabel(r'\textbf{Y [m]}', fontsize=fontsize)
ax.set_ylabel(r'\textbf{X [m]}', fontsize=fontsize)
ax.set_aspect('equal')
ax.set_title(r'\textbf{Static Voronoi [14] - 8 Robots}', fontsize=fontsize)

ax.contourf(grids_x, grids_y, goal_density.reshape(size, size), cmap='gray_r')

sim = voronoi_sim3
for i in range(8):
    ax.scatter(sim[i][-1, 1], sim[i][-1, 0], marker='o', facecolors='none', edgecolors='r', s=200, linewidths=3)
    # ax.plot(sim[i][:, 1], sim[i][:, 0], 'r', alpha=0.5, linewidth=2)
    ax.scatter(sim[i][0, 1], sim[i][0, 0], marker='o', facecolors='none', edgecolors='b', s=200, linewidths=1.5, alpha=1, linestyle='dotted')
    # ax.plot(voronoi_sim2[i][:, 1], voronoi_sim2[i][:, 0], 'g', alpha=0.5)
    # ax.plot(voronoi_sim3[i][:, 1], voronoi_sim3[i][:, 0], 'b', alpha=0.5)

plt.savefig('voronoi.pdf', dpi=600, bbox_inches='tight')
plt.tight_layout()  # Ensures that everything fits into the figure area
plt.show()

plt.close("all")
# Plot the proposed method
sim = proposed_r1[0]
fov_span = 90
fov_depth = 5
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel(r'\textbf{Y [m]}', fontsize=fontsize)
ax.set_ylabel(r'\textbf{X [m]}', fontsize=fontsize)
ax.set_aspect('equal')
ax.set_title(r'\textbf{Proposed - 1 Robot}', fontsize=fontsize)
ax.contourf(grids_x, grids_y, goal_density.reshape(size, size), cmap='gray_r')
timestep = 3000
tmp = utilities.init_fov(fov_span, fov_depth)
for r in range(sim.shape[0]):
    ax.scatter(sim[r][timestep, 1], sim[r][timestep, 0], marker='o', facecolors=f'C{r}', edgecolors='k', s=100, linewidths=2, zorder=10)
    ax.plot(sim[r][:timestep, 1], sim[r][:timestep, 0], c=f'C{r}', alpha=0.5, linewidth=2)
    ax.scatter(sim[r][0, 1], sim[r][0, 0], marker='o', facecolors='none', edgecolors='k', s=100, linewidths=2)
    fov = utilities.draw_fov_arc(sim[r][timestep, :2], sim[r][timestep, 2], fov_span, fov_depth, 10)
    # fov = utilities.clip_polygon_no_convex(sim[r][timestep, :2], fov, occ_map, True)
    ax.fill(fov[:, 1], fov[:, 0], color='black', alpha=0.2, zorder=9)
plt.savefig('proposed_r1.pdf', bbox_inches='tight', dpi=600)
plt.tight_layout()
# plt.show()

plt.close("all")
# Plot the proposed method
sim = proposed_r4[0]
fov_span = 90
fov_depth = 5
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel(r'\textbf{Y [m]}', fontsize=fontsize)
ax.set_ylabel(r'\textbf{X [m]}', fontsize=fontsize)
ax.set_aspect('equal')
ax.set_xlim(0, 50)
ax.set_ylim(0, 50)
ax.set_title(r'\textbf{Proposed - 4 Robots}', fontsize=fontsize)
ax.contourf(grids_x, grids_y, goal_density.reshape(size, size), cmap='gray_r')
for r in range(sim.shape[0]):
    ax.scatter(sim[r][-1, 1], sim[r][-1, 0], marker='o', facecolors=f'C{r}', edgecolors='k', s=100, linewidths=2, zorder=10)
    ax.plot(sim[r][:, 1], sim[r][:, 0], c=f'C{r}', alpha=0.5, linewidth=2)
    ax.scatter(sim[r][0, 1], sim[r][0, 0], marker='o', facecolors='none', edgecolors='k', s=100, linewidths=2)
    fov = utilities.draw_fov_arc(sim[r][-1, :2], sim[r][-1, 2], fov_span, fov_depth, 10)
    # fov = utilities.clip_polygon_no_convex(sim[r][timestep, :2], fov, occ_map, True)
    ax.fill(fov[:, 1], fov[:, 0], color='black', alpha=0.2, zorder=9)
plt.savefig('proposed_r4.pdf', dpi=600, bbox_inches='tight')
plt.tight_layout()
# plt.show()

plt.close("all")
# Plot the proposed method
sim = hedac_r1[0]
fov_span = 90
fov_depth = 5
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel(r'\textbf{Y [m]}', fontsize=fontsize)
ax.set_ylabel(r'\textbf{X [m]}', fontsize=fontsize)
ax.set_aspect('equal')
ax.set_xlim(0, 50)
ax.set_ylim(0, 50)
ax.set_title(r'\textbf{HEDAC [7] - 1 Robot}', fontsize=fontsize)
ax.contourf(grids_x, grids_y, goal_density.reshape(size, size), cmap='gray_r')
for r in range(sim.shape[0]):
    ax.scatter(sim[r][-1, 1], sim[r][-1, 0], marker='o', facecolors=f'C{r}', edgecolors='k', s=100, linewidths=2, zorder=10)
    ax.plot(sim[r][:, 1], sim[r][:, 0], c=f'C{r}', alpha=0.5, linewidth=2)
    ax.scatter(sim[r][0, 1], sim[r][0, 0], marker='o', facecolors='none', edgecolors='k', s=100, linewidths=2)
    # fov = utilities.draw_fov_arc(sim[r][-1, :2], sim[r][-1, 2], fov_span, fov_depth, 10)
    # # fov = utilities.clip_polygon_no_convex(sim[r][timestep, :2], fov, occ_map, True)
    # ax.fill(fov[:, 1], fov[:, 0], color='black', alpha=0.2, zorder=9)
plt.savefig('hedac_r1.pdf', dpi=600, bbox_inches='tight')
plt.tight_layout()
# plt.show()

plt.close("all")
# Plot the proposed method
sim = hedac_r4[0]
fov_span = 90
fov_depth = 5
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel(r'\textbf{Y [m]}', fontsize=fontsize)
ax.set_ylabel(r'\textbf{X [m]}', fontsize=fontsize)
ax.set_aspect('equal')
ax.set_xlim(0, 50)
ax.set_ylim(0, 50)
ax.set_title(r'\textbf{HEDAC - 4 Robots}', fontsize=fontsize)
ax.contourf(grids_x, grids_y, goal_density.reshape(size, size), cmap='gray_r')
for r in range(sim.shape[0]):
    ax.scatter(sim[r][-1, 1], sim[r][-1, 0], marker='o', facecolors=f'C{r}', edgecolors='k', s=100, linewidths=2, zorder=10)
    ax.plot(sim[r][:, 1], sim[r][:, 0], c=f'C{r}', alpha=0.5, linewidth=2)
    ax.scatter(sim[r][0, 1], sim[r][0, 0], marker='o', facecolors='none', edgecolors='k', s=100, linewidths=2)
    # fov = utilities.draw_fov_arc(sim[r][-1, :2], sim[r][-1, 2], fov_span, fov_depth, 10)
    # # fov = utilities.clip_polygon_no_convex(sim[r][timestep, :2], fov, occ_map, True)
    # ax.fill(fov[:, 1], fov[:, 0], color='black', alpha=0.2, zorder=9)
plt.savefig('hedac_r4.pdf', bbox_inches='tight', dpi=600)
plt.tight_layout()
# plt.show()

plt.close("all")
# Plot the proposed method
sim = smc_r1[0]
fov_span = 90
fov_depth = 5
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel(r'\textbf{Y [m]}', fontsize=fontsize)
ax.set_ylabel(r'\textbf{X [m]}', fontsize=fontsize)
ax.set_aspect('equal')
ax.set_xlim(0, 50)
ax.set_ylim(0, 50)
ax.set_title(r'\textbf{SMC [15] - 1 Robot}', fontsize=fontsize)
ax.contourf(grids_x, grids_y, goal_density.reshape(size, size), cmap='gray_r')
for r in range(sim.shape[0]):
    ax.scatter(sim[r][-1, 1], sim[r][-1, 0], marker='o', facecolors=f'C{r}', edgecolors='k', s=100, linewidths=2, zorder=10)
    ax.plot(sim[r][:, 1], sim[r][:, 0], c=f'C{r}', alpha=0.5, linewidth=2)
    ax.scatter(sim[r][0, 1], sim[r][0, 0], marker='o', facecolors='none', edgecolors='k', s=100, linewidths=2)
    # fov = utilities.draw_fov_arc(sim[r][-1, :2], sim[r][-1, 2], fov_span, fov_depth, 10)
    # # fov = utilities.clip_polygon_no_convex(sim[r][timestep, :2], fov, occ_map, True)
    # ax.fill(fov[:, 1], fov[:, 0], color='black', alpha=0.2, zorder=9)
plt.savefig('smc_r1.pdf', dpi=600, bbox_inches='tight')
plt.tight_layout()
# plt.show()

plt.close("all")
# Plot the proposed method
sim = smc_r4[0]
fov_span = 90
fov_depth = 5
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel(r'\textbf{Y [m]}', fontsize=fontsize)
ax.set_ylabel(r'\textbf{X [m]}', fontsize=fontsize)
ax.set_aspect('equal')
ax.set_xlim(0, 50)
ax.set_ylim(0, 50)
ax.set_title(r'\textbf{SMC [1] (Only Consensus) - 4 Robots}', fontsize=fontsize)
ax.contourf(grids_x, grids_y, goal_density.reshape(size, size), cmap='gray_r')
for r in range(sim.shape[0]):
    ax.scatter(sim[r][-1, 1], sim[r][-1, 0], marker='o', facecolors=f'C{r}', edgecolors='k', s=100, linewidths=2, zorder=10)
    ax.plot(sim[r][:, 1], sim[r][:, 0], c=f'C{r}', alpha=0.5, linewidth=2)
    ax.scatter(sim[r][0, 1], sim[r][0, 0], marker='o', facecolors='none', edgecolors='k', s=100, linewidths=2)
    # fov = utilities.draw_fov_arc(sim[r][-1, :2], sim[r][-1, 2], fov_span, fov_depth, 10)
    # # fov = utilities.clip_polygon_no_convex(sim[r][timestep, :2], fov, occ_map, True)
    # ax.fill(fov[:, 1], fov[:, 0], color='black', alpha=0.2, zorder=9)
plt.savefig('smc_r4.pdf', dpi=600, bbox_inches='tight')
plt.tight_layout()
plt.show()



