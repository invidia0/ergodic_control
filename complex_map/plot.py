import numpy as np
import matplotlib.pyplot as plt
import os 
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ergodic_control import models, utilities
import matplotlib.animation as animation

# Setup matplotlib properties, remove type 3 fonts
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
# Use tex
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

nbDataPoints = 8000
nbAgents = 4
# Preload and cache data
map_name = 'complexMap_05'
map_path = os.path.join(os.getcwd(), 'example_maps/', map_name + '.npy')
map = np.load(map_path).T
padded_map = np.pad(map, 1, 'constant', constant_values=1)
occ_map = utilities.get_occupied_polygon(padded_map) - 1


grid_x, grid_y = np.meshgrid(np.arange(map.shape[0]), 
                              np.arange(map.shape[1]), indexing='ij')

# load the goal density
path = os.path.join(os.getcwd(), 'complex_map/', 'goal_density.npy')
goal_density = np.load(path).reshape(map.shape)
path_histories = os.path.join(os.getcwd(), 'complex_map/', 'hist_array.npy')
path_histories = np.load(path_histories)

source_path = os.path.join(os.getcwd(), 'complex_map/', 'source.dat')

source_hist = np.memmap(
    source_path, dtype=np.float64, mode='r', 
    shape=(map.shape[0], map.shape[1], nbDataPoints, nbAgents)
)

fontsize = 16
width = 3.487 * 2
height = width / 1.618

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlabel(r'\textbf{X [m]}', fontsize=fontsize)
ax.set_ylabel(r'\textbf{Y [m]}', fontsize=fontsize)
ax.set_title(r"\textbf{Complex Environment Simulation}", fontsize=fontsize)

# Function to update each frame
nbAgents = 1
def update(timestep):
    print(f"Generating frame {timestep}/{nbDataPoints}")  # Print frame status

    ax.clear()
    ax.set_aspect('equal')
    ax.set_xlabel(r'\textbf{X [m]}', fontsize=fontsize)
    ax.set_ylabel(r'\textbf{Y [m]}', fontsize=fontsize)
    ax.set_title(r"\textbf{Time Step: " + str(timestep) + "}", fontsize=fontsize)
    
    # ax.contourf(grid_x, grid_y, goal_density, cmap='Greys', levels=10)
    ax.contourf(grid_x, grid_y, source_hist[:, :, timestep, 0], cmap='Oranges', levels=10)
    ax.pcolormesh(grid_x, grid_y, np.where(map == 0, np.nan, map), cmap='gray', rasterized=True)

    for i in range(nbAgents):
        ax.plot(path_histories[i, :timestep, 0], path_histories[i, :timestep, 1], color=f'C{i}', alpha=0.5, lw=2, zorder=1)
        ax.scatter(path_histories[i, 0, 0], path_histories[i, 0, 1], marker='o', facecolors='none', edgecolors='k', s=100, lw=2, zorder=10)
        ax.scatter(path_histories[i, timestep % nbDataPoints, 0], path_histories[i, timestep % nbDataPoints, 1], marker='o', facecolors=f'C{i}', edgecolors='k', s=100, lw=2, zorder=10)
        fov = utilities.draw_fov_arc(path_histories[i, timestep % nbDataPoints, :2], path_histories[i, timestep % nbDataPoints, 2], 90, 5, 10)
        fov = utilities.clip_polygon_no_convex(path_histories[i, timestep % nbDataPoints, :2], fov, occ_map, True)
        ax.fill(fov[:, 0], fov[:, 1], color='black', alpha=0.2, zorder=9)

# print the status
ani = animation.FuncAnimation(fig, update, frames=np.arange(0, 8000, 10), interval=100, repeat=False)

# Save as mp4 (requires ffmpeg)
ani.save('simulation_video_FINAL_source_2.mp4', writer='ffmpeg', fps=30, dpi=300)

plt.show()

# fontsize = 16
# # single column width
# width = 3.487 * 2
# height = width / 1.618
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_title(map_name)
# ax.set_aspect('equal')
# ax.set_xlabel(r'\textbf{X [m]}', fontsize=fontsize)
# ax.set_ylabel(r'\textbf{Y [m]}', fontsize=fontsize)
# ax.set_title(r"\textbf{Complex Environment Simulation}", fontsize=fontsize)
# ax.contourf(grid_x, grid_y, goal_density, cmap='Greys', levels=10)
# ax.pcolormesh(grid_x, grid_y, np.where(map == 0, np.nan, map), cmap='gray', rasterized=True)
# fov_depth = 5
# fov_span = 90
# time_step = 7900
# for timestep in np.arange(0, nbDataPoints, 10):
#     ax.clear()
#     ax.set_aspect('equal')
#     ax.set_xlabel(r'\textbf{X [m]}', fontsize=fontsize)
#     ax.set_ylabel(r'\textbf{Y [m]}', fontsize=fontsize)
#     ax.set_title(r"\textbf{Time Step: " + str(timestep) + "}", fontsize=fontsize)
#     ax.contourf(grid_x, grid_y, goal_density, cmap='Greys', levels=10)
#     ax.pcolormesh(grid_x, grid_y, np.where(map == 0, np.nan, map), cmap='gray', rasterized=True)
#     for i in range(nbAgents):
#         # Initial scatter
#         ax.plot(path_histories[i, :timestep, 0], path_histories[i, :timestep, 1], color=f'C{i}', alpha=0.5, lw=2, zorder=1)
#         ax.scatter(path_histories[i, 0, 0], path_histories[i, 0, 1], marker='o', facecolors='none', edgecolors='k', s=100, lw=2, zorder=10)
#         ax.scatter(path_histories[i, timestep, 0], path_histories[i, timestep, 1], marker='o', facecolors=f'C{i}', edgecolors='k', s=100, lw=2, zorder=10)

#         fov = utilities.draw_fov_arc(path_histories[i, timestep, :2], path_histories[i, timestep, 2], fov_span, fov_depth, 10)
#         fov = utilities.clip_polygon_no_convex(path_histories[i, timestep, :2], fov, occ_map, True)
#         ax.fill(*zip(*fov), color='black', alpha=0.2, zorder=9)

#     plt.pause(0.001)

# plt.show()

