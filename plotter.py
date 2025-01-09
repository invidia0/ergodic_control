from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
import os
from numba import jit
from ergodic_control import utilities

# JIT-optimized functions (no changes here, still using `compute_midpoints_and_segments`)
@jit(nopython=True)
def compute_midpoints_and_segments(x, y):
    """Compute midpoints and create segments for colored line"""
    x_mid = 0.5 * (x[1:] + x[:-1])
    y_mid = 0.5 * (y[1:] + y[:-1])
    x_midpts = np.zeros(len(x_mid) + 2)
    y_midpts = np.zeros(len(y_mid) + 2)
    x_midpts[0] = x[0]
    x_midpts[1:-1] = x_mid
    x_midpts[-1] = x[-1]
    y_midpts[0] = y[0]
    y_midpts[1:-1] = y_mid
    y_midpts[-1] = y[-1]
    
    n = len(x)
    segments = np.zeros((n-1, 3, 2))
    for i in range(n-1):
        segments[i, 0] = [x_midpts[i], y_midpts[i]]  # start
        segments[i, 1] = [x[i], y[i]]                # middle
        segments[i, 2] = [x_midpts[i+1], y_midpts[i+1]]  # end
        
    return segments

# Preload and cache data
map_name = 'simpleMap_05'
map_path = os.path.join(os.getcwd(), 'example_maps/', map_name + '.npy')
datapath = os.path.join(os.getcwd(), '_datastorage/')

map_data = np.load(map_path)
heat_hist = np.load(datapath + 'heat_hist.npy')
x_hist = np.load(datapath + 'x_hist.npy')
goal_density = np.load(datapath + 'goal_density.npy')
coverage_hist = np.load(datapath + 'coverage_hist.npy')
mu_hist = np.load(datapath + 'mu_hist.npy')
std_hist = np.load(datapath + 'std_hist.npy')

nbDataPoints = heat_hist.shape[0]
time_array = np.linspace(0, nbDataPoints, nbDataPoints)
grid_x, grid_y = np.meshgrid(np.arange(goal_density.shape[0]), 
                              np.arange(goal_density.shape[1]), indexing='ij')

combo_hist = np.exp(mu_hist) + np.exp(std_hist) - 2
source_hist = np.maximum(utilities.normalize_mat(combo_hist) - utilities.normalize_mat(coverage_hist), 0)**2
ergodic_hist = np.linalg.norm(combo_hist - coverage_hist, ord=2, axis=(1, 2))

padded_map = np.pad(map_data, 1, 'constant', constant_values=1)
occ_map = utilities.get_occupied_polygon(padded_map) - 1

fov_depth = 5
fov_span = 90
fov_list = [utilities.clip_polygon_no_convex(x_hist[t, :2], 
                                              utilities.draw_fov_arc(x_hist[t, :2], x_hist[t, 2], fov_span, fov_depth, 10), 
                                              occ_map, True) for t in range(nbDataPoints)]

# Manage the first step where t == 0
segments_list = [compute_midpoints_and_segments(x_hist[:t, 0], x_hist[:t, 1]) for t in range(1, nbDataPoints)]

# Initialize figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
contour_goal = ax1.contourf(grid_x, grid_y, goal_density, cmap='Reds', levels=10)
scatter_current = ax1.scatter([], [], s=100, c='black', marker='o', zorder=9)
fov_fill = ax1.fill([], [], color='tab:blue', alpha=0.5)
map_plot1 = ax1.pcolormesh(grid_x, grid_y, np.where(map_data == 0, np.nan, map_data), cmap='gray', zorder=10)
map_plot2 = ax2.pcolormesh(grid_x, grid_y, np.where(map_data == 0, np.nan, map_data), cmap='gray', zorder=10)
source_contour = None
current_line = None

def update(frame):
    global source_contour, current_line

    if current_line is not None:
        current_line.remove()
    current_line = LineCollection(segments_list[frame], cmap='viridis', linewidth=2, zorder=1)
    current_line.set_array(time_array[:frame])
    ax1.add_collection(current_line)

    scatter_current.set_offsets([x_hist[frame, 0], x_hist[frame, 1]])

    if source_contour is not None:
        for coll in source_contour.collections:
            coll.remove()
    source_contour = ax2.contourf(grid_x, grid_y, source_hist[frame], cmap='Blues', levels=10)

    fov_fill[0].set_xy(fov_list[frame])
    fig.suptitle(f"Frame: {frame}/{nbDataPoints}", fontsize=16)
    print(f"Frame: {frame}/{nbDataPoints}")
    return scatter_current, source_contour, current_line

anim = FuncAnimation(fig, update, frames=range(0, nbDataPoints, 10), blit=False, interval=30)
anim.save('ergodic_control_animation.mp4', writer='ffmpeg')
plt.close()
