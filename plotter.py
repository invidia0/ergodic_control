from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
import os
from numba import jit

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

# Load data
map_name = 'simpleMap_05'
map_path = os.path.join(os.getcwd(), 'example_maps/', map_name + '.npy')
datapath = os.path.join(os.getcwd(), '_datastorage/')

map_data = np.load(map_path)
heat_hist = np.load(datapath + '_heat_hist.npy')
x_hist = np.load(datapath + 'agents_x_hist.npy')
goal_density = np.load(datapath + 'goal_density.npy')

# Setup
nbDataPoints = heat_hist.shape[0]
time_array = np.linspace(0, nbDataPoints, nbDataPoints)
grid_x, grid_y = np.meshgrid(np.arange(goal_density.shape[0]), 
                            np.arange(goal_density.shape[1]), 
                            indexing='ij')

# Initialize figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
for ax in (ax1, ax2):
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

contour_goal = ax1.contourf(grid_x, grid_y, goal_density, cmap='Greys', levels=10)

# Initialize scatter points
scatter_start = ax1.scatter(x_hist[0, 0], x_hist[0, 1], s=100, facecolors='none', 
                          edgecolors='green', lw=2)
scatter_current = ax1.scatter([], [], s=100, c='black', marker='o')

# Initialize static elements
map_plot1 = ax1.pcolormesh(grid_x, grid_y, np.where(map_data == 0, np.nan, map_data), 
                          cmap='gray')  # This shows the map with walls in gray
map_plot2 = ax2.pcolormesh(grid_x, grid_y, np.where(map_data == 0, np.nan, map_data), 
                          cmap='gray')  # Same here
ax2.set_title('Heatmap')
ax1.set_title('Agent path')

# Store global variables
heat_contour = None
current_line = None

def update(frame):
    global heat_contour, current_line
    
    # Remove previous line collection
    if current_line is not None:
        current_line.remove()
    
    # Create new colored line
    if frame > 0:
        segments = compute_midpoints_and_segments(x_hist[:frame, 0], x_hist[:frame, 1])
        current_line = LineCollection(segments, cmap='Blues', capstyle='butt', linewidth=2)
        current_line.set_array(time_array[:frame-1])
        ax1.add_collection(current_line)
    
    # Update heat contour
    if heat_contour is not None:
        for coll in heat_contour.collections:
            coll.remove()
    heat_contour = ax2.contourf(grid_x, grid_y, heat_hist[frame], 
                               cmap='Blues', levels=10)
    
    
    # Update current position
    scatter_current.set_offsets([x_hist[frame, 0], x_hist[frame, 1]])

    fig.suptitle(f"Frame: {frame}/{nbDataPoints}", fontsize=16)

    print(f'\rFrame: {frame}/{nbDataPoints}', end='', flush=True)
    
    # Return only scatter and heat contour (no quiver here)
    return scatter_current, heat_contour, current_line

# Create and save animation
anim = FuncAnimation(fig, update, frames=range(0, nbDataPoints, 10), blit=False)
fig.tight_layout()
# store the animation in order to save it with different fps
anim.save('ergodic_control_map_05.mp4', writer='ffmpeg', fps=29)
plt.close()
