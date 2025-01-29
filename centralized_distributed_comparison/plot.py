from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
import os
from numba import jit
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from ergodic_control import utilities
import warnings
from matplotlib import rc
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from matplotlib import ticker
import matplotlib as mpl
cmap = mcolors.ListedColormap(['black'])
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.patches import PathPatch
from matplotlib.path import Path

def colored_line(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)

def single_gradient_line(x, y, c, ax, cmap='Blues', linewidth=2):
    """
    Plots a single continuous line with a gradient effect.
    
    Parameters:
    x, y : array-like
        Coordinates of the line.
    c : array-like
        Values used to determine the color (e.g., time or another parameter).
    ax : matplotlib.axes.Axes
        Axis object on which to plot the colored line.
    cmap : str
        Colormap to use for coloring the line.
    linewidth : float
        Line width.
    """
    # Normalize color values
    norm = mcolors.Normalize(vmin=np.min(c), vmax=np.max(c))
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    
    # Create a single Path object
    vertices = np.column_stack([x, y])
    path = Path(vertices)
    
    # Apply the gradient colormap as a single stroke
    patch = PathPatch(path, facecolor='none', edgecolor='none', lw=linewidth)
    ax.add_patch(patch)
    
    # Create an image gradient overlay (avoids multiple segments in PDF)
    gradient = np.linspace(0, 1, len(x)).reshape(-1, 1)  # 1D gradient
    ax.imshow(gradient, aspect='auto', cmap=cmap, alpha=0.5, extent=(np.min(x), np.max(x), np.min(y), np.max(y)))

    return sm  # Return ScalarMappable for optional colorbar

class Agent:
    def __init__(self, x, y, theta, id):
        self.x_hist = None
        self.heat_hist = None
        self.mu_hist = None
        self.std_hist = None
        self.coverage_hist = None
        self.combo_hist = None
        self.source_hist = None
        self.ergodic_hist = None
        self.id = id

datapath = os.path.join(os.path.dirname(__file__))

nbAgents = 4
nbDataPoints = 5000
fov_span = 90
fov_depth = 5

# Preload and cache data
map_name = 'simpleMap_05'
map_path = os.path.join(os.getcwd(), 'example_maps/', map_name + '.npy')
map_data = np.load(map_path)

grid_x, grid_y = np.meshgrid(np.arange(map_data.shape[0]), 
                              np.arange(map_data.shape[1]), indexing='ij')

goal_density = np.load(os.path.join(datapath, 'goal_density.npy'))
r1_hist = np.load(os.path.join(datapath, 'r1_hist.npy'))
r2_hist = np.load(os.path.join(datapath, 'r2_hist.npy'))

r1_distributed_hist = np.load(os.path.join(datapath, 'r1_distributed_hist.npy'))
r2_distributed_hist = np.load(os.path.join(datapath, 'r2_distributed_hist.npy'))

# Preprocess the data
print("Preprocessing...")
padded_map = np.pad(map_data, 1, 'constant', constant_values=1)
occ_map = utilities.get_occupied_polygon(padded_map) - 1

# Setup matplotlib properties, remove type 3 fonts
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
# Use tex
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

fontsize = 16

# Double column width
width = 8.9
height = width / 1.618

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(111)
ax.set_aspect('equal')
ax.set_xlabel(r'\textbf{X [m]}', fontsize=fontsize)
ax.set_ylabel(r'\textbf{Y [m]}', fontsize=fontsize)
# ax.set_title(r"\textbf{}", fontsize=fontsize)
# ax.set_xlim([0, map_data.shape[0]])
# ax.set_ylim([0, map_data.shape[1]])
# ax.contourf(grid_x, grid_y, goal_density, cmap='gray_r')
# ax.contourf(grid_x, grid_y, agents[0].source_hist[:, :, timestep], cmap='Oranges')
ax.contourf(grid_x, grid_y, goal_density, cmap='Greys', rasterized=True)
ax.pcolormesh(grid_x, grid_y, np.where(map_data == 0, np.nan, map_data), cmap='gray', rasterized=True)

# hists = [r1_hist, r2_hist]
hists = [r1_distributed_hist, r2_distributed_hist]

for i, hist in enumerate(hists):
    # Scatter initial positions
    ax.scatter(hist[0, 0], hist[0, 1], s=100, marker='o', facecolors='none', edgecolors='black', linewidth=2, zorder=10)

    # Plot final positions
    ax.scatter(hist[-1, 0],
                hist[-1, 1],
                s=100, 
                marker='o', 
                facecolors=f'C{i}', 
                edgecolors='black', 
                linewidth=2,
                zorder=10)

    ax.plot(hist[:, 0], hist[:, 1], color=f'C{i}', alpha=0.5, lw=2)

    fov = utilities.draw_fov_arc(hist[-1, :2], hist[-1, 2], fov_span, fov_depth, 10)
    fov = utilities.clip_polygon_no_convex(hist[-1, :2], fov, occ_map, True)
    ax.fill(*zip(*fov), color='black', alpha=0.2, zorder=9)
plt.savefig(os.path.join(datapath, 'proposed.pdf'), bbox_inches='tight', pad_inches=0, dpi=600)
plt.show()

# ANIMATION

# while True:
#     fig = plt.figure(figsize=(12, 5))
#     ax = fig.add_subplot(121)
#     ax2 = fig.add_subplot(122)
#     ax.set_aspect('equal')
#     ax2.set_aspect('equal')
#     for t in range(0, nbDataPoints, 10):
#         ax.cla()
#         ax.contourf(grid_x, grid_y, goal_density, cmap='Blues')
#         for i, agent in enumerate(agents):
#             ax.plot(agent.x_hist[t - 1:t + 1, 0], agent.x_hist[t - 1:t + 1, 1], color=f'C{i}', alpha=1, lw=2, linestyle='--')
#             ax.scatter(agent.x_hist[t, 0], agent.x_hist[t, 1], c='black', s=100, marker='o')
#             ax.plot(agent.x_hist[:t, 0], agent.x_hist[:t, 1], color=f'C{i}', alpha=1, lw=2)
        
#         ax2.cla()
#         # Plot the source
#         ax2.contourf(grid_x, grid_y, agents[0].source_hist[:, :, t], cmap='Reds')
#         plt.suptitle(f"Time Step {t}")
#         plt.pause(0.01)
#     plt.show()
#     if input("Continue? (y/n)") == 'n':
#         break

print("Preprocessing done.")

fig = plt.figure(figsize=(7, 4))
ax = fig.add_subplot(111)
# Plot the ergodic histogram
markevery = 500
markers = ['o', 's', '^', 'D']
for agent in agents:
    ax.plot(agent.ergodic_hist, color=f"C{agent.id}", label=f"Robot {agent.id}", linestyle='-', marker=markers[agent.id], markersize=7, linewidth=2, markevery=markevery)
ax.set_xlabel(r'\textbf{Time Step}', fontsize=fontsize)
ax.set_ylabel(r'\textbf{Ergodic Metrics}', fontsize=fontsize)
ax.set_ylim([0.025, 0.2])
ax.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right', fontsize=14, frameon=True, shadow=True, fancybox=True)
# plt.savefig(os.path.join(datapath, 'ergodic_metrics.pdf'), format='pdf', dpi=600, bbox_inches='tight')
plt.show()

