from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
import os
from numba import jit
from ergodic_control import utilities
import warnings
from matplotlib import rc
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from matplotlib import ticker
import matplotlib as mpl
cmap = mcolors.ListedColormap(['black'])
import matplotlib.gridspec as gridspec

# Initialize Agents
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

# # JIT-optimized functions (no changes here, still using `compute_midpoints_and_segments`)
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

# Setup matplotlib properties, remove type 3 fonts
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
# Set font
plt.rcParams['font.family'] = 'sans-serif'
# Computer Modern Roman
plt.rcParams['font.serif'] = 'Helvetica'
plt.rcParams['font.size'] = 12

# Preload and cache data
map_name = 'simpleMap_05'
map_path = os.path.join(os.getcwd(), 'example_maps/', map_name + '.npy')
# datapath = os.path.join(os.getcwd(), '_datastorage/time_decay_back_and_forth')
datapath = os.path.join(os.getcwd(), '_datastorage/solo_simulation/')

# Load with memmap
map_data = np.load(map_path)

fov_span = 90
fov_depth = 5
grid_x, grid_y = np.meshgrid(np.arange(map_data.shape[0]), 
                              np.arange(map_data.shape[1]), indexing='ij')
nbAgents = 1
nbDataPoints = 5000

# Remove type 3 fonts
rc('pdf', fonttype=42)
rc('ps', fonttype=42)

# Plot the ergodic metric over time
width = 7.16 # Full page width for double-column papers
golden_ratio = 0.618 # Golden ratio
# evalAgent = agents[0]
fontdict = {'weight': 'bold'}
timesteps = [100, 500, 1000, nbDataPoints-1]

############################
# Plot the comparison on exploration performance
############################
""" time_array = np.linspace(0, nbDataPoints - 1, nbDataPoints)
from scipy.interpolate import make_interp_spline, BSpline, interp1d
from scipy.ndimage import gaussian_filter1d

sims = ["sim1", "sim2", "sim3"]
paths = ["ivic2019_1/", "ivic2019_4/"]
nbAgents_sims = [1, 4]

# heat_histories = []
# path_histories = []
source_histories_1 = []
source_histories_4 = []

proposed_1 = []
proposed_4 = []

for sim in sims:
    for i, path in enumerate(paths):
        source_hist_path = os.path.join(os.path.join(os.getcwd(), '_datastorage/', sim), path + "source_hist.dat")
        souces_hist = np.memmap(
            source_hist_path, dtype=np.float64, mode='r', 
            shape=(map_data.shape[0], map_data.shape[1], nbDataPoints)
        )

        if i == 0:
            source_histories_1.append(souces_hist)
        else:
            source_histories_4.append(souces_hist)

    # Load the uncertainty arrays
    proposed_1.append(np.load(datapath + sim + '/uncertainty_array_1.npy'))
    proposed_4.append(np.load(datapath + sim + '/uncertainty_array_4.npy'))

ivic2019_1 = np.zeros((nbDataPoints, len(sims)))
ivic2019_4 = np.zeros((nbDataPoints, len(sims)))
proposed_1 = np.array(proposed_1)
proposed_4 = np.array(proposed_4)
for i in range(len(sims)):
    ivic2019_1[:, i] = np.sum(np.sum(source_histories_1[i], axis=0), axis=0)
    ivic2019_4[:, i] = np.sum(np.sum(source_histories_4[i], axis=0), axis=0)

# Take the mean of the proposed 4 among the robots
proposed_4 = np.mean(proposed_4, axis=2)

# Generate a denser time array for smoother interpolation
dense_time_array = np.linspace(time_array.min(), time_array.max(), 500)

# Apply B-spline smoothing (better than cubic interpolation for rough data)
def smooth_curve(x, y, smooth_factor=3):
    spline = make_interp_spline(x, y, k=3)  # B-spline with degree 3
    smoothed_y = spline(dense_time_array)
    return gaussian_filter1d(smoothed_y, sigma=smooth_factor)  # Further smoothing

# Smooth the curves
ivic2019_1_mean = np.mean(ivic2019_1, axis=1)
ivic2019_1_smooth = smooth_curve(time_array, ivic2019_1_mean)
ivic2019_1_std = np.std(ivic2019_1, axis=1)
ivic2019_1_std_smooth = smooth_curve(time_array, ivic2019_1_std)

ivic2019_4_mean = np.mean(ivic2019_4, axis=1)
ivic2019_4_smooth = smooth_curve(time_array, ivic2019_4_mean)
ivic2019_4_std = np.std(ivic2019_4, axis=1)
ivic2019_4_std_smooth = smooth_curve(time_array, ivic2019_1_std)

proposed_1_mean = np.mean(proposed_1, axis=0)
proposed_1_smooth = smooth_curve(time_array, proposed_1_mean)
proposed_1_std = np.std(proposed_1, axis=0)
proposed_1_std_smooth = smooth_curve(time_array, proposed_1_std)

proposed_4_mean = np.mean(proposed_4, axis=0)
proposed_4_smooth = smooth_curve(time_array, proposed_4_mean)
proposed_4_std = np.std(proposed_4, axis=0)
proposed_4_std_smooth = smooth_curve(time_array, proposed_4_std)

fig = plt.figure(figsize=(width, width * golden_ratio))
ax = fig.add_subplot(111)

# Plot the smoothed curves
ax.plot(dense_time_array, ivic2019_1_smooth, label='Ivić et al. [8] - 1 Robot', linewidth=2, zorder=5, color='tab:blue') #, marker='o', markevery=20)
ax.fill_between(dense_time_array, ivic2019_1_smooth - ivic2019_1_std_smooth, ivic2019_1_smooth + ivic2019_1_std_smooth, 
                alpha=0.3, color='tab:blue')

ax.plot(dense_time_array, ivic2019_4_smooth, label='Ivić et al. [8] - 4 Robots', linewidth=2, zorder=5, color='tab:blue', linestyle='--') #, marker='D', markevery=20)
ax.fill_between(dense_time_array, ivic2019_4_smooth - ivic2019_4_std_smooth, ivic2019_4_smooth + ivic2019_4_std_smooth, 
                alpha=0.3, color='tab:blue')

ax.plot(dense_time_array, proposed_1_smooth, label='Proposed - 1 Robot', linewidth=2, zorder=5, color='tab:orange') #, markevery=20)
ax.fill_between(dense_time_array, proposed_1_smooth - proposed_1_std_smooth, proposed_1_smooth + proposed_1_std_smooth, 
                alpha=0.3, color='tab:orange')

ax.plot(dense_time_array, proposed_4_smooth, label='Proposed - 4 Robots', linewidth=2, zorder=5, color='tab:orange', linestyle='--') #, marker='D', markevery=20)
ax.fill_between(dense_time_array, proposed_4_smooth - proposed_4_std_smooth, proposed_4_smooth + proposed_4_std_smooth, 
                alpha=0.3, color='tab:orange')

ax.legend(loc='upper right', fontsize=12, edgecolor='black', fancybox=False, shadow=True, title='Algorithms')
ax.set_xlabel('Time Step', fontdict=fontdict)
ax.set_ylabel('Uncertainty over Domain', fontdict=fontdict)
ax.grid(True, alpha=0.5)
ax.set_xlim(0, nbDataPoints)
ax.set_title('Exploration Performance', fontdict=fontdict)
plt.tight_layout()
plt.savefig('exploration_performance.pdf', bbox_inches='tight', pad_inches=0, dpi=600)
plt.show() """

# Load Data
heat_hist_path = os.path.join(datapath, "heat_hist.dat")
path_hist_path = os.path.join(datapath, "path_hist.dat")
goal_density_path = os.path.join(datapath, "goal_density.dat")
coverage_hist_path = os.path.join(datapath, "coverage_hist.dat")
mu_hist_path = os.path.join(datapath, "mu_hist.dat")
std_hist_path = os.path.join(datapath, "std_hist.dat")

# Print the file data
heat_hist = np.memmap(
    heat_hist_path, dtype=np.float64, mode='r', 
    shape=(map_data.shape[0], map_data.shape[1], nbDataPoints, nbAgents)
)
path_hist = np.memmap(
    path_hist_path, dtype=np.float64, mode='r', 
    shape=(3, nbDataPoints, nbAgents)
)
goal_density = np.memmap(
    goal_density_path, dtype=np.float64, mode='r', 
    shape=(map_data.shape[0], map_data.shape[1])
)
coverage_hist = np.memmap(
    coverage_hist_path, dtype=np.float64, mode='r', 
    shape=(map_data.shape[0], map_data.shape[1], nbDataPoints, nbAgents)
)
mu_hist = np.memmap(
    mu_hist_path, dtype=np.float64, mode='r', 
    shape=(map_data.shape[0], map_data.shape[1], nbDataPoints, nbAgents)
)
std_hist = np.memmap(
    std_hist_path, dtype=np.float64, mode='r', 
    shape=(map_data.shape[0], map_data.shape[1], nbDataPoints, nbAgents)
)
agents = [Agent(0, 0, 0, i) for i in range(nbAgents)]  # Initialize agents

# Store the data for each agent
for agent in agents:
    agent.x_hist = path_hist[:, :, agent.id].T
    agent.heat_hist = heat_hist[:, :, :, agent.id]
    agent.mu_hist = mu_hist[:, :, :, agent.id]
    agent.std_hist = std_hist[:, :, :, agent.id]
    agent.coverage_hist = coverage_hist[:, :, :, agent.id]
    agent.source_hist = np.zeros((map_data.shape[0], map_data.shape[1], nbDataPoints))
    agent.ergodic_hist = np.zeros(nbDataPoints)
    agent.combo_hist = np.zeros((map_data.shape[0], map_data.shape[1], nbDataPoints))

    for t in range(nbDataPoints):
        print(f"Processing data for agent {agent.id} at time step {t}")
        # Combo Histogram
        exp_mu = np.exp(agent.mu_hist[:, :, t])
        exp_std = np.exp(agent.std_hist[:, :, t])
        agent.combo_hist[:, :, t] = exp_mu + exp_std - 2
        
        # Source Histogram
        normalized_combo = utilities.normalize_mat(agent.combo_hist[:, :, t])
        normalized_coverage = utilities.normalize_mat(agent.coverage_hist[:, :, t])
        diff = normalized_combo - normalized_coverage
        agent.source_hist[:, :, t] = np.maximum(diff, 0) ** 2

        # Ergodic Histogram
        agent.ergodic_hist[t] = np.linalg.norm(diff)

# Connections
print("Preprocessing connections...")
adj_matrices = []
for t in range(nbDataPoints):
    adjacency_matrix = np.zeros((nbAgents, nbAgents))
    for agent in agents:
        for other_agent in agents:
            if agent.id == other_agent.id:
                continue
            if np.linalg.norm(agent.x_hist[t, :2] - other_agent.x_hist[t, :2]) <= 10:
                adjacency_matrix[agent.id, other_agent.id] = 1
    adj_matrices.append(adjacency_matrix)
            
print("Preprocessing done.")

nbDataPoints = 5000
# Time Array
time_array = np.linspace(0, nbDataPoints - 1, nbDataPoints)

# Preprocessing Map
padded_map = np.pad(map_data, 1, 'constant', constant_values=1)
occ_map = utilities.get_occupied_polygon(padded_map) - 1

# Preprocessing Data
print("Preprocessing data...")

fov_lists = [[] for _ in range(nbAgents)]

chunk_size = 500  # Adjust based on your system's memory capacity
for start_idx in range(0, nbDataPoints, chunk_size):
    print(f"Processing data from {start_idx} to {min(start_idx + chunk_size, nbDataPoints)}")
    end_idx = min(start_idx + chunk_size, nbDataPoints)
    for t in range(start_idx, end_idx):
        for agent in agents:
            fov_lists[agent.id].append(utilities.clip_polygon_no_convex(
                agent.x_hist[t, :2],
                utilities.draw_fov_arc(agent.x_hist[t, :2], agent.x_hist[t, 2], fov_span, fov_depth, 10),
                occ_map,
                True)
            )

# Remove type 3 fonts
rc('pdf', fonttype=42)
rc('ps', fonttype=42)

# Plot the ergodic metric over time
width = 7.16 # Full page width for double-column papers
golden_ratio = 0.618 # Golden ratio
evalAgent = agents[0]
fontdict = {'weight': 'bold'}
timesteps = [100, 500, 1000, nbDataPoints-1]


############################
# Plot the mean and std combination
############################
""" timestep = 1000
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.pcolormesh(grid_x, grid_y, np.where(map_data == 0, np.nan, map_data), cmap=cmap, zorder=10, rasterized=True)
normalized_std = utilities.min_max_normalize(agents[0].std_hist[:, :, timestep])
normalized_mu = utilities.min_max_normalize(agents[0].mu_hist[:, :, timestep])

levels = 10  # Number of contour levels
alpha_values = np.linspace(0, 1, levels)
alpha_values = 1

# ax1.contourf(grid_x, grid_y, normalized_mu, levels=levels, cmap='Reds', alpha=alpha_values, zorder=1)
# ax1.contour(grid_x, grid_y, normalized_mu, levels=levels, colors="black", linestyles='solid', linewidths=1, zorder=2)
ax1.contourf(grid_x, grid_y, normalized_std, levels=levels, cmap='Blues', alpha=alpha_values, zorder=3)
ax1.contour(grid_x, grid_y, normalized_std, levels=levels, colors="black", linestyles='dotted', linewidths=1, zorder=4)

# ax1.contourf(grid_x, grid_y, agents[0].combo_hist[:, :, timestep], levels=10, cmap='Greens', alpha=1, zorder=5)

initial_scatter = ax1.scatter(agents[0].x_hist[0, 0],
                                agents[0].x_hist[0, 1],
                                marker='o',
                                edgecolors='black',
                                facecolors='none',
                                s=80,
                                zorder=6,
                                linewidth=2
                                )

ax1.plot(agents[0].x_hist[:timestep, 0], agents[0].x_hist[:timestep, 1], color='black', lw=2, zorder=6, alpha=1)
ax1.scatter(agents[0].x_hist[timestep, 0],
            agents[0].x_hist[timestep, 1],
            marker='o',
            edgecolors='black',
            facecolors='tab:blue',
            s=80,
            zorder=6,
            linewidth=1.5
            )

fov_polygon = np.array(fov_lists[0][timestep])
ax1.fill(fov_polygon[:, 0],
        fov_polygon[:, 1],
        alpha=0.5,
        linewidth=2,
        zorder=5,
        edgecolor='black',
        facecolor='tab:blue'
        )

for ax in [ax1]:
    ax.set_xlim(grid_x.min(), grid_x.max())
    ax.set_ylim(grid_y.min(), grid_y.max())
    ax.set_xticks(np.arange(grid_x.min(), grid_x.max(), 10))
    ax.set_yticks(np.arange(grid_y.min(), grid_y.max(), 10))
    ax.set_aspect('equal')
    ax.set_xlabel('X [m]', fontdict=fontdict)
    ax.set_ylabel('Y [m]', fontdict=fontdict)

# divider = make_axes_locatable(ax1)
# cax = divider.append_axes("bottom", size="5%", pad=0.6)

# cbar1 = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap='Reds'), cax=cax, orientation='horizontal')
# cbar1.ax.xaxis.set_label_position('bottom')
# cbar1.ax.set_xlabel('Ground Truth', fontdict={'weight': 'bold'}, rotation=0, labelpad=0, ha='center', va='bottom')
# cbar1.outline.set_linewidth(0)
# cbar1.set_ticks([0.0, 1.0])  # Set tick positions at the beginning (0) and end (1)
# cbar1.ax.xaxis.set_tick_params(width=0)

plt.title('Pred. Unc.', fontdict=fontdict)
plt.tight_layout()
plt.savefig('pred_unc.pdf', bbox_inches='tight', pad_inches=0, dpi=600)
plt.show() """


############################
# Dynamic Goal Density
############################
""" for i in range(len(timesteps)):
    fig = plt.figure(figsize=(width, width * golden_ratio))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.pcolormesh(grid_x, grid_y, np.where(map_data == 0, np.nan, map_data), cmap=cmap, zorder=10, rasterized=True)
    ax.set_xlim(grid_x.min(), grid_x.max())
    ax.set_ylim(grid_y.min(), grid_y.max())
    ax.set_xticks(np.arange(grid_x.min(), grid_x.max(), 10))
    ax.set_yticks(np.arange(grid_y.min(), grid_y.max(), 10))
    ax.contourf(grid_x, grid_y, agents[0].combo_hist[:, :, timesteps[i]], cmap='Blues', levels=10, alpha=1)
    ax.plot(agents[0].x_hist[:timesteps[i], 0], agents[0].x_hist[:timesteps[i], 1], color='black', lw=2, zorder=2, alpha=0.3)
    initial_scatter = ax.scatter(agents[0].x_hist[0, 0],
                                    agents[0].x_hist[0, 1],
                                    marker='o',
                                    edgecolors='black',
                                    facecolors='none',
                                    s=80,
                                    zorder=5,
                                    linewidth=2
                                    )
    ax.scatter(agents[0].x_hist[timesteps[i], 0],
                agents[0].x_hist[timesteps[i], 1],
                marker='o',
                edgecolors='black',
                facecolors='black',
                s=80,
                zorder=5,
                linewidth=2
                )

    fov_polygon = np.array(fov_lists[0][timesteps[i]])
    ax.fill(fov_polygon[:, 0],
            fov_polygon[:, 1],
            alpha=0.5,
            linewidth=2,
            zorder=4,
            edgecolor='black',
            facecolor='black'
            )

    if i == 0:
        ax.set_ylabel('Y [m]', fontdict=fontdict)
        ax.set_xlabel('X [m]', fontdict=fontdict)
    else:
        ax.set_xlabel('X [m]', fontdict=fontdict)

    plt.tight_layout()
    # plt.savefig(f'dyn_goal_density_{timesteps[i]}.pdf', bbox_inches='tight', pad_inches=0, dpi=600)
# Show the plot
plt.show() """


############################
# ERGODIC METRIC
############################
""" # Chosen points
points = [1715, 3303, 5469, 6958]

for agent in agents:
    ax.plot(time_array, agent.ergodic_hist, label=f'Robot {agent.id}', linewidth=2, zorder=(5-agent.id))

# for i in range(nIntervals):
    # ax.scatter(maxTimesIndex[i], maxPoints[i], color='tab:red', zorder=5)
    # ax.scatter(minTimesIndex[i], minPoints[i], color='tab:green', zorder=5)
for i in range(len(points)):
    ax.scatter(points[i], 
               evalAgent.ergodic_hist[points[i]],  
               zorder=10,
               marker='o',
               facecolors='none',
                edgecolors='black',
               s=50,
               linewidth=2)

ax.set_xlim(0, nbDataPoints)
ax.set_ylim(0, 0.3)
ax.set_xticks(np.arange(0, nbDataPoints+1, 1000))
ax.set_yticks(np.arange(0, 0.35, 0.05))
ax.set_xlabel('Time Step', fontdict=fontdict)
ax.set_ylabel('Ergodic Metric', fontdict=fontdict)
# ax.set_title('Ergodic Metric Over Time', fontdict=fontdict)
ax.legend(loc='upper right', fontsize=12, edgecolor='black', fancybox=False, shadow=True, title='Robots')
ax.grid(True)
plt.tight_layout()
plt.savefig('ergodic_metric.pdf', bbox_inches='tight', pad_inches=0, dpi=600)
plt.show()
"""

# width = 7.16 # Full page width for double-column papers
# golden_ratio = 0.618 # Golden ratio

# Use tex
plt.rcParams['text.usetex'] = True

fig = plt.figure(figsize=(width, width * golden_ratio))
ax = fig.add_subplot(111)
ax.set_aspect('equal')
ax.set_xlabel('X [m]', fontdict={'weight': 'bold'})
ax.set_ylabel('Y [m]', fontdict={'weight': 'bold'})
ax.set_xlim(grid_x.min(), grid_x.max())
ax.set_ylim(grid_y.min(), grid_y.max())
# Set one tick every 10 meters
ax.set_xticks(np.arange(grid_x.min(), grid_x.max(), 10))
ax.set_yticks(np.arange(grid_y.min(), grid_y.max(), 10))

ax.contourf(grid_x, grid_y, goal_density, cmap='binary', levels=10, alpha=1)
ax.pcolormesh(grid_x, grid_y, np.where(map_data == 0, np.nan, map_data), cmap=cmap, zorder=11, rasterized=True)

cmaps = ["Blues", "Oranges", "Greens", "Reds"]
colors = ['blue', 'orange', 'green', 'red']
# points = [1715, 3303, 5469, 6958]
timestep = 6958
for agent in agents:
    if agent.id == 0:
        alpha = 1
        zorder = 3
    else:
        alpha = 0.3
        zorder = 2
    # Plot the initial position of the agent
    ax.scatter(agent.x_hist[0, 0], 
               agent.x_hist[0, 1], 
               marker='o', 
               s=80, 
               zorder=5,
               edgecolors='black',
               facecolors='none',
               linewidth = 2,
                alpha=alpha
               )
    
    # Plot the final position of the agent
    ax.scatter(agent.x_hist[timestep, 0],
                agent.x_hist[timestep, 1],
                color=f'C{agent.id}',
                marker='o',
                s=80,
                zorder=5,
                edgecolors='black',
                linewidth = 1.5,
                alpha=1
                )

    # Plot the FOV
    fov_polygon = np.array(fov_lists[agent.id][timestep])

    ax.fill(fov_polygon[:, 0],
            fov_polygon[:, 1],
            alpha=0.5,
            linewidth=2,
            zorder=4,
            edgecolor='black',
            facecolor=f'C{agent.id}'
            )

    # Plot the agent's path
    line = ax.plot(agent.x_hist[:timestep, 0],
                    agent.x_hist[:timestep, 1],
                    color=f'C{agent.id}',
                    lw=2,
                    zorder=zorder,
                    alpha=alpha
                    )

# Plot the connections
connection_segments = []
for i, agent in enumerate(agents):
    for j, other_agent in enumerate(agents):
        if adj_matrices[timestep][i, j] == 1:
            connection_segments.append([agent.x_hist[timestep, :2], other_agent.x_hist[timestep, :2]])
            adj_matrices[timestep][i, j] = 0
            adj_matrices[timestep][j, i] = 0

lc = LineCollection(
    segments=connection_segments,
    colors='cyan',
    linewidth=1.5,
    linestyle='dashed',
    alpha=1,
    zorder=4
)
ax.add_collection(lc)
ax.set_title(f'Time Step - {timestep}', fontdict={'weight': 'bold'})
plt.tight_layout()
# plt.savefig(f'time_step_{timestep}.pdf', bbox_inches='tight', pad_inches=0, dpi=600)
plt.show()

############################################
# TIME-DECAY PLOTS
############################################
""" datapath = os.path.join(os.getcwd(), '_datastorage/1_no_decay')
nod_heat_hist_path = os.path.join(datapath, "heat_hist.dat")
nod_path_hist_path = os.path.join(datapath, "path_hist.dat")
nod_goal_density_path = os.path.join(datapath, "goal_density.dat")
nod_coverage_hist_path = os.path.join(datapath, "coverage_hist.dat")
nod_mu_hist_path = os.path.join(datapath, "mu_hist.dat")
nod_std_hist_path = os.path.join(datapath, "std_hist.dat")

nod_heat_hist = np.memmap(
    nod_heat_hist_path, dtype=np.float64, mode='r', 
    shape=(map_data.shape[0], map_data.shape[1], nbDataPoints, nbAgents)
)
nod_path_hist = np.memmap(
    nod_path_hist_path, dtype=np.float64, mode='r', 
    shape=(3, nbDataPoints, nbAgents)
)
nod_goal_density = np.memmap(
    nod_goal_density_path, dtype=np.float64, mode='r', 
    shape=(map_data.shape[0], map_data.shape[1])
)
nod_coverage_hist = np.memmap(
    nod_coverage_hist_path, dtype=np.float64, mode='r', 
    shape=(map_data.shape[0], map_data.shape[1], nbDataPoints, nbAgents)
)
nod_mu_hist = np.memmap(
    nod_mu_hist_path, dtype=np.float64, mode='r', 
    shape=(map_data.shape[0], map_data.shape[1], nbDataPoints, nbAgents)
)
nod_std_hist = np.memmap(
    nod_std_hist_path, dtype=np.float64, mode='r', 
    shape=(map_data.shape[0], map_data.shape[1], nbDataPoints, nbAgents)
)

nod_agents = [Agent(0, 0, 0, i) for i in range(nbAgents)]  # Initialize agents

# Store the data for each agent
for agent in nod_agents:
    agent.x_hist = nod_path_hist[:, :, agent.id].T
    agent.heat_hist = nod_heat_hist[:, :, :, agent.id]
    agent.mu_hist = nod_mu_hist[:, :, :, agent.id]
    agent.std_hist = nod_std_hist[:, :, :, agent.id]
    agent.coverage_hist = nod_coverage_hist[:, :, :, agent.id]
    
    # Combo Histogram
    exp_mu = np.exp(agent.mu_hist)
    exp_std = np.exp(agent.std_hist)
    agent.combo_hist = exp_mu + exp_std - 2
    
    # Source Histogram
    normalized_combo = utilities.normalize_mat(agent.combo_hist)
    normalized_coverage = utilities.normalize_mat(agent.coverage_hist)
    agent.source_hist = np.maximum(normalized_combo - normalized_coverage, 0) ** 2

    # Ergodic Histogram
    agent.ergodic_hist = np.linalg.norm(agent.combo_hist - agent.coverage_hist, ord=2, axis=(0, 1))

# Preprocessing Data
print("Preprocessing data...")
nod_fov_lists = [[] for _ in range(nbAgents)]

chunk_size = 500  # Adjust based on your system's memory capacity
for start_idx in range(0, nbDataPoints, chunk_size):
    print(f"Processing data from {start_idx} to {min(start_idx + chunk_size, nbDataPoints)}")
    end_idx = min(start_idx + chunk_size, nbDataPoints)
    for t in range(start_idx, end_idx):
        for agent in nod_agents:
            nod_fov_lists[agent.id].append(utilities.clip_polygon_no_convex(
                agent.x_hist[t, :2],
                utilities.draw_fov_arc(agent.x_hist[t, :2], agent.x_hist[t, 2], fov_span, fov_depth, 10),
                occ_map,
                True)
            )


# Ensure type 3 fonts are not used
rc('pdf', fonttype=42)
rc('ps', fonttype=42)

fig = plt.figure()
ax = fig.add_subplot(121)
ax.set_aspect('equal')
ax.set_xlabel('X [m]', fontdict={'weight': 'bold'})
ax.set_ylabel('Y [m]', fontdict={'weight': 'bold'})
ax.set_title('With Decay')
ax.set_xlim(grid_x.min(), grid_x.max())
ax.set_ylim(grid_y.min(), grid_y.max())
# Set one tick every 10 meters
ax.set_xticks(np.arange(grid_x.min(), grid_x.max(), 10))
ax.set_yticks(np.arange(grid_y.min(), grid_y.max(), 10))

ax.contourf(grid_x, grid_y, goal_density, cmap='binary', levels=10, alpha=1)

divider = make_axes_locatable(ax)
cax = divider.append_axes("bottom", size="5%", pad=0.6)

cbar1 = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap='binary'), cax=cax, orientation='horizontal')
cbar1.ax.xaxis.set_label_position('bottom')
cbar1.ax.set_xlabel('Goal Density', fontdict={'weight': 'bold'}, rotation=0, labelpad=0, ha='center', va='bottom')
cbar1.outline.set_linewidth(0)
cbar1.set_ticks([0.0, 1.0])  # Set tick positions at the beginning (0) and end (1)
cbar1.ax.xaxis.set_tick_params(width=0)

ax.pcolormesh(grid_x, grid_y, np.where(map_data == 0, np.nan, map_data), cmap=cmap, zorder=10, rasterized=True)
# Plot the agent's path
line = colored_line(agents[0].x_hist[:-1, 0],
                    agents[0].x_hist[:-1, 1],
                    np.arange(nbDataPoints),
                    ax,
                    cmap='viridis',
                    lw=2,
                    zorder=2,
                    alpha=1
                    )

# Scatter a little triangle with the vertex pointing in the direction of the agent
ax.scatter(agents[0].x_hist[-1, 0], 
           agents[0].x_hist[-1, 1], 
           color='tab:red', 
           marker='o', 
           s=80, 
           zorder=5,
           edgecolors='black',
            linewidth = 1.5 
           )
# FOV
fov_polygon = np.array(fov_lists[0][-1])

ax.fill(fov_polygon[:, 0],
        fov_polygon[:, 1],
        alpha=0.5,
        linewidth=2,
        zorder=4,
        edgecolor='black',
        facecolor='tab:red'
        )

initial_scatters = ax.scatter(
    [agent.x_hist[0, 0] for agent in agents],
    [agent.x_hist[0, 1] for agent in agents],
    marker='o',
    edgecolors='black',
    # Remove the face color
    facecolors='none',
    s=80,
    zorder=3,
    alpha=1,
    linewidth=2
)

ax2 = fig.add_subplot(122)
ax2.set_aspect('equal')
ax2.set_xlabel('X [m]', fontdict={'weight': 'bold'})
# ax2.set_ylabel('Y (meters)', fontdict={'weight': 'bold'})
ax2.set_xlim(grid_x.min(), grid_x.max())
ax2.set_ylim(grid_y.min(), grid_y.max())
ax2.set_title('Without Decay')
# Set one tick every 10 meters
ax2.set_xticks(np.arange(grid_x.min(), grid_x.max(), 10))
ax2.set_yticks(np.arange(grid_y.min(), grid_y.max(), 10))
ax2.contourf(grid_x, grid_y, goal_density, cmap='binary', levels=10, alpha=1)
ax2.pcolormesh(grid_x, grid_y, np.where(map_data == 0, np.nan, map_data), cmap=cmap, zorder=10, rasterized=True)

divider = make_axes_locatable(ax2)
cax2 = divider.append_axes("bottom", size="5%", pad=0.6)
cbar2 = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap='viridis'), cax=cax2, orientation='horizontal')

cbar2.set_ticks([0, 1])  # Set tick positions at the beginning (0) and end (1)
cbar2.set_ticklabels(["Start", "End"])  # Set labels for the ticks
# Remove the ticks length
cbar2.ax.xaxis.set_tick_params(width=0)

cbar2.outline.set_linewidth(0)
cbar2.ax.xaxis.set_label_position('bottom')
cbar2.ax.set_xlabel('Time', fontdict={'weight': 'bold'}, rotation=0, labelpad=0, ha='center', va='bottom')

# Plot the agent's path
line2 = colored_line(nod_agents[0].x_hist[:-1, 0],
                    nod_agents[0].x_hist[:-1, 1],
                    np.arange(nbDataPoints),
                    ax2,
                    cmap='viridis',
                    lw=2,
                    zorder=2,
                    alpha=1
                    )

# Scatter a little triangle with the vertex pointing in the direction of the agent
ax2.scatter(nod_agents[0].x_hist[-1, 0],
            nod_agents[0].x_hist[-1, 1],
            marker='o',
            s=80,
            zorder=5,
            edgecolors='black',
            facecolors='tab:red',
            linewidth = 1.5
            )

# FOV
fov_polygon = np.array(nod_fov_lists[0][-1])

ax2.fill(fov_polygon[:, 0],
        fov_polygon[:, 1],
        alpha=0.5,
        linewidth=2,
        zorder=4,
        edgecolor='black',
        facecolor='tab:red'
        )

initial_scatters2 = ax2.scatter(
    [agent.x_hist[0, 0] for agent in nod_agents],
    [agent.x_hist[0, 1] for agent in nod_agents],
    marker='o',
    edgecolors='black',
    # Remove the face color
    facecolors='none',
    s=80,
    zorder=3,
    alpha=1,
    linewidth=2
)
fig.suptitle('Time-Decay Effect', fontweight='bold', fontsize=14, y=0.9)
# fig.set_size_inches(8.6 / 2.54, 6 / 2.54)
plt.tight_layout()
plt.savefig('time_decay_effect.pdf', bbox_inches='tight', pad_inches=0, dpi=600)
plt.show()
"""

# Ergodic Metric Plot

# find the min and max in some intervals
nIntervals = nbDataPoints // 2000
intervalSize = nbDataPoints // nIntervals
maxPoints = []
minPoints = []
maxTimesIndex = []
minTimesIndex = []
evalAgent = agents[0]
for i in range(nIntervals):
    start = i * intervalSize
    if i == 0:
        start = 1000
    end = (i + 1) * intervalSize
    maxPoints.append(np.max(evalAgent.ergodic_hist[start:end]))
    maxTimesIndex.append(np.argmax(evalAgent.ergodic_hist[start:end]) + start)
    minPoints.append(np.min(evalAgent.ergodic_hist[start:end]))
    minTimesIndex.append(np.argmin(evalAgent.ergodic_hist[start:end]) + start)

print("Max Points Indexes")
print("maxTimesIndex", maxTimesIndex)
print("Min Points Indexes")
print("minTimesIndex", minTimesIndex)

# Width of a double column paper
width = 7.16 # Full page width for double-column papers
golden_ratio = 0.5 # Golden ratio




fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_aspect('equal')
ax.set_xlabel('X (meters)', fontdict={'weight': 'bold'})
ax.set_ylabel('Y (meters)', fontdict={'weight': 'bold'})
ax.set_xlim(grid_x.min(), grid_x.max())
ax.set_ylim(grid_y.min(), grid_y.max())
ax.set_title('Ergodic Control Simulation', fontdict={'weight': 'bold'})
ax.pcolormesh(grid_x, grid_y, np.where(map_data == 0, np.nan, map_data), cmap=cmap, zorder=10)
# ax.contourf(grid_x, grid_y, goal_density, cmap='coolwarm', levels=10)

colorbar = plt.colorbar(ax.contourf(grid_x, grid_y, goal_density, cmap='plasma', levels=10, alpha=1), 
                        ax=ax, 
                        fraction=0.046, 
                        pad=0.04,
                        label='Goal Density')
colorbar.ax.yaxis.set_tick_params(width=0)
colorbar.outline.set_linewidth(0)
colorbar.ax.set_ylabel('Goal Density', fontdict={'weight': 'bold'})
# Reuce the padding between the colorbar and the label
# colorbar.ax.yaxis.labelpad = 0.5

timestep = 240
cmaps = ["Blues", "Oranges", "Greens", "Reds"]
for i, agent in enumerate(agents):
    # ax.plot(agent.x_hist[:timestep, 0], 
    #         agent.x_hist[:timestep, 1], 
    #         color=f'C{i}', 
    #         lw=2, 
    #         zorder=2)
    line = colored_line(agent.x_hist[:timestep, 0], 
                        agent.x_hist[:timestep, 1], 
                        np.arange(timestep), 
                        ax, 
                        cmap=cmaps[i], 
                        lw=2, 
                        zorder=2, 
                        alpha=0.8)

    # Scatter a little triangle with the vertex pointing in the direction of the agent
    ax.scatter(agent.x_hist[timestep, 0], 
               agent.x_hist[timestep, 1], 
               color=f'C{i}', 
               marker='o', 
               s=80, 
               zorder=5,
               edgecolors='black',
               linewidth = 1.5)

    connection_segments = []
    for j, other_agent in enumerate(agents):
        if adj_matrices[timestep][i, j] == 1:
            connection_segments.append([agent.x_hist[timestep, :2], other_agent.x_hist[timestep, :2]])
            adj_matrices[timestep][i, j] = 0
            adj_matrices[timestep][j, i] = 0
    lc = LineCollection(
        segments=connection_segments,
        colors='cyan',
        linewidth=2,
        linestyle='dashed',
        alpha=0.8,
        zorder=3
    )
    ax.add_collection(lc)

    initial_scatters = ax.scatter(
        [agent.x_hist[0, 0] for agent in agents],
        [agent.x_hist[0, 1] for agent in agents],
        marker='s',
        edgecolors='black',
        facecolors=[f'C{i}' for i in range(nbAgents)],
        linewidth = 1,
        s=60,
        zorder=3,
        alpha=0.8
    )

    # FOV
    fov_polygon = np.array(fov_lists[agent.id][timestep])
    ax.fill(fov_polygon[:, 0], 
            fov_polygon[:, 1],
            alpha=0.3,
            color='#D3D3D3',
            linewidth=1,
            zorder=1
    )
plt.show()


# Initialize figure and static elements
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')
ax.set_xlabel('X (meters)', fontdict={'weight': 'bold'})
ax.set_ylabel('Y (meters)', fontdict={'weight': 'bold'})
ax.set_xlim(grid_x.min(), grid_x.max())
ax.set_ylim(grid_y.min(), grid_y.max())
ax.set_title('Ergodic Control Simulation', fontdict={'weight': 'bold'})
cmap = mcolors.ListedColormap(['black'])
ax.pcolormesh(grid_x, grid_y, np.where(map_data == 0, np.nan, map_data), cmap=cmap, zorder=10)
colorbar = plt.colorbar(ax.contourf(grid_x, grid_y, goal_density, cmap='plasma', levels=10, alpha=1), 
                        ax=ax, 
                        fraction=0.046, 
                        pad=0.04,
                        label='Goal Density')
colorbar.ax.yaxis.set_tick_params(width=0)
colorbar.outline.set_linewidth(0)
colorbar.ax.set_ylabel('Goal Density', fontdict={'weight': 'bold'})
cmaps = ["Blues", "Oranges", "Greens", "Reds"]

initial_scatters = ax.scatter(
    [agent.x_hist[0, 0] for agent in agents],
    [agent.x_hist[0, 1] for agent in agents],
    marker='s',
    edgecolors='black',
    facecolors=[f'C{i}' for i in range(nbAgents)],
    linewidth = 1,
    s=60,
    zorder=3,
    alpha=0.5
)

paths = [
    ax.plot([], [],
            color=f'C{i}',
            lw=2,
            zorder=2,
            alpha=0.5
            )[0] for i in range(nbAgents)
]

fovs = [
    ax.fill([], [], 
            color='#D3D3D3', 
            alpha=0.3, 
            linewidth=1,
            zorder=1
            )[0] for _ in agents
]

scatters = [
    ax.scatter([], [],
                color=f'C{i}',
                marker='o',
                s=80,
                zorder=5,
                edgecolors='black',
                linewidth = 1.5
                ) for i in range(nbAgents)
]

# The connections between agents
connections = [
    LineCollection(
        segments=[],
        colors='cyan',
        linewidth=2,
        linestyle='dashed',
        alpha=0.8,
        zorder=4
    ) for _ in range(1)
]

# Update function
def update(frame):
    connection_segments = []
    for idx, agent in enumerate(agents):
        x_hist = agent.x_hist
        fov_list = fov_lists[agent.id]

        # Update agent path
        paths[idx].set_data(x_hist[:frame, 0], x_hist[:frame, 1])
            
        # Update FOV
        fov_polygon = np.array(fov_list[frame])
        fovs[idx].get_path().vertices = fov_polygon
        fovs[idx].set_xy(fov_polygon)

        # Update agent position
        scatters[idx].set_offsets(x_hist[frame, :2])

        # Update connections

        for i in range(nbAgents):
            for j in range(nbAgents):
                if adj_matrices[frame][i, j] == 1:
                    connection_segments.append([agents[i].x_hist[frame, :2], agents[j].x_hist[frame, :2]])
                    # Set to 0 to avoid duplicate connections
                    adj_matrices[frame][i, j] = 0
                    adj_matrices[frame][j, i] = 0
    connections[0].set_segments(connection_segments)
    ax.add_collection(connections[0])

    # Update the frame title
    ax.set_title("Ergodic Control Simulation" + "\n" + f"Time Step: {frame}/{nbDataPoints}", fontdict={'weight': 'bold'})
    # Tight layout
    plt.tight_layout()
    return paths + fovs + scatters + connections

anim = FuncAnimation(fig, update, frames=range(0, nbDataPoints, 10), interval=30, blit=True)
anim.save('ergodic_control_animation_multi.mp4', writer='ffmpeg', fps=30, dpi=300, progress_callback=lambda i, n: print(f'Saving frame {i} of {n}'))
plt.close()
