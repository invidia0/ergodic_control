from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
import os
from numba import jit
from ergodic_control import utilities
import warnings
from matplotlib import rc

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
datapath = os.path.join(os.getcwd(), '_datastorage/mrs_2/')
# Load with memmap
map_data = np.load(map_path)

fov_span = 90
fov_depth = 5
grid_x, grid_y = np.meshgrid(np.arange(map_data.shape[0]), 
                              np.arange(map_data.shape[1]), indexing='ij')
nbAgents = 4
nbDataPoints = 15000
# Load Data
heat_hist_path = os.path.join(datapath, "heat_hist.dat")
path_hist_path = os.path.join(datapath, "path_hist.dat")
goal_density_path = os.path.join(datapath, "goal_density.dat")
coverage_hist_path = os.path.join(datapath, "coverage_hist.dat")
mu_hist_path = os.path.join(datapath, "mu_hist.dat")
std_hist_path = os.path.join(datapath, "std_hist.dat")

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

nbDataPoints = 15000
# Time Array
time_array = np.linspace(0, nbDataPoints - 1, nbDataPoints)

# Ergodic Metric Plot
fig, ax = plt.subplots()
for agent in agents:
    ax.plot(time_array, agent.ergodic_hist, label=f'Agent {agent.id}', linewidth=2)
ax.set_xlabel('Time Step')
ax.set_ylabel('Ergodic Metric')
ax.set_title('Ergodic Metric vs Time Step')
ax.legend()
ax.grid()
plt.show()

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

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_aspect('equal')
ax.set_xlabel('X (meters)', fontdict={'weight': 'bold'})
ax.set_ylabel('Y (meters)', fontdict={'weight': 'bold'})
ax.set_xlim(grid_x.min(), grid_x.max())
ax.set_ylim(grid_y.min(), grid_y.max())
ax.set_title('Ergodic Control Simulation', fontdict={'weight': 'bold'})
import matplotlib.colors as mcolors
cmap = mcolors.ListedColormap(['black'])
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
