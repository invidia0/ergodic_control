from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
import os
from numba import jit
from ergodic_control import utilities

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
# @jit(nopython=True)
# def compute_midpoints_and_segments(x, y):
#     """Compute midpoints and create segments for colored line"""
#     x_mid = 0.5 * (x[1:] + x[:-1])
#     y_mid = 0.5 * (y[1:] + y[:-1])
#     x_midpts = np.zeros(len(x_mid) + 2)
#     y_midpts = np.zeros(len(y_mid) + 2)
#     x_midpts[0] = x[0]
#     x_midpts[1:-1] = x_mid
#     x_midpts[-1] = x[-1]
#     y_midpts[0] = y[0]
#     y_midpts[1:-1] = y_mid
#     y_midpts[-1] = y[-1]
    
#     n = len(x)
#     segments = np.zeros((n-1, 3, 2))
#     for i in range(n-1):
#         segments[i, 0] = [x_midpts[i], y_midpts[i]]  # start
#         segments[i, 1] = [x[i], y[i]]                # middle
#         segments[i, 2] = [x_midpts[i+1], y_midpts[i+1]]  # end
        
#     return segments

# Preload and cache data
map_name = 'simpleMap_05'
map_path = os.path.join(os.getcwd(), 'example_maps/', map_name + '.npy')
datapath = os.path.join(os.getcwd(), '_datastorage/')
# Load with memmap
map_data = np.load(map_path)

fov_span = 90
fov_depth = 5
grid_x, grid_y = np.meshgrid(np.arange(map_data.shape[0]), 
                              np.arange(map_data.shape[1]), indexing='ij')
nbAgents = 4
nbDataPoints = 5000
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
                True
            )
            )

# segments_lists = [[] for _ in range(nbAgents)]

# for start_idx in range(0, nbDataPoints, chunk_size):
#     print(f"Processing data from {start_idx} to {min(start_idx + chunk_size, nbDataPoints)}")
#     end_idx = min(start_idx + chunk_size, nbDataPoints)
#     for t in range(start_idx, end_idx):
#         if t == 0:
#             continue
#         for agent in agents:
#             segments_lists[agent.id].append(compute_midpoints_and_segments(agent.x_hist[:t, 0], agent.x_hist[:t, 1]))
print("Preprocessing done.")



# Initialize figure and static elements
fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))
ax1.set_aspect('equal')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Ergodic Control Simulation')

# Plot static elements
contour_goal = ax1.contourf(grid_x, grid_y, goal_density, cmap='Reds', levels=10)
map_plot1 = ax1.pcolormesh(grid_x, grid_y, np.where(map_data == 0, np.nan, map_data), cmap='gray', zorder=10)

# Create placeholders for dynamic elements
paths = [
    ax1.plot([], [], color='tab:blue', lw=2)[0] for _ in agents
]
fovs = [
    ax1.fill([], [], color='tab:blue', alpha=0.5)[0] for _ in agents
]
scatters = [
    ax1.scatter([], [], color='tab:red', marker='o', s=50) for _ in agents
]

# Update function
def update(frame):
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

    # Update the frame title
    fig.suptitle(f"Frame: {frame}/{nbDataPoints}", fontsize=16)
    print(f"Frame: {frame}/{nbDataPoints}")
    return paths + fovs + scatters

anim = FuncAnimation(fig, update, frames=range(0, nbDataPoints, 10), interval=30, blit=True)
anim.save('ergodic_control_animation.mp4', writer='ffmpeg')
plt.close()
