import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from ergodic_control import utilities, models
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

def plot_colored_trajectory(xy, metric_values, cmap='viridis_r'):
    # Create line segments from point data
    points = xy.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Average the ergodic metric between each pair of points to match segment count
    metric_avg = 0.5 * (metric_values[:-1] + metric_values[1:])
    
    # Create line collection with color based on ergodic metric
    norm = Normalize(vmin=np.min(metric_avg), vmax=np.max(metric_avg))
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(metric_avg)
    lc.set_linewidth(3)
    # Set zorder to ensure it is drawn above the map
    lc.set_zorder(9)
    return lc


# Load the data
ergodic_metric = np.load('video_ergodic_metric.npy')
agents_data = np.load('video_agents_data.npy', allow_pickle=True)
debug_mu = np.load('video_mu.npy')
debug_std = np.load('video_std.npy')
debug_combo_density = np.load('video_combo_density.npy')

# Normalize the ergodic metric
ergodic_metric = utilities.min_max_normalize(ergodic_metric)

# Animation parameters
total_steps = len(ergodic_metric)
fps = 30  # frames per second
duration = 30  # seconds
total_frames = fps * duration
step_per_frame = max(1, total_steps // total_frames)
process_change_step = 5000

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111)


# Set the map
map_name = 'simpleMap_05'
# Load the map
map_path = os.path.join(os.getcwd(), 'example_maps/', map_name + '.npy')
map = np.load(map_path)
x_min, x_max = 0, map.shape[0]
y_min, y_max = 0, map.shape[1]
grid_x, grid_y = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max), indexing='ij')
grid = np.vstack([grid_x.flatten(), grid_y.flatten()]).T
free_cells = np.array(np.where(map == 0)).T

# Is the map closed?
closed_map = True
# Extend the map with 1 cell to avoid index out of bounds
padded_map = np.pad(map, 1, 'constant', constant_values=1)
occ_map = utilities.get_occupied_polygon(padded_map)
occ_map = occ_map - 1

means = np.array([[30, 8], [10, 10]])
cov = np.array([[[10, 0], [0, 10]]])


density_map = utilities.gauss_pdf(grid, means[0], cov[0])
density_map = utilities.min_max_normalize(density_map)

norm_density_map = utilities.min_max_normalize(density_map).reshape(map.shape)
norm_density_map = density_map.reshape(map.shape)

goal_density = np.zeros_like(map)
goal_density[free_cells[:, 0], free_cells[:, 1]] = norm_density_map[free_cells[:, 0], free_cells[:, 1]]
# Min-max normalize the goal density
goal_density = np.abs(goal_density)
goal_density_0 = utilities.min_max_normalize(goal_density) # remember to normalize

density_map = utilities.gauss_pdf(grid, means[1], cov[0])
density_map = utilities.min_max_normalize(density_map)

norm_density_map = utilities.min_max_normalize(density_map).reshape(map.shape)
norm_density_map = density_map.reshape(map.shape)

goal_density = np.zeros_like(map)
goal_density[free_cells[:, 0], free_cells[:, 1]] = norm_density_map[free_cells[:, 0], free_cells[:, 1]]
# Min-max normalize the goal density
goal_density = np.abs(goal_density)
goal_density_1 = utilities.min_max_normalize(goal_density) # remember to normalize


fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111)
ax.contourf(grid_x, grid_y, goal_density_0, cmap='Reds', alpha=1, zorder=0)
ax.pcolormesh(grid_x, grid_y, np.where(map == 0, np.nan, map), cmap='gray', rasterized=True)

ax.set_xlim(x_min, x_max-1)
ax.set_ylim(y_min, y_max-1)
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_title('Density Map with Obstacles')
ax.set_aspect('equal', adjustable='box')
min_safe_range = 1
sensing_range = 5
fov_depth = 5
fov_deg = 90

default_fov = utilities.init_fov(fov_deg, fov_depth)

def draw_fov(ax, agent_trajectory, step, color='purple'):
    fov = default_fov.copy()
    current_fov = utilities.rotate_and_translate(fov, agent_trajectory[step, :2], agent_trajectory[step, 2])
    current_fov = utilities.clip_polygon_no_convex(agent_trajectory[step, :2], current_fov, occ_map, closed_map)
    arc_fov = utilities.draw_fov_arc(agent_trajectory[step, :2], agent_trajectory[step, 2], fov_deg, fov_depth, num_points=50)
    ax.fill(arc_fov[:, 0], arc_fov[:, 1], color=color, alpha=0.3, edgecolor='none', zorder=4)

# for step in range(0, total_steps, step_per_frame):
#     ax.clear()
#     ax.pcolormesh(grid_x, grid_y, np.where(map == 0, np.nan, map), cmap='gray', rasterized=True)
#     if step >= process_change_step:
#         ax.contourf(grid_x, grid_y, goal_density_1, cmap='Reds', alpha=1, zorder=0)
#     else:
#         ax.contourf(grid_x, grid_y, goal_density_0, cmap='Reds', alpha=1, zorder=0)
#     for agent_data in enumerate(agents_data):
#         agent_id, agent_trajectory = agent_data
#         if agent_id == 0:  # Only plot agent 0
#             ax.plot(agent_trajectory[:step+1, 0], agent_trajectory[:step+1, 1], color='purple', linewidth=3, zorder=2)
#             # Scatter current position
#             ax.scatter(agent_trajectory[step, 0], agent_trajectory[step, 1], color='purple', s=50, zorder=3)
#             # Circle the min safe range
#             circle = plt.Circle((agent_trajectory[step, 0], agent_trajectory[step, 1]), min_safe_range, color=f'C{agent_id}', fill=False, linestyle='--')
#             ax.add_artist(circle)
#             # Circle the sens range
#             circle = plt.Circle((agent_trajectory[step, 0], agent_trajectory[step, 1]), sensing_range, color=f'C{agent_id}', fill=False, linestyle=':')
#             ax.add_artist(circle)
#             # Draw FOV
#             draw_fov(ax, agent_trajectory, step, color='purple')
#         else:
#             ax.plot(agent_trajectory[:step+1, 0], agent_trajectory[:step+1, 1], color=f'C{agent_id}', alpha=0.1, linewidth=2, zorder=1)
#             ax.scatter(agent_trajectory[step, 0], agent_trajectory[step, 1], color=f'C{agent_id}', s=30, zorder=3)
#             # Circle the min safe range
#             circle = plt.Circle((agent_trajectory[step, 0], agent_trajectory[step, 1]), min_safe_range, color=f'C{agent_id}', fill=False, linestyle='--')
#             ax.add_artist(circle)
#             # Circle the sens range
#             circle = plt.Circle((agent_trajectory[step, 0], agent_trajectory[step, 1]), sensing_range, color=f'C{agent_id}', fill=False, linestyle=':')
#             ax.add_artist(circle)

#     ax.set_xlim(x_min, x_max-1)
#     ax.set_ylim(y_min, y_max-1)
#     ax.set_xlabel('X Coordinate')
#     ax.set_ylabel('Y Coordinate')
#     ax.set_title(f'Agent 0 Trajectory with Ergodic Metric (Step: {step}/{total_steps-1})')
#     ax.set_aspect('equal', adjustable='box')
#     ax.grid(True, alpha=0.3)
#     plt.pause(0.1)


# plt.tight_layout()
# plt.show()

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

def animate(frame):
    print(f"Animating frame {frame+1}/{total_frames}")
    ax.clear()
    
    # Calculate current step
    current_step = min(frame * step_per_frame, total_steps - 1)
    
    # Set background map based on current step
    # if current_step >= process_change_step:
    #     ax.contourf(grid_x, grid_y, goal_density_1, cmap='Reds', alpha=1, zorder=0)
    # else:
    #     ax.contourf(grid_x, grid_y, goal_density_0, cmap='Reds', alpha=1, zorder=0)
    ax.contourf(grid_x, grid_y, debug_mu[current_step], cmap='Purples', alpha=1, zorder=0)

    ax.pcolormesh(grid_x, grid_y, np.where(map == 0, np.nan, map), cmap='gray', rasterized=True)
    
    # Plot all agents
    for agent_id, agent_trajectory in enumerate(agents_data):
        if agent_id == 0:  # Agent 0 with colored trajectory
            # if current_step > 0:
                # # Get trajectory up to current step
                # trajectory_segment = agent_trajectory[:current_step+1, :2]
                # metric_segment = ergodic_metric[:current_step+1, agent_id]
                
                # # Create colored trajectory using ergodic metric
                # lc = plot_colored_trajectory(trajectory_segment, metric_segment)
                # ax.add_collection(lc)
            ax.plot(agent_trajectory[:current_step+1, 0], agent_trajectory[:current_step+1, 1], 
                    color=f'C{agent_id}', linewidth=3, zorder=2)
            
            # Current position
            ax.scatter(agent_trajectory[current_step, 0], agent_trajectory[current_step, 1], 
                      color=f'C{agent_id}', s=50, zorder=3)
            
            # # Min safe range circle
            # circle = plt.Circle((agent_trajectory[current_step, 0], agent_trajectory[current_step, 1]), 
            #                    min_safe_range, color=f'C{agent_id}', fill=False, linestyle='--', zorder=5)
            # ax.add_artist(circle)
            
            # # Sensing range circle
            # circle = plt.Circle((agent_trajectory[current_step, 0], agent_trajectory[current_step, 1]), 
            #                    sensing_range, color=f'C{agent_id}', fill=False, linestyle=':', zorder=5)
            # ax.add_artist(circle)
            
            # Draw FOV
            draw_fov(ax, agent_trajectory, current_step, color=f'C{agent_id}')
            
        # else:  # Other agents in gray/low alpha
        #     # Plot trajectory up to current step
        #     ax.plot(agent_trajectory[:current_step+1, 0], agent_trajectory[:current_step+1, 1], 
        #            color=f'C{agent_id}', alpha=0.1, linewidth=2, zorder=1)
            
        #     # Current position
        #     ax.scatter(agent_trajectory[current_step, 0], agent_trajectory[current_step, 1], 
        #               color=f'C{agent_id}', s=30, alpha=0.5, zorder=3)
            
        #     # Min safe range circle
        #     circle = plt.Circle((agent_trajectory[current_step, 0], agent_trajectory[current_step, 1]), 
        #                        min_safe_range, color=f'C{agent_id}', fill=False, linestyle='--', alpha=0.5, zorder=5)
        #     ax.add_artist(circle)
            
        #     # Sensing range circle
        #     circle = plt.Circle((agent_trajectory[current_step, 0], agent_trajectory[current_step, 1]), 
        #                        sensing_range, color=f'C{agent_id}', fill=False, linestyle=':', alpha=0.5, zorder=5)
        #     ax.add_artist(circle)

    # Set axis properties
    ax.set_xlim(x_min, x_max-1)
    ax.set_ylim(y_min, y_max-1)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(f'Agent 0 Prediction Mean (Step: {current_step}/{total_steps-1})')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)

# Create animation
anim = FuncAnimation(fig, animate, frames=total_frames, interval=1000/fps, repeat=True)

# Save as video (MP4)
print("Saving video...")
anim.save('agent_prediction_mean_animation.mp4', writer='ffmpeg', fps=fps, bitrate=1800)
print("Video saved as 'agent_prediction_mean_animation.mp4'")

# # Save as GIF (optional)
# print("Saving GIF...")
# anim.save('agent_trajectory_animation.gif', writer='pillow', fps=fps//2)  # Lower fps for GIF
# print("GIF saved as 'agent_trajectory_animation.gif'")

plt.tight_layout()
plt.show()


""" # Create figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Fixed y-axis limits for consistency
y_min = np.min(ergodic_metric)
y_max = np.max(ergodic_metric)

def animate(frame):
    print(f"Animating frame {frame+1}/{total_frames}")
    ax.clear()
    
    # Calculate current step
    current_step = min(frame * step_per_frame, total_steps - 1)
    
    # Plot all agents up to current step in gray
    for i in range(1, ergodic_metric.shape[1]):
        ax.plot(ergodic_metric[:current_step+1, i], color='lightgray', alpha=0.7, linewidth=2, zorder=1)
    
    # Create colored line for agent 0 up to current step
    if current_step > 0:
        time_steps = np.arange(current_step + 1)
        agent_0_metrics = ergodic_metric[:current_step+1, 0]
        
        # Create points for line collection
        points = np.array([time_steps, agent_0_metrics]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Average the metric values for coloring
        metric_avg = 0.5 * (agent_0_metrics[:-1] + agent_0_metrics[1:])
        
        # Use global normalization for consistent colors
        global_metric_min = np.min(ergodic_metric[:, 0])
        global_metric_max = np.max(ergodic_metric[:, 0])
        norm = Normalize(vmin=global_metric_min, vmax=global_metric_max)
        
        lc = LineCollection(segments, cmap='viridis_r', norm=norm)
        lc.set_array(metric_avg)
        lc.set_linewidth(3)
        lc.set_zorder(2)
        ax.add_collection(lc)
    if current_step >= 5000:
        # Draw a vertical line at the current step
        ax.axvline(5000, color='black', linestyle='--', alpha=0.7, linewidth=2, label='Process Change Point')

    # Set consistent axis properties - FIXED ZOOM TO SHOW FULL RANGE
    ax.set_xlim(0, total_steps)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Ergodic Metric')
    ax.set_title(f'Ergodic Metric Over Time (Step: {current_step}/{total_steps-1})')
    
    # Add vertical line showing current position
    ax.axvline(current_step, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    # Create custom legend
    from matplotlib.lines import Line2D
    custom_handles = [
        Line2D([0], [0], color='lightgray', lw=2, alpha=0.7, label='Other Agents'),
        Line2D([0], [0], color='purple', lw=3, label='Agent 0 (Colored by Metric)'),
        Line2D([0], [0], color='red', lw=2, linestyle='--', alpha=0.7, label='Current Step'),
        Line2D([0], [0], color='black', lw=2, linestyle='--', alpha=0.7, label='Process Change Point')
    ]
    ax.legend(handles=custom_handles, loc='upper right')
    ax.grid(True, alpha=0.3)

# Create animation
anim = FuncAnimation(fig, animate, frames=total_frames, interval=1000/fps, repeat=True)

# Save as video (MP4)
print("Saving video...")
anim.save('ergodic_metric_animation.mp4', writer='ffmpeg', fps=fps, bitrate=1800)
print("Video saved as 'ergodic_metric_animation.mp4'")

plt.tight_layout()
plt.show() """