from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
import os
from ergodic_control import models, utilities


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

runs = 3
nbRobots = 4
steps = 10000

fontsize = 16
plt.rcParams.update({'font.size': fontsize})
ergodic_metrics = np.zeros((runs, steps, nbRobots))  # Shape: (runs, steps, nbRobots)
ergodic_metric_base = np.zeros((runs, steps))  # Shape: (runs, steps)
agents_data = np.zeros((runs, nbRobots, steps, 3))  # Shape: (runs, nbRobots, steps, 2)
for run in range(runs):
    # Load the ergodic metric for the run ( 11, 12, 13)
    ergodic_metrics[run, :] = np.load(f'ergodic_metric_1{run+1}.npy')
    # Load the agents' data for the run
    agents_data[run, :] = np.load(f'agents_data_1{run+1}.npy')  # Shape: (nbRobots, steps, 2)

# Normalize data to ensure consistency
ergodic_metrics = (ergodic_metrics - np.min(ergodic_metrics)) / (np.max(ergodic_metrics) - np.min(ergodic_metrics))
# Average the ergodic metric across runs
ergodic_metric_mean = np.mean(ergodic_metrics, axis=0)  # Shape: (steps, nbRobots)
# Mean of the mean ergodic metric across runs
ergodic_metric = np.mean(ergodic_metric_mean, axis=1)
# Standard deviation of the ergodic metric across runs
ergodic_metric_std = np.std(ergodic_metrics, axis=0)  # Shape: (runs, steps, nbRobots)
# Mean of the standard deviation across runs
ergodic_metric_std_mean = np.mean(ergodic_metric_std, axis=1)


# Load the ergodic metric from the centralized benchmark
for run in range(runs):
    # Load the ergodic metric for the run
    ergodic_metric_base[run] = np.load(f'ergodic_metric_base_change_{run + 1}.npy')  # Shape: (steps, nbRobots)

# Normalize data to ensure consistency
ergodic_metric_base = (ergodic_metric_base - np.min(ergodic_metric_base)) / (np.max(ergodic_metric_base) - np.min(ergodic_metric_base))

# Average the ergodic metric across runs
ergodic_metric_mean_base = np.mean(ergodic_metric_base, axis=0)  # Shape: (steps, nbRobots)
# Std deviation of the ergodic metric across runs
ergodic_metric_std_base = np.std(ergodic_metric_base, axis=0)

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
for agent_id in range(nbRobots):
    ax.plot(ergodic_metric_mean[:, agent_id], label=f'Robot {agent_id}', alpha=1, linewidth=2)
    # Plot the standard deviation as a shaded area
    ax.fill_between(range(steps), 
                     ergodic_metric_mean[:, agent_id] - ergodic_metric_std[:, agent_id],
                     ergodic_metric_mean[:, agent_id] + ergodic_metric_std[:, agent_id],
                     alpha=0.2)
ax.plot(ergodic_metric_mean_base, label='Ivić et. al. [11]', linestyle='-', color='purple', linewidth=2)
# Plot the standard deviation for the centralized benchmark
ax.fill_between(range(steps),
                ergodic_metric_mean_base - ergodic_metric_std_base,
                ergodic_metric_mean_base + ergodic_metric_std_base,
                alpha=0.2, color='purple')

ax.axvline(5000, color='black', linestyle='--', linewidth=2, label='Process Change')
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_xlabel('Time Step')
ax.set_ylabel('Ergodic Metric Value (lower is better)')
# Limit to 2 decimals for y-axis ticks
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
# Set y-axis limits
plt.title('Dynamic Process - Ergodic Metric Over Time')
# Legend on two columns three rows
ax.legend(ncol=2, loc='lower right', fontsize=14)
plt.tight_layout()
# plt.savefig('ergodic_metric_over_time_tv.pdf', bbox_inches='tight', dpi=300)
plt.show()

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
ax.plot(ergodic_metric, label='Ergodic Metric Mean')
ax.fill_between(range(steps),
                ergodic_metric - ergodic_metric_std_mean,
                ergodic_metric + ergodic_metric_std_mean,
                alpha=0.2)
ax.plot(ergodic_metric_mean_base, label='Centralized Benchmark', linestyle='--', color='black', linewidth=2)
# Plot the standard deviation for the centralized benchmark
ax.fill_between(range(steps),
                ergodic_metric_mean_base - ergodic_metric_std_base,
                ergodic_metric_mean_base + ergodic_metric_std_base,
                alpha=0.2, color='black')

ax.grid(True, linestyle='--', alpha=0.5)
ax.set_xlabel('Time Step')
ax.set_ylabel('Ergodic Metric Value')
ax.set_title('Ergodic Metric Mean Over Time')
ax.legend()
plt.tight_layout()
plt.show() 


fontsize = 16
plt.rcParams.update({'font.size': fontsize})
ergodic_metrics = np.zeros((runs, steps, nbRobots))  # Shape: (runs, steps, nbRobots)
ergodic_metric_base = np.zeros((runs, steps))  # Shape: (runs, steps)
agents_data = np.zeros((runs, nbRobots, steps, 3))  # Shape: (runs, nbRobots, steps, 2)
for run in range(runs):
    # Load the ergodic metric for the run
    ergodic_metrics[run, :] = np.load(f'experiments/static/ergodic_metric_{run + 1:02d}.npy')  # Shape: (steps, nbRobots)
    # Load the agents' data for the run
    agents_data[run, :] = np.load(f'experiments/static/agents_data_{run + 1:02d}.npy')  # Shape: (nbRobots, steps, 2)

# Normalize data to ensure consistency
ergodic_metrics = (ergodic_metrics - np.min(ergodic_metrics)) / (np.max(ergodic_metrics) - np.min(ergodic_metrics))
# Average the ergodic metric across runs
ergodic_metric_mean = np.mean(ergodic_metrics, axis=0)  # Shape: (steps, nbRobots)
# Mean of the mean ergodic metric across runs
ergodic_metric = np.mean(ergodic_metric_mean, axis=1)
# Standard deviation of the ergodic metric across runs
ergodic_metric_std = np.std(ergodic_metrics, axis=0)  # Shape: (runs, steps, nbRobots)
# Mean of the standard deviation across runs
ergodic_metric_std_mean = np.mean(ergodic_metric_std, axis=1)

# Load the ergodic metric from the centralized benchmark
for run in range(runs):
    # Load the ergodic metric for the run
    ergodic_metric_base[run] = np.load(f'ergodic_metric_base_{run + 1}.npy')  # Shape: (steps, nbRobots)

# Average the ergodic metric across runs
ergodic_metric_mean_base = np.mean(ergodic_metric_base, axis=0)  # Shape: (steps, nbRobots)
# Std deviation of the ergodic metric across runs
ergodic_metric_std_base = np.std(ergodic_metric_base, axis=0)

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
for agent_id in range(nbRobots):
    ax.plot(ergodic_metric_mean[:, agent_id], label=f'Robot {agent_id}', alpha=1, linewidth=2)
    # Plot the standard deviation as a shaded area
    ax.fill_between(range(steps), 
                     ergodic_metric_mean[:, agent_id] - ergodic_metric_std[:, agent_id],
                     ergodic_metric_mean[:, agent_id] + ergodic_metric_std[:, agent_id],
                     alpha=0.2)
# ax.plot(ergodic_metric_mean_base, label='Centralized Benchmark', linestyle='--', color='black', linewidth=2)
# # Plot the standard deviation for the centralized benchmark
# ax.fill_between(range(steps),
#                 ergodic_metric_mean_base - ergodic_metric_std_base,
#                 ergodic_metric_mean_base + ergodic_metric_std_base,
#                 alpha=0.2, color='black')

ax.grid(True, linestyle='--', alpha=0.5)
ax.set_xlabel('Time Step')
ax.set_ylabel('Ergodic Metric Value (lower is better)')
# Limit to 2 decimals for y-axis ticks
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
# Set y-axis limits
plt.title('Static Process - Ergodic Metric Over Time')
ax.legend(ncol=2, loc='lower right', fontsize=14)

plt.tight_layout()
# plt.savefig('ergodic_metric_over_time_each_robot.pdf', bbox_inches='tight', dpi=300)
plt.show()

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
ax.plot(ergodic_metric, label='Ergodic Metric Mean')
ax.fill_between(range(steps),
                ergodic_metric - ergodic_metric_std_mean,
                ergodic_metric + ergodic_metric_std_mean,
                alpha=0.2)
ax.plot(ergodic_metric_mean_base, label='Centralized Benchmark', linestyle='--', color='black', linewidth=2)
# Plot the standard deviation for the centralized benchmark
ax.fill_between(range(steps),
                ergodic_metric_mean_base - ergodic_metric_std_base,
                ergodic_metric_mean_base + ergodic_metric_std_base,
                alpha=0.2, color='black')

ax.grid(True, linestyle='--', alpha=0.5)
ax.set_xlabel('Time Step')
ax.set_ylabel('Ergodic Metric Value')
ax.set_title('Ergodic Metric Mean Over Time')
ax.legend()
plt.tight_layout()
plt.show()


# Set the map
map_name = 'simpleMap_05'
# Load the map
map_path = os.path.join(os.getcwd(), 'example_maps/', map_name + '.npy')
map = np.load(map_path)
x_min, x_max = 0, map.shape[0]
y_min, y_max = 0, map.shape[1]
grid_x, grid_y = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max), indexing='ij')
grid = np.vstack([grid_x.flatten(), grid_y.flatten()]).T

means = np.array([[32, 12], [20, 45], [8, 8], [40, 40]])
cov = np.array([[[10, 0], [0, 10]], [[10, 0], [0, 10]], [[10, 0], [0, 10]], [[10, 0], [0, 10]]])
density_map = utilities.gauss_pdf(grid, means[0], cov[0])
density_map = utilities.min_max_normalize(density_map)

norm_density_map = utilities.min_max_normalize(density_map).reshape(map.shape)
norm_density_map = density_map.reshape(map.shape)

free_cells = np.array(np.where(map == 0)).T

goal_density = np.zeros_like(map)
goal_density[free_cells[:, 0], free_cells[:, 1]] = norm_density_map[free_cells[:, 0], free_cells[:, 1]]
# Min-max normalize the goal density
goal_density = np.abs(goal_density)
goal_density_1 = utilities.min_max_normalize(goal_density) # remember to normalize


means = np.array([[10, 15], [20, 45], [8, 8], [40, 40]])
cov = np.array([[[10, 0], [0, 10]], [[10, 0], [0, 10]], [[10, 0], [0, 10]], [[10, 0], [0, 10]]])
density_map = utilities.gauss_pdf(grid, means[0], cov[0])
density_map = utilities.min_max_normalize(density_map)

norm_density_map = utilities.min_max_normalize(density_map).reshape(map.shape)
norm_density_map = density_map.reshape(map.shape)

free_cells = np.array(np.where(map == 0)).T

goal_density = np.zeros_like(map)
goal_density[free_cells[:, 0], free_cells[:, 1]] = norm_density_map[free_cells[:, 0], free_cells[:, 1]]
# Min-max normalize the goal density
goal_density = np.abs(goal_density)
goal_density_2 = utilities.min_max_normalize(goal_density) # remember to normalize


import matplotlib.gridspec as gridspec

# Set fontsize
fontsize = 14  # Reduced for 4 plots
plt.rcParams.update({'font.size': fontsize})
# Plot the trajectories
fig = plt.figure(figsize=(10, 10))  # Larger figure for 4 subplots

# Create GridSpec for 2x2 layout
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], 
                       wspace=0.4, hspace=0.3)

run = 0
chosen_agent_id = 0
process_change_step = 5000  # Define when the process changes

# TOP ROW - BEFORE PROCESS CHANGE
# Top left: Trajectory plot before process change
ax1 = fig.add_subplot(gs[0, 0])
# ax1.set_aspect('equal', adjustable='box')

ax1.contourf(grid_x, grid_y, goal_density_1, cmap='Reds', alpha=1)
ax1.pcolormesh(grid_x, grid_y, np.where(map == 0, np.nan, map), cmap='gray', rasterized=True)

ax1.set_xlabel('X Coordinate')
ax1.set_ylabel('Y Coordinate')
ax1.set_title('Agent Trajectories (Before Change)')

# Plot trajectories up to process change
for agent_id in range(agents_data.shape[1]):
    # Plot trajectory only up to process change
    trajectory_before = agents_data[run, agent_id, :process_change_step, :2]
    
    # Initial position
    initial_pos = agents_data[run, agent_id, 0, :2]
    ax1.plot(initial_pos[0], initial_pos[1], 'o', markersize=6, 
             markerfacecolor='none', markeredgecolor='black', markeredgewidth=2, zorder=10)
    
    # Position at process change
    change_pos = agents_data[run, agent_id, process_change_step-1, :2]
    ax1.plot(change_pos[0], change_pos[1], 's', markersize=6, 
             markerfacecolor='red', markeredgecolor='black', markeredgewidth=1, zorder=10)
    
    if agent_id != chosen_agent_id:
        ax1.plot(trajectory_before[:, 0], trajectory_before[:, 1], 
                color='black', alpha=0.1, linewidth=2, zorder=1)
    else:
        # Plot the chosen agent's trajectory with color
        metric_values_before = ergodic_metrics[run, :process_change_step, agent_id]
        lc1 = plot_colored_trajectory(trajectory_before, metric_values_before)
        ax1.add_collection(lc1)
# Top right: Ergodic metrics - highlighting BEFORE process change
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_xlabel('Time Step')
ax2.set_ylabel('Ergodic Metric Value')
ax2.set_title('Ergodic Metric (Before Change)')

# Plot all agents in gray for the full timeline
for agent_id in range(agents_data.shape[1]):
    if agent_id != chosen_agent_id:
        ax2.plot(range(steps), ergodic_metrics[run, :, agent_id], 
                color='gray', alpha=0.2, linewidth=2)

# For chosen agent: plot full timeline but highlight before process change
agent_metrics_full = ergodic_metrics[run, :, chosen_agent_id]
time_steps_full = np.arange(len(agent_metrics_full))

# Plot the part AFTER process change in muted color
ax2.plot(range(process_change_step, steps), agent_metrics_full[process_change_step:], 
         color='black', alpha=0.7, linewidth=3, zorder=2, linestyle='--')

# Highlight the part BEFORE process change with colored gradient
agent_metrics_before = ergodic_metrics[run, :process_change_step, chosen_agent_id]
time_steps_before = np.arange(len(agent_metrics_before))

points_before = np.array([time_steps_before, agent_metrics_before]).T.reshape(-1, 1, 2)
segments_before = np.concatenate([points_before[:-1], points_before[1:]], axis=1)

metric_avg_before = 0.5 * (agent_metrics_before[:-1] + agent_metrics_before[1:])
norm_before = Normalize(vmin=np.min(metric_avg_before), vmax=np.max(metric_avg_before))
lc2 = LineCollection(segments_before, cmap='viridis_r', norm=norm_before)
lc2.set_array(metric_avg_before)
lc2.set_linewidth(3)
lc2.set_zorder(3)
ax2.add_collection(lc2)

# Add vertical line at process change
ax2.axvline(process_change_step, color='red', linestyle='--', alpha=0.7, zorder=1, linewidth=2)
# ax2.set_xlim(0, steps-1)
# ax2.set_ylim(np.min(agent_metrics_full), np.max(agent_metrics_full))
ax2.grid(True, linestyle='--', alpha=0.5)

# BOTTOM ROW - AFTER PROCESS CHANGE
# Bottom left: Trajectory plot after process change (keep same as before)
ax3 = fig.add_subplot(gs[1, 0])
# ax3.set_aspect('equal', adjustable='box')

ax3.contourf(grid_x, grid_y, goal_density_2, cmap='Reds', alpha=1)
ax3.pcolormesh(grid_x, grid_y, np.where(map == 0, np.nan, map), cmap='gray', rasterized=True)

ax3.set_xlabel('X Coordinate')
ax3.set_ylabel('Y Coordinate')
ax3.set_title('Agent Trajectories (After Change)')

# Plot trajectories from process change onwards
for agent_id in range(agents_data.shape[1]):
    # Plot trajectory only after process change
    trajectory_after = agents_data[run, agent_id, process_change_step:, :2]
    
    # Position at process change (start)
    change_pos = agents_data[run, agent_id, process_change_step, :2]
    ax3.plot(change_pos[0], change_pos[1], 's', markersize=6, 
             markerfacecolor='red', markeredgecolor='black', markeredgewidth=1, zorder=10)
    
    # Final position
    final_pos = agents_data[run, agent_id, -1, :2]
    ax3.plot(final_pos[0], final_pos[1], 'X', markersize=6, 
             markerfacecolor='black', markeredgecolor='black', markeredgewidth=2, zorder=10)
    
    if agent_id != chosen_agent_id:
        ax3.plot(trajectory_after[:, 0], trajectory_after[:, 1], 
                color='black', alpha=0.1, linewidth=2, zorder=1)
    else:
        # Plot the chosen agent's trajectory with color
        metric_values_after = ergodic_metrics[run, process_change_step:, agent_id]
        lc3 = plot_colored_trajectory(trajectory_after, metric_values_after)
        ax3.add_collection(lc3)

# Bottom right: Ergodic metrics - highlighting AFTER process change
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_xlabel('Time Step')
ax4.set_ylabel('Ergodic Metric Value')
ax4.set_title('Ergodic Metric (After Change)')

# Plot all agents in gray for the full timeline
for agent_id in range(agents_data.shape[1]):
    if agent_id != chosen_agent_id:
        ax4.plot(range(steps), ergodic_metrics[run, :, agent_id], 
                color='gray', alpha=0.2, linewidth=2)

# For chosen agent: plot full timeline but highlight after process change
# Plot the part BEFORE process change in muted color
ax4.plot(range(process_change_step), agent_metrics_full[:process_change_step], 
         color='black', alpha=0.7, linewidth=3, zorder=2, linestyle='--')

# Highlight the part AFTER process change with colored gradient
agent_metrics_after = ergodic_metrics[run, process_change_step:, chosen_agent_id]
time_steps_after = np.arange(process_change_step, steps)

points_after = np.array([time_steps_after, agent_metrics_after]).T.reshape(-1, 1, 2)
segments_after = np.concatenate([points_after[:-1], points_after[1:]], axis=1)

metric_avg_after = 0.5 * (agent_metrics_after[:-1] + agent_metrics_after[1:])
norm_after = Normalize(vmin=np.min(metric_avg_after), vmax=np.max(metric_avg_after))
lc4 = LineCollection(segments_after, cmap='viridis_r', norm=norm_after)
lc4.set_array(metric_avg_after)
lc4.set_linewidth(3)
lc4.set_zorder(3)
ax4.add_collection(lc4)

# Add vertical line at process change
ax4.axvline(process_change_step, color='red', linestyle='--', alpha=0.7, zorder=1, linewidth=2)
# ax4.set_xlim(0, steps-1)
# ax4.set_ylim(np.min(agent_metrics_full), np.max(agent_metrics_full))
ax4.grid(True, linestyle='--', alpha=0.5)

# Apply tight layout
plt.tight_layout()

# Save the pdf
plt.savefig("four_panel_process_change.pdf", dpi=300, bbox_inches='tight')
plt.show()