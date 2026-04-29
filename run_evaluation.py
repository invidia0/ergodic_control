import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ergodic_control import models, utilities
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

mpl.use("Agg")


TRAJ_LINE_WIDTH = 3.0
TRAJ_LINE_ALPHA = 0.5
START_MARKER_SIZE = 60
END_MARKER_SIZE = 200
CIRCLE_EDGE_WIDTH = 2.0
FADED_TRAJ_COLOR = "tab:blue"
FADED_TRAJ_ALPHA = 0.25
AXIS_LABEL_FONT_SIZE = 32
AXIS_TICKS_FONT_SIZE = 24


def eval_coverage(agent_pos, cell_grid, cell_weights, map_array):
    cov = np.linalg.norm(cell_grid - agent_pos, axis=1) ** 2 * cell_weights
    return cov


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_norm(z, k=10.0, x0=0.5):
    num = sigmoid(k * (z - x0)) - sigmoid(-k / 2.0)
    den = sigmoid(k / 2.0) - sigmoid(-k / 2.0)
    return num / den


def compute_final_kl_records(agents, goal_density, map_array, episode_idx, eps=1e-12):
    free_mask = map_array == 0

    gt = np.clip(goal_density[free_mask].astype(float), 0.0, None)
    gt_sum = np.sum(gt)
    if gt_sum <= 0.0:
        gt = np.full(gt.shape, 1.0 / max(len(gt), 1), dtype=float)
    else:
        gt = gt / gt_sum

    records = []
    for agent in agents:
        gp_map = agent.mu.reshape(map_array.shape)
        gp = np.clip(gp_map[free_mask].astype(float), 0.0, None)
        gp_sum = np.sum(gp)
        if gp_sum <= 0.0:
            gp = np.full(gp.shape, 1.0 / max(len(gp), 1), dtype=float)
        else:
            gp = gp / gp_sum

        kl_gt_to_gp = float(np.sum(gt * np.log((gt + eps) / (gp + eps))))
        records.append(
            {
                "episode": episode_idx,
                "agent_id": agent.id,
                "kl_gt_to_gp": kl_gt_to_gp,
            }
        )

    return records


def load_params(param_path):
    with open(param_path, "r", encoding="utf-8") as f:
        param_data = json.load(f)
    param = lambda: None
    for key, value in param_data.items():
        setattr(param, key, value)
    return param


def initialize_agents(param, free_cells):
    agents = []
    fixed_x0 = np.array(
        [[9, 7], [15, 20], [5, 40], [40, 40], [25, 25]]
    )  # for simpleMap_05
    # fixed_x0 = np.array([[10, 45], [10, 8], [25, 20], [45, 30], [45, 8]])               # for complexMap_05
    # fixed_x0 = np.array([[10, 40], [10, 8], [25, 20], [48, 30], [42, 11], [8, 25]])     # for generated_map
    # fixed_x0 = np.array([]).reshape(0, 2)  # for random initialization
    if param.nbAgents <= len(fixed_x0):
        x0_array = fixed_x0[: param.nbAgents]
    else:
        extra = param.nbAgents - len(fixed_x0)
        sampled_idx = np.random.choice(free_cells.shape[0], extra, replace=False)
        x0_array = np.vstack((fixed_x0, free_cells[sampled_idx]))

    for i in range(param.nbAgents):
        theta0 = np.random.uniform(0.0, 2.0 * np.pi)
        agent = models.DoubleIntegratorAgentNoHeading(
            x=x0_array[i],
            theta=theta0,
            max_dx=param.max_dx,
            max_ddx=param.max_ddx,
            # max_dtheta=param.max_dtheta,
            # max_ddtheta=param.max_ddtheta,
            dt=param.dt_agent,
            id=i,
        )
        agent.sens_range = param.sens_range
        agents.append(agent)
    return agents


def build_goal_density(grid, map_array, param):
    free_cells = np.array(np.where(map_array == 0)).T
    means = np.array([[40, 20], [20, 30], [8, 8]])
    # means = np.array([[25, 20], [10, 35], [8, 8]])
    # means = np.array([[35, 40], [75, 70]])
    # random means within free space
    # means = 10 + 20 * np.random.rand(param.nbGaussian, 2)
    cov = np.array([[[10, 0], [0, 10]], [[10, 0], [0, 10]], [[10, 0], [0, 10]]])

    density_map = utilities.gmm_eval(
        grid,
        means[: param.nbGaussian],
        cov[: param.nbGaussian],
        weights=np.ones(param.nbGaussian) / param.nbGaussian,
    )
    density_map = utilities.min_max_normalize(density_map)
    norm_density_map = density_map.reshape(map_array.shape)

    goal_density = np.zeros_like(map_array, dtype=float)
    goal_density[free_cells[:, 0], free_cells[:, 1]] = norm_density_map[
        free_cells[:, 0], free_cells[:, 1]
    ]
    goal_density = np.abs(goal_density)
    goal_density = utilities.min_max_normalize(goal_density)
    return goal_density


def save_ground_truth_plot(
    goal_density,
    map_array,
    grid_x,
    grid_y,
    output_path,
    x0=None,
):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    contour = ax.contourf(grid_x, grid_y, goal_density, cmap="RdPu", levels=20)
    ax.pcolormesh(
        grid_x, grid_y, np.where(map_array == 0, np.nan, map_array), cmap="gray"
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.08)
    cbar = fig.colorbar(contour, cax=cax, label="Density")
    cbar.ax.tick_params(labelsize=AXIS_TICKS_FONT_SIZE)
    if x0 is not None:
        for i in range(x0.shape[0]):
            ax.scatter(
                x0[i, 0],
                x0[i, 1],
                edgecolor=f"C{i % 10}",
                facecolors="none",
                linewidths=CIRCLE_EDGE_WIDTH,
                s=START_MARKER_SIZE,
                marker="o",
                zorder=4,
            )
    ax.set_title("Ground Truth Density")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_highlighted_trajectories(
    fig, ax, agents, highlight_agent_id, std2_by_agent_step
):
    added_colorbar = False

    for plot_agent in agents:
        x = plot_agent.x_hist[:, 0]
        y = plot_agent.x_hist[:, 1]

        if plot_agent.id == highlight_agent_id:
            std_by_step = std2_by_agent_step.get(plot_agent.id, {})
            sorted_steps = sorted(std_by_step)
            ordered_std = np.array([std_by_step[s] for s in sorted_steps], dtype=float)

            if ordered_std.size > 0:
                if len(x) == ordered_std.size + 1:
                    std2_values = np.concatenate(([ordered_std[0]], ordered_std))
                elif len(x) == ordered_std.size:
                    std2_values = ordered_std
                else:
                    std2_values = np.array(
                        [std_by_step.get(i, np.nan) for i in range(len(x))], dtype=float
                    )
            else:
                std2_values = np.full(len(x), np.nan)

            valid_std = std2_values[~np.isnan(std2_values)]
            if len(x) > 1 and valid_std.size > 0:
                points = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                seg_std2 = 0.5 * (std2_values[:-1] + std2_values[1:])
                valid_seg = ~np.isnan(seg_std2)

                if np.any(valid_seg):
                    vmin = np.nanmin(valid_std)
                    vmax = np.nanmax(valid_std)
                    if np.isclose(vmin, vmax):
                        vmax = vmin + 1e-12

                    line_collection = LineCollection(
                        segments[valid_seg],
                        cmap="plasma",
                        norm=plt.Normalize(vmin=vmin, vmax=vmax),
                        linewidth=TRAJ_LINE_WIDTH,
                        alpha=1.0,
                        zorder=3,
                    )
                    line_collection.set_array(seg_std2[valid_seg])
                    ax.add_collection(line_collection)
                    if not added_colorbar:
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("right", size="4%", pad=0.08)
                        cbar = fig.colorbar(line_collection, cax=cax)
                        # cbar.set_label(
                        # r"$\overline{\sigma}^2$", fontsize=AXIS_LABEL_FONT_SIZE
                        # )
                        cbar.ax.tick_params(labelsize=AXIS_TICKS_FONT_SIZE)
                        added_colorbar = True
                else:
                    ax.plot(
                        x,
                        y,
                        color=f"C{plot_agent.id % 10}",
                        lw=TRAJ_LINE_WIDTH,
                        alpha=1.0,
                        zorder=3,
                    )
            else:
                ax.plot(
                    x,
                    y,
                    color=f"C{plot_agent.id % 10}",
                    lw=TRAJ_LINE_WIDTH,
                    alpha=1.0,
                    zorder=3,
                )

            ax.scatter(
                x[0],
                y[0],
                edgecolor=f"C{plot_agent.id % 10}",
                facecolors="none",
                linewidths=CIRCLE_EDGE_WIDTH,
                s=START_MARKER_SIZE,
                marker="o",
                zorder=4,
            )
            ax.scatter(
                x[-1],
                y[-1],
                color=f"C{plot_agent.id % 10}",
                s=END_MARKER_SIZE,
                marker="o",
                zorder=4,
            )
        else:
            ax.plot(
                x,
                y,
                color=FADED_TRAJ_COLOR,
                lw=TRAJ_LINE_WIDTH,
                alpha=FADED_TRAJ_ALPHA,
                zorder=2,
            )
            ax.scatter(
                x[0],
                y[0],
                facecolor="none",
                edgecolors=FADED_TRAJ_COLOR,
                linewidths=CIRCLE_EDGE_WIDTH,
                s=START_MARKER_SIZE,
                marker="o",
                alpha=FADED_TRAJ_ALPHA,
                zorder=2,
            )
            ax.scatter(
                x[-1],
                y[-1],
                color=FADED_TRAJ_COLOR,
                s=END_MARKER_SIZE,
                marker="o",
                alpha=FADED_TRAJ_ALPHA,
                zorder=2,
            )


def save_gp_plot(
    agent, agents, std2_by_agent_step, map_array, grid_x, grid_y, output_path
):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    gp_map = agent.mu.reshape(map_array.shape)
    contour = ax.contourf(grid_x, grid_y, gp_map, cmap="Greys", levels=20)
    _plot_highlighted_trajectories(
        fig=fig,
        ax=ax,
        agents=agents,
        highlight_agent_id=agent.id,
        std2_by_agent_step=std2_by_agent_step,
    )
    ax.tick_params(axis="both", labelsize=AXIS_TICKS_FONT_SIZE)
    # ax.pcolormesh(
    #     grid_x, grid_y, np.where(map_array == 0, np.nan, map_array), cmap="gray"
    # )
    # fig.colorbar(contour, ax=ax, label="GP Mean")
    # ax.set_title(f"Reconstructed GP - Agent {agent.id}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_std_heatmap_plot(
    agent, agents, std2_by_agent_step, map_array, grid_x, grid_y, output_path
):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    std_map = agent.std.reshape(map_array.shape)
    contour = ax.contourf(
        grid_x,
        grid_y,
        std_map,
        cmap="Greys",
        levels=20,
        vmin=0.0,
        vmax=1.0,
    )
    _plot_highlighted_trajectories(
        fig=fig,
        ax=ax,
        agents=agents,
        highlight_agent_id=agent.id,
        std2_by_agent_step=std2_by_agent_step,
    )
    ax.tick_params(axis="both", labelsize=AXIS_TICKS_FONT_SIZE)
    # ax.pcolormesh(
    #     grid_x, grid_y, np.where(map_array == 0, np.nan, map_array), cmap="gray"
    # )
    # fig.colorbar(contour, ax=ax, label="GP Std")
    # ax.set_title(f"GP Std Heatmap - Agent {agent.id}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_trajectories_plot(
    agents, goal_density, map_array, grid_x, grid_y, output_path
):
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    contour = ax.contourf(grid_x, grid_y, goal_density, cmap="Greys", levels=20)
    ax.pcolormesh(
        grid_x, grid_y, np.where(map_array == 0, np.nan, map_array), cmap="gray"
    )
    # fig.colorbar(contour, ax=ax, label="Density")

    for agent in agents:
        ax.plot(
            agent.x_hist[:, 0],
            agent.x_hist[:, 1],
            color=f"C{agent.id % 10}",
            lw=TRAJ_LINE_WIDTH,
            alpha=TRAJ_LINE_ALPHA,
        )
        ax.scatter(
            agent.x_hist[0, 0],
            agent.x_hist[0, 1],
            edgecolor=f"C{agent.id % 10}",
            facecolors="none",
            linewidths=CIRCLE_EDGE_WIDTH,
            s=START_MARKER_SIZE,
            marker="o",
        )
        ax.scatter(
            agent.x_hist[-1, 0],
            agent.x_hist[-1, 1],
            color=f"C{agent.id % 10}",
            s=END_MARKER_SIZE,
            marker="o",
        )

    # ax.set_title("Robot Trajectories over Ground Truth Density")
    ax.tick_params(axis="both", labelsize=AXIS_TICKS_FONT_SIZE)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run_episode(
    episode_idx,
    args,
    base_output_dir,
    param_template,
    map_array,
    occ_map,
    closed_map,
    collect_final_kl=False,
):
    param = lambda: None
    for key, value in vars(param_template).items():
        setattr(param, key, value)

    if args.nb_data_points is not None:
        param.nbDataPoints = args.nb_data_points
    if args.nb_agents is not None:
        param.nbAgents = args.nb_agents

    np.random.seed(param.random_seed + episode_idx)

    x_min, x_max = 0, map_array.shape[0]
    y_min, y_max = 0, map_array.shape[1]
    grid_x, grid_y = np.meshgrid(
        np.arange(x_min, x_max), np.arange(y_min, y_max), indexing="ij"
    )
    grid = np.vstack([grid_x.flatten(), grid_y.flatten()]).T

    free_cells = np.array(np.where(map_array == 0)).T
    free_mask = map_array == 0

    param.nbResX = map_array.shape[0]
    param.nbResY = map_array.shape[1]
    param.dt = min(1.0, (param.dx**2) / (4.0 * np.max(param.alpha)))
    param.width = map_array.shape[0]
    param.height = map_array.shape[1]

    cell_area = param.dx * param.dx
    param.area = np.sum(map_array == 0) * cell_area
    param.beta = param.beta / param.area
    param.local_cooling = param.local_cooling / param.area

    agents = initialize_agents(param, free_cells)
    goal_density = build_goal_density(grid, map_array, param)

    coverage_block = utilities.agent_block(
        param.nbVar, param.min_kernel_val, param.agent_radius
    )
    param.kernel_size = coverage_block.shape[0]

    kernel = C(1.0, constant_value_bounds=(1e-3, 1e3)) * RBF(
        length_scale=1.0, length_scale_bounds=(1e-3, 1e3)
    ) + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-8, 1e-2))
    gpr = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=1, alpha=1e-2, normalize_y=True
    )
    gpr.kernel_ = kernel

    buffer_size = 500
    for agent in agents:
        agent.heat = np.empty_like(goal_density)
        agent.samples = np.empty((0, 4))
        agent.dataset = np.empty((0, 4))
        agent.coverage_buffer = np.zeros((buffer_size, *goal_density.shape))
        agent.buffer_index = 0
        agent.buffer_full = False
        agent.neighbors = []
        agent.last_neighbors = []
        agent.local_cooling = np.zeros_like(goal_density)
        agent.coverage_density = np.zeros_like(goal_density)
        agent.cov_dens = np.zeros_like(goal_density)
        agent.in_ergodic = True

    adjacency_matrix = np.eye(param.nbAgents)
    min_safe_range = 1

    fov_template = utilities.init_fov(param.fov_deg, param.fov_depth)
    trajectory_records = []
    std_metric_records = []
    alpha_records = []

    spatial_decay = 100_000
    temporal_decay = 100_000
    take_th = 0.80
    remove_th = 0.75
    exploration_threshold = args.exploration_threshold

    for step in range(param.nbDataPoints):
        if step % param.plot_frequency == 0:
            print(f"Episode {episode_idx + 1}, Step {step}/{param.nbDataPoints}")
        if param.nbAgents > 1 and step > 0:
            adjacency_matrix = utilities.share_samples(
                agents, map_array, param.sens_range, adjacency_matrix
            )

        states = np.array([agent.x for agent in agents])
        voronoi_masks = utilities.compute_voronoi_partitioning(
            grid, states[:, :2], param.sens_range
        )

        for agent in agents:
            if map_array[int(agent.x[0]), int(agent.x[1])] == 1:
                raise ValueError(f"Agent {agent.id} collided with an obstacle")

            agent.local_cooling = np.zeros_like(goal_density)
            fov_edges_moved = utilities.rotate_and_translate(
                fov_template, agent.x, agent.theta
            )
            fov_edges_clipped = utilities.clip_polygon_no_convex(
                agent.x, fov_edges_moved, occ_map, closed_map
            )
            fov_points = utilities.insidepolygon(fov_edges_clipped).astype(int)

            fov_probs = utilities.fov_coverage_block(
                fov_points, fov_edges_clipped, param.fov_depth
            )

            current_coverage = np.zeros_like(goal_density)
            current_coverage[fov_points[:, 0], fov_points[:, 1]] = fov_probs

            agent.coverage_buffer[agent.buffer_index] = current_coverage
            agent.buffer_index = (agent.buffer_index + 1) % buffer_size
            if agent.buffer_index == 0:
                agent.buffer_full = True

            if agent.buffer_full:
                coverage_density = np.sum(agent.coverage_buffer, axis=0)
            else:
                coverage_density = np.sum(
                    agent.coverage_buffer[: agent.buffer_index], axis=0
                )

            if len(agent.neighbors) > 0:
                for neighbor_id in agent.neighbors:
                    neighbor_agent = agents[neighbor_id]
                    x_idx = int(neighbor_agent.x[0] // param.dx)
                    y_idx = int(neighbor_agent.x[1] // param.dx)

                    x_start = max(0, x_idx - param.sens_range // 2)
                    x_end = min(map_array.shape[0], x_idx + param.sens_range // 2 + 1)
                    y_start = max(0, y_idx - param.sens_range // 2)
                    y_end = min(map_array.shape[1], y_idx + param.sens_range // 2 + 1)

                    coverage_density[x_start:x_end, y_start:y_end] += (
                        neighbor_agent.coverage_density[x_start:x_end, y_start:y_end]
                    )

            agent.fov_edges = fov_edges_moved
            y_samples = goal_density[fov_points[:, 0], fov_points[:, 1]]
            agent.samples = np.hstack(
                (
                    fov_points,
                    step * np.ones((fov_points.shape[0], 1), dtype=int).reshape(-1, 1),
                    y_samples.reshape(-1, 1),
                )
            )

            test_set = np.empty((0, 4))
            pooled_samples = utilities.max_pooling(agent.samples, 5)
            test_set = np.unique(np.vstack((test_set, pooled_samples)), axis=0)

            if agent.neighbors:
                all_samples = np.vstack(
                    [agents[n_id].dataset for n_id in agent.neighbors]
                )
                test_set = np.unique(np.vstack([test_set, all_samples]), axis=0)

            new_samples = test_set
            if step > 0 and hasattr(agent, "std"):
                std_test = agent.std[
                    test_set[:, 0].astype(int),
                    test_set[:, 1].astype(int),
                ]
                new_samples = test_set[std_test > take_th]

            agent.dataset = np.vstack((agent.dataset, new_samples))

            if len(new_samples) != 0:
                gpr.fit(agent.dataset[:, :2], agent.dataset[:, 3])

            agent.dataset = agent.dataset[np.argsort(agent.dataset[:, 2])]

            D, d = utilities.compute_spatio_decay_matrix(
                agent.dataset[:, :2], spatial_decay, agent.x
            )
            T, t = utilities.compute_temporal_decay_matrix(
                agent.dataset[:, 2], step, temporal_decay
            )

            agent.combo_density, agent.mu, agent.std = utilities.compute_combo(
                agent.dataset[:, :2],
                agent.dataset[:, 3],
                grid,
                map_array,
                gpr.kernel_,
                D,
                T,
                d,
                t,
            )
            agent.std = np.where(map_array == 0, agent.std, 0)
            agent.coverage_density = coverage_density

            # std2_mean = np.mean(agent.std**2)
            std_metric_records.append(
                {
                    "episode": episode_idx,
                    "step": step,
                    "agent_id": agent.id,
                    "std2_mean": np.mean(agent.std**2),
                }
            )

            if step > 0:
                std_test = agent.std[
                    agent.dataset[:, 0].astype(int),
                    agent.dataset[:, 1].astype(int),
                ]
                agent.dataset = agent.dataset[std_test < remove_th]
                agent.dataset = agent.dataset[np.argsort(agent.dataset[:, 2])]

            if step == 0:
                agent.heat = np.array(utilities.normalize_mat(agent.combo_density))

            diff = utilities.normalize_mat(
                agent.combo_density
            ) - utilities.normalize_mat(coverage_density)

            agent.weights = agent.mu.flatten() * voronoi_masks[agent.id]
            if np.sum(agent.weights) > 0:
                agent.centroid = np.sum(
                    grid * agent.weights[:, np.newaxis], axis=0
                ) / np.sum(agent.weights)
            else:
                agent.centroid = agent.x

            erg_source = np.maximum(diff, 0) ** 2
            erg_source = np.where(map_array == 0, erg_source, 0)
            erg_source = utilities.normalize_mat(erg_source)

            sigma_c = 5.0
            coverage_source = np.exp(
                -(np.linalg.norm(grid - agent.centroid, axis=1) ** 2) / (2 * sigma_c**2)
            )
            agent.coverage_source = utilities.normalize_mat(
                coverage_source.reshape(map_array.shape)
            )

            alpha = sigmoid_norm(
                1.0 - np.mean(agent.std**2),
                k=args.exploration_sigmoid_k,
                x0=exploration_threshold,
            )
            alpha_records.append(
                {
                    "episode": episode_idx,
                    "step": step,
                    "agent_id": agent.id,
                    "alpha": float(alpha),
                }
            )
            source = (1.0 - alpha) * erg_source + alpha * agent.coverage_source
            source = np.where(map_array == 0, source, 0)

            agent.source = utilities.normalize_mat(source) * param.area
            agent.heat = utilities.update_heat_optimized(
                agent.heat,
                agent.source,
                map_array,
                agent.local_cooling,
                param.dt,
                param.alpha,
                param.source_strength,
                param.beta,
                param.local_cooling,
                param.dx,
            ).astype(np.float32)

            gradient_y, gradient_x = np.gradient(agent.heat.T, 1, 1)
            grad_mag = np.sqrt(np.mean(gradient_x**2) + np.mean(gradient_y**2))
            gradient_x /= grad_mag + 1e-8
            gradient_y /= grad_mag + 1e-8

            agent.grad = utilities.calculate_gradient_map_v2(
                param, agent, gradient_x, gradient_y, map_array
            )

            if len(agent.neighbors) > 0:
                for neighbor in agent.neighbors:
                    neighbor_agent = agents[neighbor]
                    q_ij = np.linalg.norm(agent.x - neighbor_agent.x) ** 2
                    p = (
                        4
                        * (param.sens_range**2 - min_safe_range**2)
                        * (q_ij - param.sens_range**2)
                        * (agent.x - neighbor_agent.x)
                        / (q_ij - min_safe_range**2) ** 3
                    )
                    agent.grad -= p

            k_target = 2
            v_target = k_target * agent.grad
            theta_target = np.atan2(agent.grad[1], agent.grad[0])
            # agent.track_velocity_and_heading(
            #     v_target, theta_target, penalize_lateral=True
            # )
            agent.update(v_target)

            # eval coverage
            coverage = 0.0
            for agent in agents:
                weights = goal_density.flatten() * voronoi_masks[agent.id]
                coverage += eval_coverage(agent.x, grid, weights.flatten(), map_array)

            trajectory_records.append(
                {
                    "episode": episode_idx,
                    "step": step,
                    "agent_id": agent.id,
                    "x": float(agent.x[0]),
                    "y": float(agent.x[1]),
                    "theta": float(agent.theta),
                }
            )

    print(f"Episode {episode_idx + 1} completed. Saving results...")
    episode_dir = base_output_dir / f"episode_{episode_idx:03d}"
    episode_dir.mkdir(parents=True, exist_ok=True)

    save_ground_truth_plot(
        goal_density=goal_density,
        map_array=map_array,
        grid_x=grid_x,
        grid_y=grid_y,
        output_path=episode_dir / "ground_truth_density.png",
    )
    save_trajectories_plot(
        agents=agents,
        goal_density=goal_density,
        map_array=map_array,
        grid_x=grid_x,
        grid_y=grid_y,
        output_path=episode_dir / "trajectories_over_ground_truth.png",
    )

    std2_by_agent_step = {}
    for record in std_metric_records:
        agent_id = int(record["agent_id"])
        step = int(record["step"])
        std2_by_agent_step.setdefault(agent_id, {})[step] = float(record["std2_mean"])

    for agent in agents:
        save_gp_plot(
            agent=agent,
            agents=agents,
            std2_by_agent_step=std2_by_agent_step,
            map_array=map_array,
            grid_x=grid_x,
            grid_y=grid_y,
            output_path=episode_dir / f"reconstructed_gp_agent_{agent.id}.png",
        )
        if args.save_std_heatmaps:
            save_std_heatmap_plot(
                agent=agent,
                agents=agents,
                std2_by_agent_step=std2_by_agent_step,
                map_array=map_array,
                grid_x=grid_x,
                grid_y=grid_y,
                output_path=episode_dir / f"std_heatmap_agent_{agent.id}.png",
            )

    save_one_minus_std2_episode_plot(
        std_metric_records, episode_dir / "std2_timeseries.png"
    )
    save_alpha_episode_plot(alpha_records, episode_dir / "alpha_timeseries.png")

    if collect_final_kl:
        final_kl_records = compute_final_kl_records(
            agents=agents,
            goal_density=goal_density,
            map_array=map_array,
            episode_idx=episode_idx,
        )
        return trajectory_records, std_metric_records, alpha_records, final_kl_records

    return trajectory_records, std_metric_records, alpha_records


def write_csv(records, csv_path, fieldnames):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def save_one_minus_std2_timeseries_plot(std_metric_records, output_path):
    if not std_metric_records:
        return

    episodes = np.array([int(r["episode"]) for r in std_metric_records], dtype=int)
    steps = np.array([int(r["step"]) for r in std_metric_records], dtype=int)
    agent_ids = np.array([int(r["agent_id"]) for r in std_metric_records], dtype=int)
    values = np.array([float(r["std2_mean"]) for r in std_metric_records], dtype=float)

    unique_episodes = np.unique(episodes)
    unique_agents = np.unique(agent_ids)

    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111)

    for agent_id in unique_agents:
        color = f"C{agent_id % 10}"

        for ep in unique_episodes:
            mask = (agent_ids == agent_id) & (episodes == ep)
            if not np.any(mask):
                continue
            order = np.argsort(steps[mask])
            ax.plot(
                steps[mask][order],
                values[mask][order],
                color=color,
                alpha=0.15,
                lw=1,
            )

        all_steps = np.unique(steps[agent_ids == agent_id])
        mean_vals = []
        for step in all_steps:
            mask = (agent_ids == agent_id) & (steps == step)
            mean_vals.append(np.mean(values[mask]))

        ax.plot(
            all_steps,
            mean_vals,
            color=color,
            lw=2,
            label=f"Agent {agent_id} (mean)",
        )

    ax.set_xlabel("Time Step")
    ax.set_ylabel("std^2")
    ax.set_title("std^2 Time-Series per Robot")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_one_minus_std2_episode_plot(std_metric_records, output_path):
    if not std_metric_records:
        return

    steps = np.array([int(r["step"]) for r in std_metric_records], dtype=int)
    agent_ids = np.array([int(r["agent_id"]) for r in std_metric_records], dtype=int)
    values = np.array([float(r["std2_mean"]) for r in std_metric_records], dtype=float)

    unique_agents = np.unique(agent_ids)
    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111)

    for agent_id in unique_agents:
        mask = agent_ids == agent_id
        order = np.argsort(steps[mask])
        ax.plot(
            steps[mask][order],
            values[mask][order],
            color=f"C{agent_id % 10}",
            lw=2,
            label=f"Agent {agent_id}",
        )

    episode_id = int(std_metric_records[0]["episode"])
    ax.set_xlabel("Time Step")
    ax.set_ylabel("std^2")
    ax.set_title(f"std^2 Time-Series per Robot (Episode {episode_id})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_alpha_episode_plot(alpha_records, output_path):
    if not alpha_records:
        return

    steps = np.array([int(r["step"]) for r in alpha_records], dtype=int)
    agent_ids = np.array([int(r["agent_id"]) for r in alpha_records], dtype=int)
    values = np.array([float(r["alpha"]) for r in alpha_records], dtype=float)

    unique_agents = np.unique(agent_ids)
    episode_id = int(alpha_records[0]["episode"])

    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111)

    for agent_id in unique_agents:
        mask = agent_ids == agent_id
        order = np.argsort(steps[mask])
        ax.plot(
            steps[mask][order],
            values[mask][order],
            color=f"C{agent_id % 10}",
            lw=2,
            label=f"Agent {agent_id}",
        )

    ax.set_xlabel("Time Step")
    ax.set_ylabel("alpha")
    ax.set_title(f"Alpha Time-Series per Robot (Episode {episode_id})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-episode evaluation from dars2 logic"
    )
    parser.add_argument(
        "--episodes", type=int, default=1, help="Number of episodes to run"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output3",
        help="Directory where plots and logs are saved",
    )
    parser.add_argument(
        "--param-file",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "params", "dars.json"
        ),
        help="Path to params JSON file",
    )
    parser.add_argument(
        "--map-name",
        type=str,
        default="simpleMap2",
        help="Map name in example_maps/<map_name>.npy",
    )
    parser.add_argument(
        "--nb-data-points", type=int, default=None, help="Override nbDataPoints"
    )
    parser.add_argument("--nb-agents", type=int, default=None, help="Override nbAgents")
    parser.add_argument(
        "--exploration-threshold",
        type=float,
        default=0.7,
        help="Threshold used in alpha sigmoid computation",
    )
    parser.add_argument(
        "--exploration-sigmoid-k",
        type=float,
        default=500.0,
        help="Slope parameter k used in alpha sigmoid computation",
    )
    parser.add_argument(
        "--no-save-std-heatmaps",
        dest="save_std_heatmaps",
        action="store_false",
        help="Disable saving per-agent GP std heatmaps",
    )
    parser.set_defaults(save_std_heatmaps=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    map_path = os.path.join(os.getcwd(), "example_maps", f"{args.map_name}.npy")
    map_array = np.load(map_path)
    padded_map = np.pad(map_array, 1, "constant", constant_values=1)
    occ_map = utilities.get_occupied_polygon(padded_map) - 1
    closed_map = True

    param_template = load_params(args.param_file)

    all_trajectories = []
    all_std_metrics = []
    all_alpha_metrics = []

    for episode_idx in range(args.episodes):
        print(f"Running episode {episode_idx + 1}/{args.episodes}")
        traj_records, std_records, alpha_records = run_episode(
            episode_idx=episode_idx,
            args=args,
            base_output_dir=output_dir,
            param_template=param_template,
            map_array=map_array,
            occ_map=occ_map,
            closed_map=closed_map,
        )
        all_trajectories.extend(traj_records)
        all_std_metrics.extend(std_records)
        all_alpha_metrics.extend(alpha_records)

    write_csv(
        all_trajectories,
        output_dir / "trajectory_log.csv",
        ["episode", "step", "agent_id", "x", "y", "theta"],
    )
    write_csv(
        all_std_metrics,
        output_dir / "std2_log.csv",
        ["episode", "step", "agent_id", "std2_mean"],
    )
    write_csv(
        all_alpha_metrics,
        output_dir / "alpha_log.csv",
        ["episode", "step", "agent_id", "alpha"],
    )
    save_one_minus_std2_timeseries_plot(
        all_std_metrics, output_dir / "std2_timeseries.png"
    )

    print(f"Evaluation completed. Results saved in: {output_dir}")


if __name__ == "__main__":
    main()
