import argparse
import os
from pathlib import Path

import numpy as np
from ergodic_control import utilities

import run_evaluation as reval


def parse_means(means_text):
    """Parse means from 'x1,y1;x2,y2;...'."""
    means = []
    for pair in means_text.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        x_str, y_str = pair.split(",")
        means.append([float(x_str), float(y_str)])
    return np.array(means, dtype=float)


def parse_thresholds(thresholds_text):
    return [float(x.strip()) for x in thresholds_text.split(",") if x.strip()]


def build_goal_density_with_means(grid, map_array, param, means):
    free_cells = np.array(np.where(map_array == 0)).T

    if means.shape[0] < param.nbGaussian:
        raise ValueError(
            f"Need at least {param.nbGaussian} means, got {means.shape[0]}"
        )

    cov = np.repeat(np.array([[[10.0, 0.0], [0.0, 10.0]]]), param.nbGaussian, axis=0)
    density_map = utilities.gmm_eval(
        grid,
        means[: param.nbGaussian],
        cov,
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


def run_experiment(exp_name, map_name, args, param_template, means_override=None):
    output_dir = Path(args.output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    map_path = os.path.join(os.getcwd(), "example_maps", f"{map_name}.npy")
    map_array = np.load(map_path)
    padded_map = np.pad(map_array, 1, "constant", constant_values=1)
    occ_map = utilities.get_occupied_polygon(padded_map) - 1
    closed_map = True

    all_trajectories = []
    all_std_metrics = []
    all_alpha_metrics = []

    original_build_goal_density = reval.build_goal_density
    if means_override is not None:
        reval.build_goal_density = lambda grid, map_arr, param: (
            build_goal_density_with_means(grid, map_arr, param, means_override)
        )

    try:
        for episode_idx in range(args.episodes):
            print(
                f"[{exp_name}] Running episode {episode_idx + 1}/{args.episodes} on map '{map_name}'"
            )
            traj_records, std_records, alpha_records = reval.run_episode(
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
    finally:
        reval.build_goal_density = original_build_goal_density

    reval.write_csv(
        all_trajectories,
        output_dir / "trajectory_log.csv",
        ["episode", "step", "agent_id", "x", "y", "theta"],
    )
    reval.write_csv(
        all_std_metrics,
        output_dir / "std2_log.csv",
        ["episode", "step", "agent_id", "std2_mean"],
    )
    reval.write_csv(
        all_alpha_metrics,
        output_dir / "alpha_log.csv",
        ["episode", "step", "agent_id", "alpha"],
    )
    reval.save_one_minus_std2_timeseries_plot(
        all_std_metrics, output_dir / "std2_timeseries.png"
    )

    print(f"[{exp_name}] Completed. Results saved in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run DARS evaluation twice: first baseline map, second obstacle map with different means"
        )
    )
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_obs_evaluation",
        help="Root output directory",
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
        "--map-name-first",
        type=str,
        default="complexMap_05",
        help="Map for first experiment",
    )
    parser.add_argument(
        "--map-name-second",
        type=str,
        default="generated_map",
        help="Map for second experiment",
    )
    parser.add_argument(
        "--means-second",
        type=str,
        default="35,10;10,35;25,25",
        help="Gaussian means for second experiment as 'x1,y1;x2,y2;...'",
    )
    parser.add_argument(
        "--exploration-thresholds",
        type=str,
        default="0.75, 0.8",
        help="Comma-separated thresholds, one run per threshold per map",
    )
    parser.add_argument(
        "--nb-data-points", type=int, default=None, help="Override nbDataPoints"
    )
    parser.add_argument("--nb-agents", type=int, default=None, help="Override nbAgents")
    parser.add_argument(
        "--no-save-std-heatmaps",
        dest="save_std_heatmaps",
        action="store_false",
        help="Disable saving per-agent GP std heatmaps",
    )
    parser.set_defaults(save_std_heatmaps=True)
    args = parser.parse_args()

    param_template = reval.load_params(args.param_file)
    second_means = parse_means(args.means_second)

    thresholds = parse_thresholds(args.exploration_thresholds)
    map_runs = [
        ("map_1", args.map_name_first, None),
        ("map_2", args.map_name_second, second_means),
    ]

    for threshold in thresholds:
        args.exploration_threshold = threshold
        for map_tag, map_name, means_override in map_runs:
            run_experiment(
                exp_name=f"th_{threshold:.2f}_{map_tag}",
                map_name=map_name,
                args=args,
                param_template=param_template,
                means_override=means_override,
            )

    print(f"Both experiments completed. Root output: {args.output_dir}")


if __name__ == "__main__":
    main()
