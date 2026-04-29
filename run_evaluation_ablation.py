import argparse
import os
from pathlib import Path

import numpy as np

import run_evaluation as eval_run
from ergodic_control import utilities


def _format_value_for_dir(value):
    formatted = f"{value:g}"
    return formatted.replace("-", "m").replace(".", "p")


def _build_combo_dir_name(exploration_threshold, exploration_sigmoid_k):
    th_str = _format_value_for_dir(exploration_threshold)
    k_str = _format_value_for_dir(exploration_sigmoid_k)
    return f"th_{th_str}__k_{k_str}"


def _compute_average_traveled_distance(trajectory_records):
    if not trajectory_records:
        return float("nan")

    grouped = {}
    for row in trajectory_records:
        key = (int(row["episode"]), int(row["agent_id"]))
        grouped.setdefault(key, []).append(
            (int(row["step"]), float(row["x"]), float(row["y"]))
        )

    distances = []
    for _, samples in grouped.items():
        ordered = sorted(samples, key=lambda t: t[0])
        if len(ordered) < 2:
            distances.append(0.0)
            continue

        pts = np.array([(x, y) for _, x, y in ordered], dtype=float)
        seg_lengths = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
        distances.append(float(np.sum(seg_lengths)))

    if not distances:
        return float("nan")

    return float(np.mean(distances))


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-episode ablation over exploration threshold and sigmoid k"
    )
    parser.add_argument(
        "--episodes", type=int, default=1, help="Number of episodes to run"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_ablation",
        help="Directory where ablation outputs are saved",
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
        "--exploration-thresholds",
        type=float,
        nargs="+",
        default=[0.4, 0.6, 0.7, 0.8, 0.95],
        help="One or more threshold values for alpha sigmoid (e.g. 0.6 0.7 0.8)",
    )
    parser.add_argument(
        "--exploration-sigmoid-ks",
        type=float,
        nargs="+",
        default=[10.0, 50.0, 100.0],
        help="One or more k values for alpha sigmoid (e.g. 300 500 700)",
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

    param_template = eval_run.load_params(args.param_file)

    total_combos = len(args.exploration_thresholds) * len(args.exploration_sigmoid_ks)
    combo_idx = 0
    combo_summaries = []

    for exploration_threshold in args.exploration_thresholds:
        for exploration_sigmoid_k in args.exploration_sigmoid_ks:
            combo_idx += 1
            combo_name = _build_combo_dir_name(
                exploration_threshold=exploration_threshold,
                exploration_sigmoid_k=exploration_sigmoid_k,
            )
            combo_output_dir = output_dir / combo_name
            combo_output_dir.mkdir(parents=True, exist_ok=True)

            print(
                f"Running combo {combo_idx}/{total_combos}: "
                f"threshold={exploration_threshold}, k={exploration_sigmoid_k}"
            )

            run_args = argparse.Namespace(**vars(args))
            run_args.exploration_threshold = exploration_threshold
            run_args.exploration_sigmoid_k = exploration_sigmoid_k

            all_trajectories = []
            all_std_metrics = []
            all_alpha_metrics = []
            all_final_kl_metrics = []

            for episode_idx in range(args.episodes):
                print(f"Running episode {episode_idx + 1}/{args.episodes}")
                traj_records, std_records, alpha_records, kl_records = (
                    eval_run.run_episode(
                        episode_idx=episode_idx,
                        args=run_args,
                        base_output_dir=combo_output_dir,
                        param_template=param_template,
                        map_array=map_array,
                        occ_map=occ_map,
                        closed_map=closed_map,
                        collect_final_kl=True,
                    )
                )
                all_trajectories.extend(traj_records)
                all_std_metrics.extend(std_records)
                all_alpha_metrics.extend(alpha_records)
                all_final_kl_metrics.extend(kl_records)

            eval_run.write_csv(
                all_trajectories,
                combo_output_dir / "trajectory_log.csv",
                ["episode", "step", "agent_id", "x", "y", "theta"],
            )
            eval_run.write_csv(
                all_std_metrics,
                combo_output_dir / "std2_log.csv",
                ["episode", "step", "agent_id", "std2_mean"],
            )
            eval_run.write_csv(
                all_alpha_metrics,
                combo_output_dir / "alpha_log.csv",
                ["episode", "step", "agent_id", "alpha"],
            )
            eval_run.write_csv(
                all_final_kl_metrics,
                combo_output_dir / "final_kl_log.csv",
                ["episode", "agent_id", "kl_gt_to_gp"],
            )
            eval_run.save_one_minus_std2_timeseries_plot(
                all_std_metrics, combo_output_dir / "std2_timeseries.png"
            )

            episode_ids = sorted({int(r["episode"]) for r in all_std_metrics})
            max_step = max((int(r["step"]) for r in all_std_metrics), default=0)

            final_std2_vals = []
            for ep in episode_ids:
                ep_steps = [
                    int(r["step"]) for r in all_std_metrics if int(r["episode"]) == ep
                ]
                if not ep_steps:
                    continue
                ep_last_step = max(ep_steps)
                ep_final_vals = [
                    float(r["std2_mean"])
                    for r in all_std_metrics
                    if int(r["episode"]) == ep and int(r["step"]) == ep_last_step
                ]
                if ep_final_vals:
                    final_std2_vals.append(float(np.mean(ep_final_vals)))

            final_alpha_vals = []
            for ep in episode_ids:
                ep_steps = [
                    int(r["step"]) for r in all_alpha_metrics if int(r["episode"]) == ep
                ]
                if not ep_steps:
                    continue
                ep_last_step = max(ep_steps)
                ep_final_vals = [
                    float(r["alpha"])
                    for r in all_alpha_metrics
                    if int(r["episode"]) == ep and int(r["step"]) == ep_last_step
                ]
                if ep_final_vals:
                    final_alpha_vals.append(float(np.mean(ep_final_vals)))

            final_kl_vals = [float(r["kl_gt_to_gp"]) for r in all_final_kl_metrics]
            avg_traveled_distance = _compute_average_traveled_distance(all_trajectories)

            combo_summary = {
                "combo": combo_name,
                "exploration_threshold": float(exploration_threshold),
                "exploration_sigmoid_k": float(exploration_sigmoid_k),
                "episodes": int(args.episodes),
                "max_step": int(max_step),
                "final_mean_std2": float(np.mean(final_std2_vals))
                if final_std2_vals
                else float("nan"),
                "final_mean_alpha": float(np.mean(final_alpha_vals))
                if final_alpha_vals
                else float("nan"),
                "final_mean_kl_gt_to_gp": float(np.mean(final_kl_vals))
                if final_kl_vals
                else float("nan"),
                "avg_traveled_distance": avg_traveled_distance,
            }
            combo_summaries.append(combo_summary)

            print(f"Completed combo: {combo_name}")

    eval_run.write_csv(
        combo_summaries,
        output_dir / "ablation_summary.csv",
        [
            "combo",
            "exploration_threshold",
            "exploration_sigmoid_k",
            "episodes",
            "max_step",
            "final_mean_std2",
            "final_mean_alpha",
            "final_mean_kl_gt_to_gp",
            "avg_traveled_distance",
        ],
    )

    print(f"Ablation completed. Results saved in: {output_dir}")


if __name__ == "__main__":
    main()
