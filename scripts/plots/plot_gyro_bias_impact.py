"""
Plot gyro bias correction impact on orientation drift.

Compares raw vs bias-corrected gyro-only propagation against EuRoC GT.

Error convention — identical to scripts/rerun/rerun_euroc_gyro_propagation.py
(shared helpers in slam_core_tools.viz):
  R_est_ref = estimated rotation at t0 (first raw CSV row)
  R_gt_ref  = nearest GT rotation at t0

  R_est_rel(t) = R_est_ref.T @ R_est(t)
  R_gt_rel(t)  = R_gt_ref.T  @ R_gt(t)
  R_err        = R_gt_rel.T  @ R_est_rel
  err_deg      = geodesic_deg(R_err)

Both series share the same reference pair (first timestamp of the raw CSV).

Usage:
    uv run python scripts/plots/plot_gyro_bias_impact.py \\
      results/imu/MH_01_easy_gyro_orientations.csv \\
      results/imu/MH_01_easy_gyro_orientations_bias_corrected.csv \\
      --config configs/datasets/euroc_mh01.yaml \\
      --dataset-root "$HOME/datasets" \\
      --output results/plots/MH_01_easy_gyro_bias_impact.png \\
      --max-duration-s 180
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from slam_core_tools.viz import (
    build_gt_index,
    geodesic_deg,
    load_orientation_csv,
    load_sequence_gt,
    nearest_by_index,
)


# ---------------------------------------------------------------------------
# Error series computation
# ---------------------------------------------------------------------------

def compute_error_series(
    rows: list[tuple[float, np.ndarray]],
    start_s: float,
    max_duration_s: float | None,
    gt_times: list[float],
    gt_mats: list[np.ndarray],
    R_est_ref: np.ndarray,
    R_gt_ref: np.ndarray,
) -> tuple[list[float], list[float]]:
    t_rel_list: list[float] = []
    err_list: list[float] = []
    for ts, mat in rows:
        t_rel = ts - start_s
        if max_duration_s is not None and t_rel > max_duration_s:
            break
        R_est     = mat
        R_gt      = nearest_by_index(ts, gt_mats, gt_times)
        R_est_rel = R_est_ref.T @ R_est
        R_gt_rel  = R_gt_ref.T  @ R_gt
        R_err     = R_gt_rel.T  @ R_est_rel
        t_rel_list.append(t_rel)
        err_list.append(geodesic_deg(R_err))
    return t_rel_list, err_list


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_plot(
    t_raw: list[float],
    err_raw: list[float],
    t_corr: list[float],
    err_corr: list[float],
    output_path: Path,
) -> None:
    mean_raw   = sum(err_raw)  / len(err_raw)
    mean_corr  = sum(err_corr) / len(err_corr)
    final_raw  = err_raw[-1]
    final_corr = err_corr[-1]
    reduction  = mean_raw / mean_corr if mean_corr > 0 else float("inf")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(t_raw,  err_raw,  color="#d62728", linewidth=1.2,
            label="Raw gyro (no bias correction)")
    ax.plot(t_corr, err_corr, color="#1f77b4", linewidth=1.2,
            label="Bias-corrected (offline, user-supplied constant bias)")
    ax.axhline(0.0, color="#aaaaaa", linewidth=0.7, linestyle="--")

    annotation = (
        f"mean   {mean_raw:.1f}° → {mean_corr:.1f}°\n"
        f"final  {final_raw:.1f}° → {final_corr:.1f}°\n"
        f"~{reduction:.0f}× mean-error reduction"
    )
    ax.annotate(
        annotation,
        xy=(0.97, 0.95),
        xycoords="axes fraction",
        ha="right", va="top",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor="white",
            edgecolor="#cccccc",
            alpha=0.92,
        ),
    )

    ax.set_title(
        "Gyro Bias Correction Reduces Orientation Drift",
        fontsize=13, fontweight="bold", pad=10,
    )
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Orientation Error (deg)", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, linewidth=0.4, alpha=0.5)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot gyro bias correction impact on orientation drift."
    )
    parser.add_argument("raw_csv",       help="Raw orientation CSV from export_gyro_propagation")
    parser.add_argument("corrected_csv", help="Bias-corrected orientation CSV")
    parser.add_argument("--config",       required=True, metavar="PATH",
                        help="EuRoC sequence config YAML")
    parser.add_argument("--dataset-root", metavar="PATH",
                        help="Parent directory containing euroc/<sequence>/")
    parser.add_argument("--output",       metavar="PATH",
                        default="results/plots/gyro_bias_impact.png",
                        help="Output PNG path (default: results/plots/gyro_bias_impact.png)")
    parser.add_argument("--max-duration-s", type=float, default=None, metavar="FLOAT",
                        help="Truncate both series at this relative time (default: full)")
    args = parser.parse_args()

    _, _, gt_samples = load_sequence_gt(Path(args.config), args.dataset_root)
    gt_times, gt_mats = build_gt_index(gt_samples)

    try:
        raw_rows  = load_orientation_csv(Path(args.raw_csv))
        corr_rows = load_orientation_csv(Path(args.corrected_csv))
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"Error: {e}")

    if not raw_rows or not corr_rows:
        sys.exit("Error: one or both orientation CSVs are empty")

    # Both series share the reference pair from the first raw row.
    first_ts, R_est_ref = raw_rows[0]
    R_gt_ref  = nearest_by_index(first_ts, gt_mats, gt_times)
    start_s   = first_ts

    max_dur = args.max_duration_s
    t_raw,  err_raw  = compute_error_series(
        raw_rows,  start_s, max_dur, gt_times, gt_mats, R_est_ref, R_gt_ref)
    t_corr, err_corr = compute_error_series(
        corr_rows, start_s, max_dur, gt_times, gt_mats, R_est_ref, R_gt_ref)

    if not err_raw or not err_corr:
        sys.exit("Error: no rows within the requested duration")

    output_path = Path(args.output)
    make_plot(t_raw, err_raw, t_corr, err_corr, output_path)

    mean_raw   = sum(err_raw)  / len(err_raw)
    mean_corr  = sum(err_corr) / len(err_corr)

    print(f"  Rows plotted:                 {len(err_raw)} raw, {len(err_corr)} corrected")
    print(f"  Duration covered:             {max(t_raw[-1], t_corr[-1]):.1f} s")
    print(f"  Raw   [mean / final / max]:   "
          f"{mean_raw:.2f} / {err_raw[-1]:.2f} / {max(err_raw):.2f} deg")
    print(f"  Corr  [mean / final / max]:   "
          f"{mean_corr:.2f} / {err_corr[-1]:.2f} / {max(err_corr):.2f} deg")
    print(f"  Mean-error reduction:         {mean_raw / mean_corr:.1f}×")
    print(f"  Saved: {output_path}")


if __name__ == "__main__":
    main()
