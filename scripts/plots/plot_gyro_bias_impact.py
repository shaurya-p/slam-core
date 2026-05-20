"""
Plot gyro bias correction impact on orientation drift.

Compares raw vs bias-corrected gyro-only propagation against EuRoC GT.

Error convention — identical to scripts/rerun/rerun_euroc_gyro_propagation.py:
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
import bisect
import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from slam_core_tools.datasets.euroc import (
    GroundTruthSample,
    load_config,
    read_groundtruth_csv,
    resolve_sequence_root,
)

_EXPECTED_COLS = {
    "timestamp_s", "r00", "r01", "r02",
    "r10", "r11", "r12",
    "r20", "r21", "r22",
}


# ---------------------------------------------------------------------------
# Geometry helpers — kept identical to rerun_euroc_gyro_propagation.py
# ---------------------------------------------------------------------------

def quat_to_mat(w: float, x: float, y: float, z: float) -> np.ndarray:
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x)],
        [2*(x*z - w*y),      2*(y*z + w*x),       1 - 2*(x*x + y*y)],
    ])


def build_gt_index(
    samples: list[GroundTruthSample],
) -> tuple[list[float], list[np.ndarray]]:
    times: list[float] = []
    mats: list[np.ndarray] = []
    for s in samples:
        norm = math.sqrt(s.q_w**2 + s.q_x**2 + s.q_y**2 + s.q_z**2)
        if norm < 1e-10:
            continue
        w, x, y, z = s.q_w/norm, s.q_x/norm, s.q_y/norm, s.q_z/norm
        times.append(s.timestamp_s)
        mats.append(quat_to_mat(w, x, y, z))
    return times, mats


def nearest_gt_mat(
    ts: float,
    gt_times: list[float],
    gt_mats: list[np.ndarray],
) -> np.ndarray:
    idx = bisect.bisect_left(gt_times, ts)
    if idx == 0:
        return gt_mats[0]
    if idx == len(gt_times):
        return gt_mats[-1]
    if ts - gt_times[idx - 1] <= gt_times[idx] - ts:
        return gt_mats[idx - 1]
    return gt_mats[idx]


def geodesic_deg(R_err: np.ndarray) -> float:
    cos_angle = (np.trace(R_err) - 1.0) / 2.0
    return math.acos(max(-1.0, min(1.0, float(cos_angle)))) * 180.0 / math.pi


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_orientations(csv_path: Path) -> list[tuple[float, list[float]]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Orientation CSV not found: {csv_path}")
    rows: list[tuple[float, list[float]]] = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not _EXPECTED_COLS.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"Orientation CSV missing required columns.\n"
                f"Expected: {sorted(_EXPECTED_COLS)}\n"
                f"Got: {reader.fieldnames}"
            )
        for row in reader:
            ts  = float(row["timestamp_s"])
            mat = [float(row[f"r{r}{c}"]) for r in range(3) for c in range(3)]
            rows.append((ts, mat))
    return rows


# ---------------------------------------------------------------------------
# Error series computation
# ---------------------------------------------------------------------------

def compute_error_series(
    rows: list[tuple[float, list[float]]],
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
        R_est     = np.array(mat).reshape(3, 3)
        R_gt      = nearest_gt_mat(ts, gt_times, gt_mats)
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

    try:
        cfg      = load_config(Path(args.config))
        seq_root = resolve_sequence_root(args.dataset_root, cfg)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"Error: {e}")

    gt_csv = seq_root / "mav0/state_groundtruth_estimate0/data.csv"
    try:
        gt_samples = read_groundtruth_csv(gt_csv)
    except FileNotFoundError:
        sys.exit(f"Error: GT CSV not found: {gt_csv}")
    gt_times, gt_mats = build_gt_index(gt_samples)

    try:
        raw_rows  = load_orientations(Path(args.raw_csv))
        corr_rows = load_orientations(Path(args.corrected_csv))
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"Error: {e}")

    if not raw_rows or not corr_rows:
        sys.exit("Error: one or both orientation CSVs are empty")

    # Both series share the reference pair from the first raw row.
    first_ts, first_mat = raw_rows[0]
    R_est_ref = np.array(first_mat).reshape(3, 3)
    R_gt_ref  = nearest_gt_mat(first_ts, gt_times, gt_mats)
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
