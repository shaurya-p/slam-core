"""
Compare raw vs bias-corrected full IMU state propagation against EuRoC GT in Rerun.

Scalar error dashboard — three time-series panels:
  position_error/raw_norm_m
  position_error/bias_corrected_norm_m
  velocity_error/raw_norm_mps
  velocity_error/bias_corrected_norm_mps
  orientation_error/raw_geodesic_deg
  orientation_error/bias_corrected_geodesic_deg

Usage:
    uv run python scripts/rerun/rerun_euroc_imu_state_bias_comparison.py \\
      results/imu/MH_01_easy_imu_state.csv \\
      results/imu/MH_01_easy_imu_state_bias_corrected.csv \\
      --config configs/datasets/euroc_mh01.yaml \\
      --dataset-root "$HOME/datasets" \\
      --max-duration-s 180 \\
      --output results/rerun/MH_01_easy_imu_state_bias_comparison.rrd
"""

import argparse
import bisect
import csv
import math
import sys
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from slam_core_tools.datasets.euroc import (
    load_config,
    read_groundtruth_csv,
    resolve_sequence_root,
)

_EXPECTED_COLS = {
    "timestamp_s",
    "p_x", "p_y", "p_z",
    "v_x", "v_y", "v_z",
    "q_w", "q_x", "q_y", "q_z",
    "r00", "r01", "r02",
    "r10", "r11", "r12",
    "r20", "r21", "r22",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_state_csv(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"State CSV not found: {csv_path}")
    rows: list[dict] = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not _EXPECTED_COLS.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"State CSV missing required columns.\n"
                f"Expected subset: {sorted(_EXPECTED_COLS)}\n"
                f"Got: {reader.fieldnames}"
            )
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def _quat_to_mat(w: float, x: float, y: float, z: float) -> np.ndarray:
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y)],
        [2*(x*y + w*z),      1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [2*(x*z - w*y),      2*(y*z + w*x),        1 - 2*(x*x + y*y)],
    ])


def _geodesic_deg(R_err: np.ndarray) -> float:
    cos_angle = (np.trace(R_err) - 1.0) / 2.0
    return math.acos(max(-1.0, min(1.0, float(cos_angle)))) * 180.0 / math.pi


def _nearest(ts: float, samples: list, times: list[float]):
    idx = bisect.bisect_left(times, ts)
    if idx == 0:
        return samples[0]
    if idx >= len(samples):
        return samples[-1]
    if ts - times[idx - 1] <= times[idx] - ts:
        return samples[idx - 1]
    return samples[idx]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare raw vs bias-corrected IMU state propagation in Rerun."
    )
    parser.add_argument("raw_csv",       help="Raw IMU state CSV from export_imu_state_propagation")
    parser.add_argument("corrected_csv", help="Bias-corrected IMU state CSV")
    parser.add_argument("--config",       required=True, metavar="PATH",
                        help="EuRoC sequence config YAML")
    parser.add_argument("--dataset-root", metavar="PATH",
                        help="Parent directory containing euroc/<sequence>/")
    parser.add_argument("--frame-stride", type=int, default=50, metavar="N",
                        help="(no-op, retained for compatibility)")
    parser.add_argument("--axis-length",  type=float, default=1.0, metavar="FLOAT",
                        help="(no-op, retained for compatibility)")
    parser.add_argument("--max-duration-s", type=float, default=None, metavar="FLOAT",
                        help="Truncate at this relative time in seconds (default: full)")
    parser.add_argument("--output", metavar="PATH",
                        help="Output .rrd path (default: results/rerun/<sequence>_imu_state_bias_comparison.rrd)")
    args = parser.parse_args()

    try:
        cfg      = load_config(Path(args.config))
        seq_root = resolve_sequence_root(args.dataset_root, cfg)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"Error: {e}")

    sequence = cfg["sequence"]

    gt_csv = seq_root / "mav0/state_groundtruth_estimate0/data.csv"
    try:
        gt_samples = read_groundtruth_csv(gt_csv)
    except FileNotFoundError:
        sys.exit(f"Error: GT CSV not found: {gt_csv}")
    gt_times = [s.timestamp_s for s in gt_samples]

    try:
        raw_rows  = _load_state_csv(Path(args.raw_csv))
        corr_rows = _load_state_csv(Path(args.corrected_csv))
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"Error: {e}")

    if not raw_rows or not corr_rows:
        sys.exit("Error: one or both state CSVs are empty")

    print(f"  Raw  rows:  {len(raw_rows)}  ({args.raw_csv})")
    print(f"  Corr rows:  {len(corr_rows)}  ({args.corrected_csv})")
    print(f"  GT samples: {len(gt_samples)}  ({gt_csv})")

    if args.output:
        rrd_path = Path(args.output)
    else:
        rrd_path = Path("results/rerun") / f"{sequence}_imu_state_bias_comparison.rrd"
    rrd_path.parent.mkdir(parents=True, exist_ok=True)

    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.TimeSeriesView(name="Position Error (m)",      origin="position_error"),
            rrb.TimeSeriesView(name="Velocity Error (m/s)",    origin="velocity_error"),
            rrb.TimeSeriesView(name="Orientation Error (deg)", origin="orientation_error"),
            row_shares=[3, 1, 1],
        ),
    )

    rr.init(f"slam_core/{sequence}/imu_state_bias_comparison", spawn=False)
    rr.save(str(rrd_path), default_blueprint=blueprint)

    start_s = raw_rows[0]["timestamp_s"]
    n_scalars = 0

    for raw, cor in zip(raw_rows, corr_rows):
        ts    = raw["timestamp_s"]
        t_rel = ts - start_s
        if args.max_duration_s is not None and t_rel > args.max_duration_s:
            break

        rr.set_time("time", duration=t_rel)

        p_raw = np.array([raw["p_x"], raw["p_y"], raw["p_z"]])
        p_cor = np.array([cor["p_x"], cor["p_y"], cor["p_z"]])
        v_raw = np.array([raw["v_x"], raw["v_y"], raw["v_z"]])
        v_cor = np.array([cor["v_x"], cor["v_y"], cor["v_z"]])
        R_raw = np.array([[raw["r00"], raw["r01"], raw["r02"]],
                          [raw["r10"], raw["r11"], raw["r12"]],
                          [raw["r20"], raw["r21"], raw["r22"]]])
        R_cor = np.array([[cor["r00"], cor["r01"], cor["r02"]],
                          [cor["r10"], cor["r11"], cor["r12"]],
                          [cor["r20"], cor["r21"], cor["r22"]]])

        gt = _nearest(ts, gt_samples, gt_times)
        p_gt = np.array([gt.p_x, gt.p_y, gt.p_z])
        v_gt = np.array([gt.v_x, gt.v_y, gt.v_z])
        q_norm = math.sqrt(gt.q_w**2 + gt.q_x**2 + gt.q_y**2 + gt.q_z**2)
        if q_norm < 1e-10:
            continue
        R_gt = _quat_to_mat(gt.q_w/q_norm, gt.q_x/q_norm,
                            gt.q_y/q_norm, gt.q_z/q_norm)

        rr.log("position_error/raw_norm_m",            rr.Scalars(float(np.linalg.norm(p_raw - p_gt))))
        rr.log("position_error/bias_corrected_norm_m", rr.Scalars(float(np.linalg.norm(p_cor - p_gt))))
        rr.log("velocity_error/raw_norm_mps",            rr.Scalars(float(np.linalg.norm(v_raw - v_gt))))
        rr.log("velocity_error/bias_corrected_norm_mps", rr.Scalars(float(np.linalg.norm(v_cor - v_gt))))
        rr.log("orientation_error/raw_geodesic_deg",            rr.Scalars(_geodesic_deg(R_gt.T @ R_raw)))
        rr.log("orientation_error/bias_corrected_geodesic_deg", rr.Scalars(_geodesic_deg(R_gt.T @ R_cor)))
        n_scalars += 1

    print(f"  Scalar rows logged: {n_scalars}")
    if args.max_duration_s is not None:
        print(f"  Max duration:       {args.max_duration_s} s")
    print(f"  Saved: {rrd_path}")


if __name__ == "__main__":
    main()
