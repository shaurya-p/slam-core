"""
Visualize gyro-only SO(3) orientation propagation in Rerun.

WARNING: gyro-only propagation has no gravity alignment, no accelerometer
fusion, no bias correction, and no visual correction. Drift is expected.

Reads:
  1. EuRoC config YAML — for sequence name and IMU CSV path
  2. Orientation CSV exported by export_gyro_propagation (C++ source of truth)

Orientation CSV columns (row-major R_W_B entries):
  timestamp_s,r00,r01,r02,r10,r11,r12,r20,r21,r22

R_W_B maps body frame B into world frame W.
Columns of R_W_B are body x/y/z axes expressed in world frame.
Roll/pitch/yaw are ZYX Euler angles derived from the exported matrix entries.
SO(3) propagation is NOT recomputed here.

Usage:
    uv run python scripts/rerun_euroc_gyro_propagation.py \\
      configs/datasets/euroc_mh01.yaml \\
      results/imu/MH_01_easy_gyro_orientations.csv \\
      --dataset-root "$HOME/datasets" \\
      --imu-stride 2 \\
      --frame-stride 20

Output: results/rerun/<sequence>_gyro_propagation.rrd
"""

import argparse
import bisect
import csv
import math
import sys
from pathlib import Path

import numpy as np
import rerun as rr

from slam_core_tools.datasets.euroc import (
    GroundTruthSample,
    load_config,
    read_groundtruth_csv,
    read_imu_csv,
    resolve_sequence_root,
)

_DRIFT_WARNING = (
    "WARNING: gyro-only propagation has no gravity alignment, no accelerometer "
    "fusion, no bias correction, and no visual correction. Drift is expected."
)

# 3D axis colors: x-axis dominates, y/z provide faint orientation context
_EST_X_COLOR   = [255,   0,   0]   # estimated forward (x): red
_GT_X_COLOR    = [  0,   0, 255]   # GT forward (x): blue
_EST_AUX_COLOR = [170, 170, 170]   # estimated y/z: light gray
_GT_AUX_COLOR  = [ 90,  90,  90]   # GT y/z: dark gray

_MAIN_AXIS_LENGTH = 1.0   # forward (x) arrow scale
_AUX_AXIS_LENGTH  = 0.45  # y/z arrow scale

_EXPECTED_COLS = {
    "timestamp_s", "r00", "r01", "r02",
    "r10", "r11", "r12",
    "r20", "r21", "r22",
}


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_orientations(csv_path: Path) -> list[tuple[float, list[float]]]:
    """Read exported orientation CSV.

    Returns list of (timestamp_s, [r00, r01, ..., r22]) in row-major order.

    Raises:
        FileNotFoundError: if csv_path does not exist.
        ValueError: if required header columns are missing.
    """
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
# Derived quantities — local visualization helpers, not algorithm logic
# ---------------------------------------------------------------------------

def rpy_from_mat(R: np.ndarray) -> tuple[float, float, float]:
    """ZYX Euler angles (rad) from 3x3 rotation matrix.

    roll  = atan2(R[2,1], R[2,2])
    pitch = asin(clamp(-R[2,0], -1, 1))
    yaw   = atan2(R[1,0], R[0,0])
    """
    roll  = math.atan2(float(R[2, 1]), float(R[2, 2]))
    pitch = math.asin(max(-1.0, min(1.0, float(-R[2, 0]))))
    yaw   = math.atan2(float(R[1, 0]), float(R[0, 0]))
    return roll, pitch, yaw


def quat_to_mat(w: float, x: float, y: float, z: float) -> np.ndarray:
    """3x3 rotation matrix from unit quaternion (Hamilton convention, w scalar first)."""
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x)],
        [2*(x*z - w*y),      2*(y*z + w*x),       1 - 2*(x*x + y*y)],
    ])


def build_gt_index(
    samples: list[GroundTruthSample],
) -> tuple[list[float], list[np.ndarray]]:
    """Sorted timestamps and precomputed rotation matrices for nearest-neighbor lookup.

    Each quaternion is normalized before conversion.
    """
    times: list[float] = []
    mats:  list[np.ndarray] = []
    for s in samples:
        norm = math.sqrt(s.q_w**2 + s.q_x**2 + s.q_y**2 + s.q_z**2)
        if norm < 1e-10:
            continue
        w, x, y, z = s.q_w/norm, s.q_x/norm, s.q_y/norm, s.q_z/norm
        times.append(s.timestamp_s)
        mats.append(quat_to_mat(w, x, y, z))
    return times, mats


def nearest_gt_mat(ts: float, gt_times: list[float],
                   gt_mats: list[np.ndarray]) -> np.ndarray:
    """Nearest-neighbor GT rotation matrix for timestamp ts."""
    idx = bisect.bisect_left(gt_times, ts)
    if idx == 0:
        return gt_mats[0]
    if idx == len(gt_times):
        return gt_mats[-1]
    if ts - gt_times[idx - 1] <= gt_times[idx] - ts:
        return gt_mats[idx - 1]
    return gt_mats[idx]


def geodesic_deg(R_err: np.ndarray) -> float:
    """SO(3) geodesic angle in degrees from an error rotation matrix."""
    cos_angle = (np.trace(R_err) - 1.0) / 2.0
    return math.acos(max(-1.0, min(1.0, float(cos_angle)))) * 180.0 / math.pi


# ---------------------------------------------------------------------------
# Rerun helpers
# ---------------------------------------------------------------------------

def set_rerun_time(t_s: float, start_s: float) -> None:
    rr.set_time("time", duration=t_s - start_s)


def log_gyro(samples, start_s: float, stride: int,
             max_duration_s: float | None = None) -> int:
    logged = 0
    for i, s in enumerate(samples):
        if max_duration_s is not None and s.timestamp_s - start_s > max_duration_s:
            break  # samples are chronologically ordered
        if i % stride != 0:
            continue
        gx, gy, gz = s.gyro_radps
        set_rerun_time(s.timestamp_s, start_s)
        # rr.log("imu/gyro/x", rr.Scalars(gx))
        # rr.log("imu/gyro/y", rr.Scalars(gy))
        # rr.log("imu/gyro/z", rr.Scalars(gz))
        logged += 1
    return logged


def log_orientations(
    rows: list[tuple[float, list[float]]],
    start_s: float,
    stride: int,
    max_duration_s: float | None,
    gt_times: list[float],
    gt_mats: list[np.ndarray],
    R_est_ref: np.ndarray,
    R_gt_ref: np.ndarray,
) -> tuple[int, list[float]]:
    """Log estimated RPY + 3D frame, GT RPY + 3D frame, and geodesic error.

    Returns (frames_logged, error_deg_per_logged_frame).
    """
    logged = 0
    error_degs: list[float] = []

    for i, (ts, mat) in enumerate(rows):
        if max_duration_s is not None and ts - start_s > max_duration_s:
            break  # rows are chronologically ordered
        if i % stride != 0:
            continue

        R_est = np.array(mat).reshape(3, 3)

        # Nearest GT and relative comparison using first matched pair as reference:
        # R_est_rel(t) = R_est_ref.T @ R_est(t)
        # R_gt_rel(t)  = R_gt_ref.T  @ R_gt(t)
        # R_err        = R_gt_rel.T  @ R_est_rel
        R_gt = nearest_gt_mat(ts, gt_times, gt_mats)
        R_est_rel = R_est_ref.T @ R_est
        R_gt_rel  = R_gt_ref.T  @ R_gt
        R_err     = R_gt_rel.T  @ R_est_rel
        err_deg   = geodesic_deg(R_err)
        error_degs.append(err_deg)

        # Both RPY and 3D arrows use relative rotations — consistent with error metric,
        # both frames coincide at the reference timestamp.
        roll,    pitch,    yaw    = rpy_from_mat(R_est_rel)
        roll_gt, pitch_gt, yaw_gt = rpy_from_mat(R_gt_rel)

        # ZYX error components (interpretability only — geodesic_deg is the main metric)
        roll_err_rad, pitch_err_rad, yaw_err_rad = rpy_from_mat(R_err)

        set_rerun_time(ts, start_s)

        # Estimated orientation (relative to reference)
        # x-axis: red (full length); y/z: light gray (shorter, orientation context only)
        # rr.log("orientation/roll_rad",  rr.Scalars(roll))
        # rr.log("orientation/pitch_rad", rr.Scalars(pitch))
        # rr.log("orientation/yaw_rad",   rr.Scalars(yaw))
        rr.log("orientation/body_frame",
               rr.Arrows3D(
                   origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                   vectors=[
                       R_est_rel[:, 0] * _MAIN_AXIS_LENGTH,
                       R_est_rel[:, 1] * _AUX_AXIS_LENGTH,
                       R_est_rel[:, 2] * _AUX_AXIS_LENGTH,
                   ],
                   colors=[_EST_X_COLOR, _EST_AUX_COLOR, _EST_AUX_COLOR],
               ))

        # GT orientation (relative to reference)
        # x-axis: blue (full length); y/z: dark gray (shorter)
        # rr.log("orientation_gt/roll_rad",  rr.Scalars(roll_gt))
        # rr.log("orientation_gt/pitch_rad", rr.Scalars(pitch_gt))
        # rr.log("orientation_gt/yaw_rad",   rr.Scalars(yaw_gt))
        rr.log("orientation/body_frame_gt",
               rr.Arrows3D(
                   origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                   vectors=[
                       R_gt_rel[:, 0] * _MAIN_AXIS_LENGTH,
                       R_gt_rel[:, 1] * _AUX_AXIS_LENGTH,
                       R_gt_rel[:, 2] * _AUX_AXIS_LENGTH,
                   ],
                   colors=[_GT_X_COLOR, _GT_AUX_COLOR, _GT_AUX_COLOR],
               ))

        # Orientation error scalar plots
        rr.log("orientation_error/geodesic_deg", rr.Scalars(err_deg))
        rr.log("orientation_error/roll_deg",  rr.Scalars(roll_err_rad  * 180.0 / math.pi))
        rr.log("orientation_error/pitch_deg", rr.Scalars(pitch_err_rad * 180.0 / math.pi))
        rr.log("orientation_error/yaw_deg",   rr.Scalars(yaw_err_rad   * 180.0 / math.pi))

        logged += 1
    return logged, error_degs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize gyro-only orientation propagation in Rerun."
    )
    parser.add_argument(
        "config",
        help="EuRoC sequence config YAML (e.g. configs/datasets/euroc_mh01.yaml)",
    )
    parser.add_argument(
        "orientation_csv",
        help="Orientation CSV from export_gyro_propagation (C++ source of truth)",
    )
    parser.add_argument(
        "--dataset-root", metavar="PATH",
        help="Parent directory containing euroc/<sequence>/",
    )
    parser.add_argument(
        "--imu-stride", type=int, default=10, metavar="N",
        help="Log every Nth IMU sample for raw gyro (default: 10)",
    )
    parser.add_argument(
        "--frame-stride", type=int, default=50, metavar="N",
        help="Log every Nth orientation row for RPY + body frame (default: 50)",
    )
    parser.add_argument(
        "--max-duration-s", type=float, default=None, metavar="FLOAT",
        help="Only log samples where timestamp_s - start_s <= this value (default: full sequence)",
    )
    args = parser.parse_args()

    print(_DRIFT_WARNING)

    config_path = Path(args.config)
    try:
        cfg      = load_config(config_path)
        seq_root = resolve_sequence_root(args.dataset_root, cfg)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"Error: {e}")

    sequence = cfg["sequence"]
    print(f"Sequence: {sequence}")
    print(f"Root:     {seq_root}")

    try:
        imu_samples = read_imu_csv(seq_root / cfg["imu_csv"])
    except FileNotFoundError as e:
        sys.exit(f"Error: {e}")
    print(f"  IMU: {len(imu_samples)} samples")

    try:
        orient_rows = load_orientations(Path(args.orientation_csv))
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"Error: {e}")
    print(f"  Orientations: {len(orient_rows)} rows  ({args.orientation_csv})")

    # Ground-truth — required; fail clearly if absent
    gt_csv_path = seq_root / "mav0/state_groundtruth_estimate0/data.csv"
    try:
        gt_samples = read_groundtruth_csv(gt_csv_path)
    except FileNotFoundError:
        sys.exit(
            f"Error: Ground-truth CSV not found: {gt_csv_path}\n"
            f"Expected EuRoC path relative to sequence root: "
            f"mav0/state_groundtruth_estimate0/data.csv"
        )
    print(f"  GT: {len(gt_samples)} samples")
    gt_times, gt_mats = build_gt_index(gt_samples)

    # Reference pair: first orientation row matched to nearest GT.
    # All relative comparisons use this pair as t0.
    first_ts, first_mat = orient_rows[0]
    R_est_ref = np.array(first_mat).reshape(3, 3)
    R_gt_ref  = nearest_gt_mat(first_ts, gt_times, gt_mats)

    output_dir = Path("results/rerun")
    output_dir.mkdir(parents=True, exist_ok=True)
    rrd_path = output_dir / f"{sequence}_gyro_propagation.rrd"

    rr.init(f"slam_core/{sequence}/gyro_propagation", spawn=False)
    rr.save(str(rrd_path))

    start_s = imu_samples[0].timestamp_s if imu_samples else 0.0

    max_dur = args.max_duration_s
    n_gyro = log_gyro(imu_samples, start_s, args.imu_stride, max_dur)
    n_orient, error_degs = log_orientations(
        orient_rows, start_s, args.frame_stride, max_dur,
        gt_times, gt_mats, R_est_ref, R_gt_ref,
    )

    print(f"  Gyro logged:        {n_gyro} samples  (stride {args.imu_stride})")
    print(f"  Orientation logged: {n_orient} frames  (stride {args.frame_stride})")
    if max_dur is not None:
        print(f"  Max duration:       {max_dur} s")
    print(f"  Saved: {rrd_path}")

    if error_degs:
        print(f"  Error initial:      {error_degs[0]:.2f} deg")
        print(f"  Error final:        {error_degs[-1]:.2f} deg")
        print(f"  Error min:          {min(error_degs):.2f} deg")
        print(f"  Error max:          {max(error_degs):.2f} deg")
        print(f"  Error mean:         {sum(error_degs)/len(error_degs):.2f} deg")


if __name__ == "__main__":
    main()
