"""
Visualize full IMU state propagation in Rerun.

WARNING: no gravity alignment, no visual correction, no bias estimation.
Position and velocity drift is expected due to double-integrated accelerometer noise.

Reads:
  State CSV exported by export_imu_state_propagation (C++ source of truth).

CSV columns:
  timestamp_s,
  p_x,p_y,p_z,
  v_x,v_y,v_z,
  q_w,q_x,q_y,q_z,
  r00,r01,r02,r10,r11,r12,r20,r21,r22,
  gyro_bias_x,gyro_bias_y,gyro_bias_z,
  accel_bias_x,accel_bias_y,accel_bias_z

Propagation is NOT recomputed here.

Usage (CSV-only):
    uv run python scripts/rerun/rerun_euroc_imu_state_propagation.py \\
      results/imu/MH_01_easy_imu_state.csv

Usage (with GT comparison):
    uv run python scripts/rerun/rerun_euroc_imu_state_propagation.py \\
      results/imu/MH_01_easy_imu_state.csv \\
      --config configs/datasets/euroc_mh01.yaml \\
      --dataset-root "$HOME/datasets" \\
      --frame-stride 50 \\
      --max-duration-s 30 \\
      --output results/rerun/MH_01_easy_imu_state_vs_gt.rrd
"""

import argparse
import bisect
import csv
import math
import sys
from pathlib import Path

import numpy as np
import rerun as rr

_DRIFT_WARNING = (
    "WARNING: no gravity alignment, no visual correction, no bias estimation. "
    "Position and velocity drift is expected."
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

# Estimated: red x-axis, gray y/z (same palette as gyro visualization)
_EST_X_COLOR   = [255,   0,   0]
_EST_AUX_COLOR = [170, 170, 170]
# GT: blue x-axis, dark gray y/z (same palette as gyro visualization)
_GT_X_COLOR    = [  0,   0, 255]
_GT_AUX_COLOR  = [ 90,  90,  90]

_AUX_SCALE = 0.45  # y/z axis length as fraction of main axis length


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_state_csv(csv_path: Path) -> list[dict]:
    """Read exported state CSV. Returns list of row dicts with float values.

    Raises:
        FileNotFoundError: if csv_path does not exist.
        ValueError: if required header columns are missing.
    """
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


# ---------------------------------------------------------------------------
# Local geometry helpers (visualization utilities only, not algorithm logic)
# ---------------------------------------------------------------------------

def _quat_to_mat(w: float, x: float, y: float, z: float) -> np.ndarray:
    """3x3 rotation matrix from unit quaternion (Hamilton convention, w scalar first).

    Same implementation as the gyro propagation visualization.
    """
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y)],
        [2*(x*y + w*z),      1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [2*(x*z - w*y),      2*(y*z + w*x),        1 - 2*(x*x + y*y)],
    ])


def _geodesic_deg(R_err: np.ndarray) -> float:
    """SO(3) geodesic angle in degrees from R_err = R_gt.T @ R_est.

    Same convention as the gyro propagation visualization.
    """
    cos_angle = (np.trace(R_err) - 1.0) / 2.0
    return math.acos(max(-1.0, min(1.0, float(cos_angle)))) * 180.0 / math.pi


# ---------------------------------------------------------------------------
# Rerun helpers
# ---------------------------------------------------------------------------

def _set_time(t_s: float, start_s: float) -> None:
    rr.set_time("time", duration=t_s - start_s)


def _log_axes(entity: str, R: np.ndarray, length: float,
              x_color: list[int], aux_color: list[int],
              position: list[float] | None = None) -> None:
    if position is None:
        position = [0.0, 0.0, 0.0]
    rr.log(
        entity,
        rr.Arrows3D(
            origins=[position, position, position],
            vectors=[
                R[:, 0] * length,
                R[:, 1] * length * _AUX_SCALE,
                R[:, 2] * length * _AUX_SCALE,
            ],
            colors=[x_color, aux_color, aux_color],
        ),
    )


def _nearest_by_index(ts: float, samples: list, times: list[float]):
    """Nearest-neighbor using a pre-built sorted times list.

    Callers build times once before a loop to avoid O(n) rebuilding per call.
    Equivalent to nearest_gt_sample(ts, samples) but reuses an existing index.
    """
    idx = bisect.bisect_left(times, ts)
    if idx == 0:
        return samples[0]
    if idx >= len(samples):
        return samples[-1]
    if ts - times[idx - 1] <= times[idx] - ts:
        return samples[idx - 1]
    return samples[idx]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_imu_state(
    rows: list[dict],
    start_s: float,
    frame_stride: int,
    max_duration_s: float | None,
    axis_length: float,
    gt_samples: list | None = None,
) -> tuple[int, int]:
    """Log estimated state and optional GT comparison to Rerun.

    Always logs:
      imu_state/* scalar plots, estimated/trajectory, estimated/body_frame

    When gt_samples is provided, also logs:
      ground_truth/trajectory, ground_truth/body_frame,
      error/position_{x,y,z,norm}_m, error/velocity_norm_mps,
      error/orientation_geodesic_deg, ground_truth/velocity_norm_mps

    Returns (scalar_rows_logged, body_frames_logged).
    """
    # Pre-build GT index once to avoid rebuilding per row in the loop.
    gt_times: list[float] = [s.timestamp_s for s in gt_samples] if gt_samples else []

    # Display origin: shift all 3D spatial entities so the first estimated position
    # is at the world origin. Scalar/error plots use raw values.
    p0 = np.array([rows[0]["p_x"], rows[0]["p_y"], rows[0]["p_z"]])

    est_positions: list[list[float]] = []
    n_scalars = 0
    n_frames  = 0

    for i, row in enumerate(rows):
        ts = row["timestamp_s"]
        if max_duration_s is not None and ts - start_s > max_duration_s:
            break

        p     = np.array([row["p_x"], row["p_y"], row["p_z"]])
        v     = np.array([row["v_x"], row["v_y"], row["v_z"]])
        R_est = np.array([
            [row["r00"], row["r01"], row["r02"]],
            [row["r10"], row["r11"], row["r12"]],
            [row["r20"], row["r21"], row["r22"]],
        ])
        p_disp = p - p0
        est_positions.append(p_disp.tolist())
        _set_time(ts, start_s)

        # Estimated scalar plots (always present, including CSV-only mode)
        rr.log("imu_state/position_x_m",      rr.Scalars(p[0]))
        rr.log("imu_state/position_y_m",      rr.Scalars(p[1]))
        rr.log("imu_state/position_z_m",      rr.Scalars(p[2]))
        rr.log("imu_state/velocity_x_mps",    rr.Scalars(v[0]))
        rr.log("imu_state/velocity_y_mps",    rr.Scalars(v[1]))
        rr.log("imu_state/velocity_z_mps",    rr.Scalars(v[2]))
        rr.log("imu_state/position_norm_m",   rr.Scalars(float(np.linalg.norm(p))))
        rr.log("imu_state/velocity_norm_mps", rr.Scalars(float(np.linalg.norm(v))))
        n_scalars += 1

        # Estimated body-frame axes at stride
        if i % frame_stride == 0:
            _log_axes("estimated/body_frame", R_est, axis_length,
                      _EST_X_COLOR, _EST_AUX_COLOR, position=p_disp.tolist())
            n_frames += 1

        # GT comparison (only when GT was loaded)
        if gt_samples:
            gt = _nearest_by_index(ts, gt_samples, gt_times)
            gt_p = np.array([gt.p_x, gt.p_y, gt.p_z])
            gt_v = np.array([gt.v_x, gt.v_y, gt.v_z])

            # Normalize quaternion before converting to rotation matrix
            q_norm = math.sqrt(gt.q_w**2 + gt.q_x**2 + gt.q_y**2 + gt.q_z**2)
            if q_norm < 1e-10:
                continue
            R_gt = _quat_to_mat(
                gt.q_w / q_norm, gt.q_x / q_norm,
                gt.q_y / q_norm, gt.q_z / q_norm,
            )

            p_err = p - gt_p
            v_err = v - gt_v
            rr.log("error/position_x_m",          rr.Scalars(p_err[0]))
            rr.log("error/position_y_m",          rr.Scalars(p_err[1]))
            rr.log("error/position_z_m",          rr.Scalars(p_err[2]))
            rr.log("error/position_norm_m",       rr.Scalars(float(np.linalg.norm(p_err))))
            rr.log("error/velocity_norm_mps",     rr.Scalars(float(np.linalg.norm(v_err))))
            rr.log("ground_truth/velocity_norm_mps",
                   rr.Scalars(float(np.linalg.norm(gt_v))))

            # Orientation error: R_err = R_gt.T @ R_est
            # Same geodesic convention as the gyro propagation visualization.
            R_err = R_gt.T @ R_est
            rr.log("error/orientation_geodesic_deg", rr.Scalars(_geodesic_deg(R_err)))

            if i % frame_stride == 0:
                _log_axes("ground_truth/body_frame", R_gt, axis_length,
                          _GT_X_COLOR, _GT_AUX_COLOR,
                          position=(gt_p - p0).tolist())

    # Static trajectory strips (logged after the time-series loop)
    if len(est_positions) >= 2:
        rr.log("estimated/trajectory", rr.LineStrips3D([est_positions]), static=True)

    if gt_samples:
        gt_positions = [
            [s.p_x - p0[0], s.p_y - p0[1], s.p_z - p0[2]] for s in gt_samples
            if max_duration_s is None or s.timestamp_s - start_s <= max_duration_s
        ]
        if len(gt_positions) >= 2:
            rr.log("ground_truth/trajectory",
                   rr.LineStrips3D([gt_positions]), static=True)

    return n_scalars, n_frames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize full IMU state propagation in Rerun."
    )
    parser.add_argument(
        "state_csv",
        help="State CSV exported by export_imu_state_propagation",
    )
    parser.add_argument(
        "--config", metavar="PATH",
        help="EuRoC sequence config YAML; enables GT comparison mode",
    )
    parser.add_argument(
        "--dataset-root", metavar="PATH",
        help="Parent directory containing euroc/<sequence>/",
    )
    parser.add_argument(
        "--frame-stride", type=int, default=50, metavar="N",
        help="Log body-frame axes every Nth row (default: 50)",
    )
    parser.add_argument(
        "--axis-length", type=float, default=1.0, metavar="FLOAT",
        help="Length of the main body-frame axis arrow (default: 1.0)",
    )
    parser.add_argument(
        "--max-duration-s", type=float, default=None, metavar="FLOAT",
        help="Only log rows where timestamp_s - start_s <= this value (default: full sequence)",
    )
    parser.add_argument(
        "--output", metavar="PATH",
        help="Output .rrd path (default: results/rerun/<csv-stem>.rrd)",
    )
    args = parser.parse_args()

    print(_DRIFT_WARNING)

    csv_path = Path(args.state_csv)
    try:
        rows = load_state_csv(csv_path)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"Error: {e}")
    print(f"  State rows: {len(rows)}  ({csv_path})")

    if not rows:
        sys.exit("Error: state CSV is empty")

    # Optional GT loading — only when --config is provided
    gt_samples = None
    if args.config:
        from slam_core_tools.datasets.euroc import (
            load_config,
            nearest_gt_sample,  # imported to confirm availability; lookup uses _nearest_by_index
            read_groundtruth_csv,
            resolve_sequence_root,
        )
        try:
            cfg      = load_config(Path(args.config))
            seq_root = resolve_sequence_root(args.dataset_root, cfg)
        except (FileNotFoundError, ValueError) as e:
            sys.exit(f"Error: {e}")

        gt_csv = seq_root / "mav0/state_groundtruth_estimate0/data.csv"
        try:
            gt_samples = read_groundtruth_csv(gt_csv)
        except FileNotFoundError:
            sys.exit(
                f"Error: Ground-truth CSV not found: {gt_csv}\n"
                f"Expected EuRoC path: mav0/state_groundtruth_estimate0/data.csv"
            )
        print(f"  GT samples:  {len(gt_samples)}  ({gt_csv})")

    if args.output:
        rrd_path = Path(args.output)
    else:
        output_dir = Path("results/rerun")
        output_dir.mkdir(parents=True, exist_ok=True)
        rrd_path = output_dir / f"{csv_path.stem}.rrd"

    rrd_path.parent.mkdir(parents=True, exist_ok=True)

    rr.init(f"slam_core/{csv_path.stem}", spawn=False)
    rr.save(str(rrd_path))

    start_s = rows[0]["timestamp_s"]

    n_scalars, n_frames = log_imu_state(
        rows,
        start_s,
        frame_stride=args.frame_stride,
        max_duration_s=args.max_duration_s,
        axis_length=args.axis_length,
        gt_samples=gt_samples,
    )

    print(f"  Scalar rows logged: {n_scalars}")
    print(f"  Body frames logged: {n_frames}  (stride {args.frame_stride})")
    if args.max_duration_s is not None:
        print(f"  Max duration:       {args.max_duration_s} s")
    if gt_samples is not None:
        print(f"  GT mode:            on")
    print(f"  Saved: {rrd_path}")


if __name__ == "__main__":
    main()
