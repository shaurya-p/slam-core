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
import csv
import math
import sys
from pathlib import Path

import rerun as rr

from slam_core_tools.datasets.euroc import (
    load_config,
    read_imu_csv,
    resolve_sequence_root,
)

_DRIFT_WARNING = (
    "WARNING: gyro-only propagation has no gravity alignment, no accelerometer "
    "fusion, no bias correction, and no visual correction. Drift is expected."
)

# Body-axis arrow colors: x=red, y=green, z=blue
_AXIS_COLORS = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]

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
# Derived quantities
# ---------------------------------------------------------------------------

def rpy_from_row_major(mat: list[float]) -> tuple[float, float, float]:
    """ZYX Euler angles (rad) from flat row-major R_W_B.

    mat layout: [r00, r01, r02, r10, r11, r12, r20, r21, r22]
    roll  = atan2(r21, r22)
    pitch = asin(clamp(-r20, -1, 1))
    yaw   = atan2(r10, r00)
    """
    r00 = mat[0]
    r10 = mat[3]
    r20 = mat[6]; r21 = mat[7]; r22 = mat[8]
    roll  = math.atan2(r21, r22)
    pitch = math.asin(max(-1.0, min(1.0, -r20)))
    yaw   = math.atan2(r10, r00)
    return roll, pitch, yaw


# ---------------------------------------------------------------------------
# Rerun helpers
# ---------------------------------------------------------------------------

def set_rerun_time(t_s: float, start_s: float) -> None:
    rr.set_time("time", duration=t_s - start_s)


def log_gyro(samples, start_s: float, stride: int) -> int:
    logged = 0
    for i, s in enumerate(samples):
        if i % stride != 0:
            continue
        gx, gy, gz = s.gyro_radps
        set_rerun_time(s.timestamp_s, start_s)
        rr.log("imu/gyro/x", rr.Scalars(gx))
        rr.log("imu/gyro/y", rr.Scalars(gy))
        rr.log("imu/gyro/z", rr.Scalars(gz))
        logged += 1
    return logged


def log_orientations(rows: list[tuple[float, list[float]]], start_s: float,
                     stride: int) -> int:
    """Log RPY scalars and body-frame arrows at the given stride."""
    logged = 0
    for i, (ts, mat) in enumerate(rows):
        if i % stride != 0:
            continue
        roll, pitch, yaw = rpy_from_row_major(mat)

        # Columns of R_W_B = body axes in world frame (row-major indexing):
        # bx = col 0: [r00, r10, r20] = [mat[0], mat[3], mat[6]]
        # by = col 1: [r01, r11, r21] = [mat[1], mat[4], mat[7]]
        # bz = col 2: [r02, r12, r22] = [mat[2], mat[5], mat[8]]
        bx = [mat[0], mat[3], mat[6]]
        by = [mat[1], mat[4], mat[7]]
        bz = [mat[2], mat[5], mat[8]]

        set_rerun_time(ts, start_s)
        rr.log("orientation/roll_rad",  rr.Scalars(roll))
        rr.log("orientation/pitch_rad", rr.Scalars(pitch))
        rr.log("orientation/yaw_rad",   rr.Scalars(yaw))
        rr.log(
            "orientation/body_frame",
            rr.Arrows3D(
                origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                vectors=[bx, by, bz],
                colors=_AXIS_COLORS,
            ),
        )
        logged += 1
    return logged


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

    output_dir = Path("results/rerun")
    output_dir.mkdir(parents=True, exist_ok=True)
    rrd_path = output_dir / f"{sequence}_gyro_propagation.rrd"

    rr.init(f"slam_core/{sequence}/gyro_propagation", spawn=False)
    rr.save(str(rrd_path))

    start_s = imu_samples[0].timestamp_s if imu_samples else 0.0

    n_gyro   = log_gyro(imu_samples, start_s, args.imu_stride)
    n_orient = log_orientations(orient_rows, start_s, args.frame_stride)

    print(f"  Gyro logged:        {n_gyro} samples  (stride {args.imu_stride})")
    print(f"  Orientation logged: {n_orient} frames  (stride {args.frame_stride})")
    print(f"  Saved: {rrd_path}")


if __name__ == "__main__":
    main()
