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

Usage:
    uv run python scripts/rerun_euroc_imu_state_propagation.py \\
      results/imu/MH_01_easy_imu_state.csv \\
      --frame-stride 50 \\
      --max-duration-s 30 \\
      --output results/rerun/MH_01_easy_imu_state_propagation.rrd
"""

import argparse
import csv
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

# Body-frame axis colors (same convention as gyro visualization)
_EST_X_COLOR   = [255,   0,   0]  # x-axis: red
_EST_AUX_COLOR = [170, 170, 170]  # y/z-axes: light gray

_MAIN_AXIS_LENGTH = 1.0
_AUX_AXIS_LENGTH  = 0.45


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
# Rerun helpers
# ---------------------------------------------------------------------------

def set_rerun_time(t_s: float, start_s: float) -> None:
    rr.set_time("time", duration=t_s - start_s)


def log_imu_state(
    rows: list[dict],
    start_s: float,
    frame_stride: int,
    max_duration_s: float | None,
) -> tuple[int, int]:
    """Log scalar plots for all rows and body-frame axes at stride.

    Full trajectory is logged as a static line strip after the loop.

    Returns (scalar_rows_logged, body_frames_logged).
    """
    all_positions: list[list[float]] = []
    n_scalars = 0
    n_frames  = 0

    for i, row in enumerate(rows):
        ts = row["timestamp_s"]
        if max_duration_s is not None and ts - start_s > max_duration_s:
            break

        p = np.array([row["p_x"], row["p_y"], row["p_z"]])
        v = np.array([row["v_x"], row["v_y"], row["v_z"]])
        all_positions.append(p.tolist())

        set_rerun_time(ts, start_s)

        rr.log("imu_state/position_x_m",      rr.Scalars(p[0]))
        rr.log("imu_state/position_y_m",      rr.Scalars(p[1]))
        rr.log("imu_state/position_z_m",      rr.Scalars(p[2]))
        rr.log("imu_state/velocity_x_mps",    rr.Scalars(v[0]))
        rr.log("imu_state/velocity_y_mps",    rr.Scalars(v[1]))
        rr.log("imu_state/velocity_z_mps",    rr.Scalars(v[2]))
        rr.log("imu_state/position_norm_m",   rr.Scalars(float(np.linalg.norm(p))))
        rr.log("imu_state/velocity_norm_mps", rr.Scalars(float(np.linalg.norm(v))))
        n_scalars += 1

        if i % frame_stride == 0:
            R = np.array([
                [row["r00"], row["r01"], row["r02"]],
                [row["r10"], row["r11"], row["r12"]],
                [row["r20"], row["r21"], row["r22"]],
            ])
            rr.log(
                "orientation/body_frame",
                rr.Arrows3D(
                    origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    vectors=[
                        R[:, 0] * _MAIN_AXIS_LENGTH,
                        R[:, 1] * _AUX_AXIS_LENGTH,
                        R[:, 2] * _AUX_AXIS_LENGTH,
                    ],
                    colors=[_EST_X_COLOR, _EST_AUX_COLOR, _EST_AUX_COLOR],
                ),
            )
            n_frames += 1

    # Log full trajectory as a static line strip (not time-varying).
    if len(all_positions) >= 2:
        rr.log(
            "imu_state/trajectory",
            rr.LineStrips3D([all_positions]),
            static=True,
        )

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
        "--frame-stride", type=int, default=50, metavar="N",
        help="Log body-frame axes every Nth row (default: 50)",
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
        rows, start_s, args.frame_stride, args.max_duration_s
    )

    print(f"  Scalar rows logged: {n_scalars}")
    print(f"  Body frames logged: {n_frames}  (stride {args.frame_stride})")
    if args.max_duration_s is not None:
        print(f"  Max duration:       {args.max_duration_s} s")
    print(f"  Saved: {rrd_path}")


if __name__ == "__main__":
    main()
