"""
Visualize the EuRoC IMU-chain optimization (optimize_imu_chain) in Rerun.

Shows GT vs dead-reckoned vs optimized keyframe trajectories, position
error over time for both (the before/after of the factor graph), and the
estimated gyro/accel bias evolution.

The dead-reckoned trajectory diverges by hundreds of meters; in the 3D
view it is clipped once its error exceeds --dr-clip-m so the scene stays
readable (the full error curve is in the time-series panel).

Usage:
    ./build/tools/optimize_imu_chain \\
      "$SLAM_CORE_DATASETS/euroc/MH_01_easy/mav0/imu0/data.csv" \\
      results/optim/MH_01_easy_imu_chain.csv
    uv run python scripts/rerun/rerun_imu_chain.py \\
      results/optim/MH_01_easy_imu_chain.csv
    rerun results/rerun/MH_01_easy_imu_chain.rrd

Output: results/rerun/<csv-stem>.rrd (or --output)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from slam_core_tools.viz import GT_X_COLOR, SERIES_X_COLORS, load_csv_rows, set_time_since

_EXPECTED_COLS = frozenset({
    "timestamp_s",
    "gt_p_x", "gt_p_y", "gt_p_z",
    "dr_p_x", "dr_p_y", "dr_p_z", "dr_pos_err_m", "dr_rot_err_deg",
    "opt_p_x", "opt_p_y", "opt_p_z", "opt_pos_err_m", "opt_rot_err_deg",
    "bg_x", "bg_y", "bg_z", "ba_x", "ba_y", "ba_z",
})

_GT_COLOR  = GT_X_COLOR          # blue
_OPT_COLOR = SERIES_X_COLORS[1]  # green
_DR_COLOR  = SERIES_X_COLORS[2]  # orange


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize the IMU-chain optimization before/after in Rerun."
    )
    parser.add_argument("chain_csv", help="CSV written by optimize_imu_chain")
    parser.add_argument("--dr-clip-m", type=float, default=5.0, metavar="FLOAT",
                        help="Hide dead-reckon 3D points once error exceeds this (default: 5)")
    parser.add_argument("--output", metavar="PATH",
                        help="Output .rrd path (default: results/rerun/<csv-stem>.rrd)")
    args = parser.parse_args()

    csv_path = Path(args.chain_csv)
    try:
        rows = load_csv_rows(csv_path, _EXPECTED_COLS, "Chain")
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"Error: {e}")
    if not rows:
        sys.exit("Error: chain CSV is empty")

    rrd_path = Path(args.output) if args.output else Path("results/rerun") / f"{csv_path.stem}.rrd"
    rrd_path.parent.mkdir(parents=True, exist_ok=True)

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(name="Trajectories", origin="chain"),
            rrb.Vertical(
                rrb.TimeSeriesView(name="Position Error (m)", origin="position_error"),
                rrb.TimeSeriesView(name="Gyro Bias (rad/s)",  origin="gyro_bias"),
                rrb.TimeSeriesView(name="Accel Bias (m/s^2)", origin="accel_bias"),
            ),
            column_shares=[3, 2],
        ),
    )

    rr.init(f"slam_core/{csv_path.stem}", spawn=False)
    rr.save(str(rrd_path), default_blueprint=blueprint)

    start_s = rows[0]["timestamp_s"]
    p0      = np.array([rows[0]["gt_p_x"], rows[0]["gt_p_y"], rows[0]["gt_p_z"]])

    gt_pts, opt_pts, dr_pts = [], [], []
    for row in rows:
        set_time_since(row["timestamp_s"], start_s)

        gt  = np.array([row["gt_p_x"], row["gt_p_y"], row["gt_p_z"]]) - p0
        opt = np.array([row["opt_p_x"], row["opt_p_y"], row["opt_p_z"]]) - p0
        gt_pts.append(gt.tolist())
        opt_pts.append(opt.tolist())
        if row["dr_pos_err_m"] <= args.dr_clip_m:
            dr_pts.append((np.array([row["dr_p_x"], row["dr_p_y"], row["dr_p_z"]]) - p0).tolist())

        rr.log("position_error/dead_reckoning", rr.Scalars(row["dr_pos_err_m"]))
        rr.log("position_error/optimized",      rr.Scalars(row["opt_pos_err_m"]))
        rr.log("gyro_bias/x",  rr.Scalars(row["bg_x"]))
        rr.log("gyro_bias/y",  rr.Scalars(row["bg_y"]))
        rr.log("gyro_bias/z",  rr.Scalars(row["bg_z"]))
        rr.log("accel_bias/x", rr.Scalars(row["ba_x"]))
        rr.log("accel_bias/y", rr.Scalars(row["ba_y"]))
        rr.log("accel_bias/z", rr.Scalars(row["ba_z"]))

    rr.log("chain/ground_truth",
           rr.LineStrips3D([gt_pts], colors=[_GT_COLOR], radii=0.01), static=True)
    rr.log("chain/optimized",
           rr.Points3D(opt_pts, colors=_OPT_COLOR, radii=0.03), static=True)
    if len(dr_pts) >= 2:
        rr.log("chain/dead_reckoning_clipped",
               rr.LineStrips3D([dr_pts], colors=[_DR_COLOR], radii=0.01), static=True)

    final = rows[-1]
    print(f"  Keyframes: {len(rows)}")
    print(f"  Final errors: dead-reckon {final['dr_pos_err_m']:.2f} m, "
          f"optimized {final['opt_pos_err_m']:.3f} m")
    print(f"  DR points shown in 3D: {len(dr_pts)} (clip {args.dr_clip_m} m)")
    print(f"  Saved: {rrd_path}")


if __name__ == "__main__":
    main()
