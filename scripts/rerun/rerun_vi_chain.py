"""
Visualize the EuRoC visual-inertial optimization (optimize_vi_chain) in Rerun.

The first full VI result: estimated trajectory + estimated sparse landmark
map + bias convergence, all from IMU factors + reprojection factors with
only a first-pose gauge prior.

3D view: GT trajectory (blue line), optimized keyframes (green), the
estimated landmark cloud colored by position error, and gray residual
lines from each estimated landmark to its ground-truth position.
Panels: keyframe position error (init vs optimized, log-friendly),
gyro/accel bias estimates over the chain.

Usage:
    ./build/tools/optimize_vi_chain \\
      "$SLAM_CORE_DATASETS/euroc/MH_01_easy/mav0/imu0/data.csv" \\
      results/optim/MH_01_easy_vi_keyframes.csv \\
      results/optim/MH_01_easy_vi_landmarks.csv
    uv run python scripts/rerun/rerun_vi_chain.py \\
      results/optim/MH_01_easy_vi_keyframes.csv \\
      results/optim/MH_01_easy_vi_landmarks.csv
    rerun results/rerun/MH_01_easy_vi_chain.rrd

Output: results/rerun/<keyframes-stem-without-_keyframes>_chain.rrd (or --output)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from slam_core_tools.viz import GT_X_COLOR, SERIES_X_COLORS, load_csv_rows, set_time_since

_KF_COLS = frozenset({
    "timestamp_s",
    "gt_p_x", "gt_p_y", "gt_p_z",
    "init_p_x", "init_p_y", "init_p_z", "init_pos_err_m",
    "opt_p_x", "opt_p_y", "opt_p_z", "opt_pos_err_m", "opt_rot_err_deg",
    "bg_x", "bg_y", "bg_z", "ba_x", "ba_y", "ba_z",
})
_LM_COLS = frozenset({
    "gt_x", "gt_y", "gt_z", "opt_x", "opt_y", "opt_z", "n_obs", "opt_err_m",
})


def error_colors(errors: np.ndarray, err_lo: float, err_hi: float) -> np.ndarray:
    """Green -> red gradient by landmark error, clamped to [err_lo, err_hi]."""
    t = np.clip((errors - err_lo) / max(err_hi - err_lo, 1e-9), 0.0, 1.0)
    colors = np.zeros((len(errors), 3), dtype=np.uint8)
    colors[:, 0] = (60 + 195 * t).astype(np.uint8)
    colors[:, 1] = (200 - 140 * t).astype(np.uint8)
    colors[:, 2] = 60
    return colors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize the full VI optimization result in Rerun."
    )
    parser.add_argument("keyframes_csv", help="Keyframes CSV from optimize_vi_chain")
    parser.add_argument("landmarks_csv", help="Landmarks CSV from optimize_vi_chain")
    parser.add_argument("--output", metavar="PATH",
                        help="Output .rrd path (default: results/rerun/<stem>_chain.rrd)")
    args = parser.parse_args()

    try:
        kfs = load_csv_rows(Path(args.keyframes_csv), _KF_COLS, "Keyframes")
        lms = load_csv_rows(Path(args.landmarks_csv), _LM_COLS, "Landmarks")
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"Error: {e}")
    if not kfs or not lms:
        sys.exit("Error: empty input CSV")

    stem = Path(args.keyframes_csv).stem.replace("_keyframes", "")
    rrd_path = Path(args.output) if args.output else Path("results/rerun") / f"{stem}_chain.rrd"
    rrd_path.parent.mkdir(parents=True, exist_ok=True)

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(name="Trajectory + Map", origin="vi"),
            rrb.Vertical(
                rrb.TimeSeriesView(name="Position Error (m)", origin="position_error"),
                rrb.TimeSeriesView(name="Gyro Bias (rad/s)",  origin="gyro_bias"),
                rrb.TimeSeriesView(name="Accel Bias (m/s^2)", origin="accel_bias"),
            ),
            column_shares=[3, 2],
        ),
    )

    rr.init(f"slam_core/{stem}", spawn=False)
    rr.save(str(rrd_path), default_blueprint=blueprint)

    start_s = kfs[0]["timestamp_s"]
    p0 = np.array([kfs[0]["gt_p_x"], kfs[0]["gt_p_y"], kfs[0]["gt_p_z"]])

    gt_pts, opt_pts = [], []
    for row in kfs:
        set_time_since(row["timestamp_s"], start_s)
        gt_pts.append((np.array([row["gt_p_x"], row["gt_p_y"], row["gt_p_z"]]) - p0).tolist())
        opt_pts.append((np.array([row["opt_p_x"], row["opt_p_y"], row["opt_p_z"]]) - p0).tolist())

        rr.log("position_error/initialization", rr.Scalars(row["init_pos_err_m"]))
        rr.log("position_error/optimized",      rr.Scalars(row["opt_pos_err_m"]))
        rr.log("gyro_bias/x",  rr.Scalars(row["bg_x"]))
        rr.log("gyro_bias/y",  rr.Scalars(row["bg_y"]))
        rr.log("gyro_bias/z",  rr.Scalars(row["bg_z"]))
        rr.log("accel_bias/x", rr.Scalars(row["ba_x"]))
        rr.log("accel_bias/y", rr.Scalars(row["ba_y"]))
        rr.log("accel_bias/z", rr.Scalars(row["ba_z"]))

    rr.log("vi/ground_truth", rr.LineStrips3D([gt_pts], colors=[GT_X_COLOR], radii=0.01),
           static=True)
    rr.log("vi/optimized", rr.Points3D(opt_pts, colors=SERIES_X_COLORS[1], radii=0.03),
           static=True)

    lm_opt = np.array([[r["opt_x"], r["opt_y"], r["opt_z"]] for r in lms]) - p0
    lm_gt  = np.array([[r["gt_x"], r["gt_y"], r["gt_z"]] for r in lms]) - p0
    errors = np.array([r["opt_err_m"] for r in lms])
    rr.log("vi/landmarks",
           rr.Points3D(lm_opt, colors=error_colors(errors, 0.02, 0.5), radii=0.05),
           static=True)
    rr.log("vi/landmark_residuals",
           rr.LineStrips3D([[lm_opt[i].tolist(), lm_gt[i].tolist()] for i in range(len(lms))],
                           colors=[[150, 150, 150]], radii=0.004),
           static=True)

    print(f"  Keyframes: {len(kfs)}, landmarks: {len(lms)}")
    print(f"  Keyframe pos err (final row): init {kfs[-1]['init_pos_err_m']:.3f} m, "
          f"optimized {kfs[-1]['opt_pos_err_m']:.4f} m")
    print(f"  Landmark err [median/max]: {np.median(errors):.3f} / {errors.max():.3f} m")
    print(f"  Saved: {rrd_path}")


if __name__ == "__main__":
    main()
