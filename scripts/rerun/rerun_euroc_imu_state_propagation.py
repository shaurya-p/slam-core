"""
Visualize full IMU state propagation in Rerun.

WARNING: no visual correction and no online bias estimation. Position and
velocity drift is expected from double-integrated accelerometer noise.

Takes one or more state CSVs exported by export_imu_state_propagation
(C++ source of truth; propagation is NOT recomputed here). With multiple
CSVs — e.g. raw vs bias-corrected — each series gets its own trajectory
and per-series error channels against GT.

State CSV columns:
  timestamp_s, p_x/y/z, v_x/y/z, q_w/x/y/z, r00..r22 (row-major R_W_B),
  gyro_bias_x/y/z, accel_bias_x/y/z

GT comparison requires --config (and is mandatory with multiple CSVs).
All 3D entities are displayed relative to the first series' first position.

Usage (CSV-only):
    uv run python scripts/rerun/rerun_euroc_imu_state_propagation.py \\
      results/imu/MH_01_easy_imu_state.csv

Usage (raw vs bias-corrected comparison against GT):
    uv run python scripts/rerun/rerun_euroc_imu_state_propagation.py \\
      results/imu/MH_01_easy_imu_state.csv \\
      results/imu/MH_01_easy_imu_state_bias_corrected.csv \\
      --labels raw bias_corrected \\
      --config configs/datasets/euroc_mh01.yaml \\
      --dataset-root "$HOME/datasets" --max-duration-s 180

Output: results/rerun/<first-csv-stem>.rrd (or --output)
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from slam_core_tools.viz import (
    EST_AUX_COLOR,
    GT_AUX_COLOR,
    GT_X_COLOR,
    SERIES_X_COLORS,
    geodesic_deg,
    load_sequence_gt,
    load_state_csv,
    log_axis_triad,
    nearest_by_index,
    quat_to_mat,
    set_time_since,
)
from slam_core_tools.viz.common import mat_from_row

_DRIFT_WARNING = (
    "WARNING: no visual correction, no online bias estimation. "
    "Position and velocity drift is expected."
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize full IMU state propagation in Rerun."
    )
    parser.add_argument(
        "state_csvs", nargs="+", metavar="state_csv",
        help="State CSV(s) exported by export_imu_state_propagation",
    )
    parser.add_argument(
        "--labels", nargs="+", metavar="LABEL",
        help="One label per CSV for entity paths and legends (default: CSV stems)",
    )
    parser.add_argument(
        "--config", metavar="PATH",
        help="EuRoC sequence config YAML; enables GT comparison "
             "(required with multiple CSVs)",
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
        help="Output .rrd path (default: results/rerun/<first-csv-stem>.rrd)",
    )
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.state_csvs):
        sys.exit("Error: --labels count must match the number of state CSVs")
    multi = len(args.state_csvs) > 1
    if multi and not args.config:
        sys.exit("Error: --config is required when comparing multiple state CSVs")
    labels = args.labels or (
        [Path(p).stem for p in args.state_csvs] if multi else ["estimated"]
    )
    if len(set(labels)) != len(labels):
        sys.exit("Error: series labels must be unique")

    print(_DRIFT_WARNING)

    series: list[list[dict]] = []
    for csv_arg in args.state_csvs:
        try:
            rows = load_state_csv(Path(csv_arg))
        except (FileNotFoundError, ValueError) as e:
            sys.exit(f"Error: {e}")
        if not rows:
            sys.exit(f"Error: state CSV is empty: {csv_arg}")
        print(f"  State rows: {len(rows)}  ({csv_arg})")
        series.append(rows)

    n_rows = len(series[0])
    if any(len(s) != n_rows for s in series):
        sys.exit(
            "Error: row count mismatch across state CSVs. "
            "All CSVs must come from the same IMU input and init mode."
        )

    gt_samples = None
    gt_times: list[float] = []
    if args.config:
        _, _, gt_samples = load_sequence_gt(Path(args.config), args.dataset_root)
        print(f"  GT samples:  {len(gt_samples)}")
        gt_times = [s.timestamp_s for s in gt_samples]

    first_stem = Path(args.state_csvs[0]).stem
    if args.output:
        rrd_path = Path(args.output)
    else:
        rrd_path = Path("results/rerun") / f"{first_stem}.rrd"
    rrd_path.parent.mkdir(parents=True, exist_ok=True)

    blueprint = None
    if gt_samples:
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(name="Trajectories", origin="/"),
                rrb.Vertical(
                    rrb.TimeSeriesView(name="Position Error (m)",      origin="position_error"),
                    rrb.TimeSeriesView(name="Velocity Error (m/s)",    origin="velocity_error"),
                    rrb.TimeSeriesView(name="Orientation Error (deg)", origin="orientation_error"),
                ),
                column_shares=[3, 2],
            ),
        )

    rr.init(f"slam_core/{first_stem}", spawn=False)
    if blueprint is not None:
        rr.save(str(rrd_path), default_blueprint=blueprint)
    else:
        rr.save(str(rrd_path))

    start_s = series[0][0]["timestamp_s"]

    # Display origin: shift all 3D entities so the first series starts at the
    # world origin. Scalar/error plots use raw values.
    p0 = np.array([series[0][0]["p_x"], series[0][0]["p_y"], series[0][0]["p_z"]])

    positions: list[list[list[float]]] = [[] for _ in labels]
    n_scalars = 0
    n_frames  = 0

    for i in range(n_rows):
        ts    = series[0][i]["timestamp_s"]
        t_rel = ts - start_s
        if args.max_duration_s is not None and t_rel > args.max_duration_s:
            break

        set_time_since(ts, start_s)
        log_frames = i % args.frame_stride == 0

        gt = None
        if gt_samples:
            gt = nearest_by_index(ts, gt_samples, gt_times)
            q_norm = math.sqrt(gt.q_w**2 + gt.q_x**2 + gt.q_y**2 + gt.q_z**2)
            if q_norm < 1e-10:
                continue
            R_gt = quat_to_mat(gt.q_w/q_norm, gt.q_x/q_norm,
                               gt.q_y/q_norm, gt.q_z/q_norm)
            gt_p = np.array([gt.p_x, gt.p_y, gt.p_z])
            gt_v = np.array([gt.v_x, gt.v_y, gt.v_z])
            rr.log("ground_truth/velocity_norm_mps",
                   rr.Scalars(float(np.linalg.norm(gt_v))))
            if log_frames:
                log_axis_triad("ground_truth/body_frame", R_gt,
                               GT_X_COLOR, GT_AUX_COLOR,
                               length=args.axis_length,
                               position=(gt_p - p0).tolist())

        for k, label in enumerate(labels):
            row   = series[k][i]
            p     = np.array([row["p_x"], row["p_y"], row["p_z"]])
            v     = np.array([row["v_x"], row["v_y"], row["v_z"]])
            R_est = mat_from_row(row)
            p_disp = p - p0
            positions[k].append(p_disp.tolist())

            if not multi:
                # Raw state channels (single-series mode only)
                rr.log("imu_state/position_x_m",      rr.Scalars(p[0]))
                rr.log("imu_state/position_y_m",      rr.Scalars(p[1]))
                rr.log("imu_state/position_z_m",      rr.Scalars(p[2]))
                rr.log("imu_state/velocity_x_mps",    rr.Scalars(v[0]))
                rr.log("imu_state/velocity_y_mps",    rr.Scalars(v[1]))
                rr.log("imu_state/velocity_z_mps",    rr.Scalars(v[2]))
                rr.log("imu_state/position_norm_m",   rr.Scalars(float(np.linalg.norm(p))))
                rr.log("imu_state/velocity_norm_mps", rr.Scalars(float(np.linalg.norm(v))))

            if log_frames:
                x_color = SERIES_X_COLORS[k % len(SERIES_X_COLORS)]
                log_axis_triad(f"{label}/body_frame", R_est, x_color,
                               EST_AUX_COLOR, length=args.axis_length,
                               position=p_disp.tolist())

            if gt is not None:
                p_err = p - gt_p
                rr.log(f"position_error/{label}_norm_m",
                       rr.Scalars(float(np.linalg.norm(p_err))))
                rr.log(f"velocity_error/{label}_norm_mps",
                       rr.Scalars(float(np.linalg.norm(v - gt_v))))
                rr.log(f"orientation_error/{label}_geodesic_deg",
                       rr.Scalars(geodesic_deg(R_gt.T @ R_est)))
                if not multi:
                    rr.log("position_error/x_m", rr.Scalars(p_err[0]))
                    rr.log("position_error/y_m", rr.Scalars(p_err[1]))
                    rr.log("position_error/z_m", rr.Scalars(p_err[2]))

        n_scalars += 1
        if log_frames:
            n_frames += 1

    # Static trajectory strips (logged after the time-series loop)
    for label, pts in zip(labels, positions):
        if len(pts) >= 2:
            rr.log(f"{label}/trajectory", rr.LineStrips3D([pts]), static=True)

    if gt_samples:
        gt_positions = [
            [s.p_x - p0[0], s.p_y - p0[1], s.p_z - p0[2]] for s in gt_samples
            if args.max_duration_s is None or s.timestamp_s - start_s <= args.max_duration_s
        ]
        if len(gt_positions) >= 2:
            rr.log("ground_truth/trajectory",
                   rr.LineStrips3D([gt_positions]), static=True)

    print(f"  Scalar rows logged: {n_scalars}")
    print(f"  Body frames logged: {n_frames}  (stride {args.frame_stride})")
    if args.max_duration_s is not None:
        print(f"  Max duration:       {args.max_duration_s} s")
    if gt_samples is not None:
        print(f"  GT mode:            on")
    print(f"  Saved: {rrd_path}")


if __name__ == "__main__":
    main()
