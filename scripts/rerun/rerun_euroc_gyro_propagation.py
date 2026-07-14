"""
Visualize gyro-only SO(3) orientation propagation against EuRoC GT in Rerun.

WARNING: gyro-only propagation has no gravity alignment, no accelerometer
fusion, and no visual correction. Drift is expected.

Takes one or more orientation CSVs exported by export_gyro_propagation
(C++ source of truth; propagation is NOT recomputed here). With multiple
CSVs — e.g. raw vs bias-corrected — each series gets its own 3D panel
against GT plus a shared error plot.

Orientation CSV columns (row-major R_W_B entries):
  timestamp_s,r00,r01,r02,r10,r11,r12,r20,r21,r22

R_W_B maps body frame B into world frame W.
All series are compared relative to the reference pair (first row of the
first CSV matched to nearest GT), so estimated and GT frames coincide at
the reference timestamp.

Usage (single series):
    uv run python scripts/rerun/rerun_euroc_gyro_propagation.py \\
      configs/datasets/euroc_mh01.yaml \\
      results/imu/MH_01_easy_gyro_orientations.csv \\
      --dataset-root "$HOME/datasets" --frame-stride 20

Usage (raw vs bias-corrected comparison):
    uv run python scripts/rerun/rerun_euroc_gyro_propagation.py \\
      configs/datasets/euroc_mh01.yaml \\
      results/imu/MH_01_easy_gyro_orientations.csv \\
      results/imu/MH_01_easy_gyro_orientations_bias_corrected.csv \\
      --labels raw bias_corrected \\
      --dataset-root "$HOME/datasets" --frame-stride 5 --max-duration-s 180

Output: results/rerun/<sequence>_gyro_propagation.rrd (or --output)
"""

import argparse
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
    build_gt_index,
    geodesic_deg,
    load_orientation_csv,
    load_sequence_gt,
    nearest_by_index,
    set_time_since,
    log_axis_triad,
)

_DRIFT_WARNING = (
    "WARNING: gyro-only propagation has no gravity alignment, no accelerometer "
    "fusion, and no visual correction. Drift is expected."
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize gyro-only orientation propagation in Rerun."
    )
    parser.add_argument(
        "config",
        help="EuRoC sequence config YAML (e.g. configs/datasets/euroc_mh01.yaml)",
    )
    parser.add_argument(
        "orientation_csvs", nargs="+", metavar="orientation_csv",
        help="Orientation CSV(s) from export_gyro_propagation",
    )
    parser.add_argument(
        "--labels", nargs="+", metavar="LABEL",
        help="One label per CSV for entity paths and legends (default: CSV stems)",
    )
    parser.add_argument(
        "--dataset-root", metavar="PATH",
        help="Parent directory containing euroc/<sequence>/",
    )
    parser.add_argument(
        "--frame-stride", type=int, default=50, metavar="N",
        help="Log body-frame axes every Nth orientation row (default: 50)",
    )
    parser.add_argument(
        "--max-duration-s", type=float, default=None, metavar="FLOAT",
        help="Truncate at this relative time in seconds (default: full sequence)",
    )
    parser.add_argument(
        "--output", metavar="PATH",
        help="Output .rrd path (default: results/rerun/<sequence>_gyro_propagation.rrd)",
    )
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.orientation_csvs):
        sys.exit("Error: --labels count must match the number of orientation CSVs")
    labels = args.labels or [Path(p).stem for p in args.orientation_csvs]
    if len(set(labels)) != len(labels):
        sys.exit("Error: series labels must be unique")

    print(_DRIFT_WARNING)

    cfg, seq_root, gt_samples = load_sequence_gt(Path(args.config), args.dataset_root)
    sequence = cfg["sequence"]
    print(f"Sequence: {sequence}")
    print(f"Root:     {seq_root}")
    print(f"  GT: {len(gt_samples)} samples")
    gt_times, gt_mats = build_gt_index(gt_samples)

    series: list[list[tuple[float, np.ndarray]]] = []
    for csv_arg in args.orientation_csvs:
        try:
            rows = load_orientation_csv(Path(csv_arg))
        except (FileNotFoundError, ValueError) as e:
            sys.exit(f"Error: {e}")
        if not rows:
            sys.exit(f"Error: orientation CSV is empty: {csv_arg}")
        print(f"  Orientations: {len(rows)} rows  ({csv_arg})")
        series.append(rows)

    n_rows = len(series[0])
    if any(len(s) != n_rows for s in series):
        sys.exit(
            "Error: row count mismatch across orientation CSVs. "
            "All CSVs must come from the same IMU input."
        )

    if args.output:
        rrd_path = Path(args.output)
    else:
        rrd_path = Path("results/rerun") / f"{sequence}_gyro_propagation.rrd"
    rrd_path.parent.mkdir(parents=True, exist_ok=True)

    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(*[
                rrb.Spatial3DView(name=f"{label} vs GT", origin=label)
                for label in labels
            ]),
            rrb.TimeSeriesView(name="Orientation Error (deg)", origin="orientation_error"),
            row_shares=[3, 2],
        ),
    )

    rr.init(f"slam_core/{sequence}/gyro_propagation", spawn=False)
    rr.save(str(rrd_path), default_blueprint=blueprint)

    # Reference pair: first row of the first series matched to nearest GT.
    # All series share it so their relative errors are comparable.
    first_ts, R_est_ref = series[0][0]
    R_gt_ref = nearest_by_index(first_ts, gt_mats, gt_times)
    start_s  = first_ts

    errors: list[list[float]] = [[] for _ in labels]
    n_frames = 0

    for i in range(n_rows):
        ts = series[0][i][0]
        t_rel = ts - start_s
        if args.max_duration_s is not None and t_rel > args.max_duration_s:
            break

        R_gt     = nearest_by_index(ts, gt_mats, gt_times)
        R_gt_rel = R_gt_ref.T @ R_gt

        set_time_since(ts, start_s)

        log_frames = i % args.frame_stride == 0
        for k, label in enumerate(labels):
            R_est_rel = R_est_ref.T @ series[k][i][1]
            err_deg   = geodesic_deg(R_gt_rel.T @ R_est_rel)
            errors[k].append(err_deg)
            rr.log(f"orientation_error/{label}_geodesic_deg", rr.Scalars(err_deg))

            if log_frames:
                x_color = SERIES_X_COLORS[k % len(SERIES_X_COLORS)]
                log_axis_triad(f"{label}/body_frame",    R_est_rel, x_color,    EST_AUX_COLOR)
                log_axis_triad(f"{label}/body_frame_gt", R_gt_rel,  GT_X_COLOR, GT_AUX_COLOR)
        if log_frames:
            n_frames += 1

    if not errors[0]:
        sys.exit("Error: no rows within the requested duration")

    print(f"  Scalar rows logged: {len(errors[0])}")
    print(f"  Body frames logged: {n_frames}  (stride {args.frame_stride})")
    if args.max_duration_s is not None:
        print(f"  Max duration:       {args.max_duration_s} s")
    for label, errs in zip(labels, errors):
        print(f"  {label} error [mean/final/max]: "
              f"{sum(errs)/len(errs):.2f} / {errs[-1]:.2f} / {max(errs):.2f} deg")
    print(f"  Saved: {rrd_path}")


if __name__ == "__main__":
    main()
