"""
Visualize gyro bias evaluation results in Rerun.

Reads the CSV produced by evaluate_gyro_bias_from_gt (C++ tool).

CSV columns required:
  timestamp_start_s, timestamp_end_s,
  gyro_bias_est_x_radps, gyro_bias_est_y_radps, gyro_bias_est_z_radps,
  gyro_bias_est_norm_radps, error_angle_deg

Each row represents one non-overlapping window. Logs one scalar point per window.

Sign convention (from evaluate_gyro_bias_from_gt):
  gyro_bias_est = -log_so3(R_err) / integrated_dt
  omega_meas = omega_true + bias

Usage:
    uv run python scripts/rerun/rerun_euroc_gyro_bias_eval.py \\
      results/tmp/MH_01_easy_gyro_bias_1s.csv \\
      --output results/rerun/MH_01_easy_gyro_bias_1s.rrd
"""

import argparse
import csv
import sys
from pathlib import Path

import rerun as rr

_EXPECTED_COLS = {
    "timestamp_start_s",
    "gyro_bias_est_x_radps",
    "gyro_bias_est_y_radps",
    "gyro_bias_est_z_radps",
    "gyro_bias_est_norm_radps",
    "error_angle_deg",
}


def load_bias_csv(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Bias eval CSV not found: {csv_path}")
    rows: list[dict] = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not _EXPECTED_COLS.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"Bias eval CSV missing required columns.\n"
                f"Expected subset: {sorted(_EXPECTED_COLS)}\n"
                f"Got: {reader.fieldnames}"
            )
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize gyro bias evaluation results in Rerun.",
    )
    parser.add_argument(
        "bias_eval_csv",
        help="CSV produced by evaluate_gyro_bias_from_gt",
    )
    parser.add_argument(
        "--output", metavar="PATH",
        help="Output .rrd path (default: results/rerun/<csv-stem>.rrd)",
    )
    parser.add_argument(
        "--max-duration-s", type=float, default=None, metavar="FLOAT",
        help="Only log rows where t_rel <= this value (default: full CSV)",
    )
    args = parser.parse_args()

    csv_path = Path(args.bias_eval_csv)
    try:
        rows = load_bias_csv(csv_path)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"Error: {e}")

    if not rows:
        sys.exit("Error: bias eval CSV is empty")

    if args.output:
        rrd_path = Path(args.output)
    else:
        rrd_path = Path("results/rerun") / f"{csv_path.stem}.rrd"
    rrd_path.parent.mkdir(parents=True, exist_ok=True)

    rr.init(f"slam_core/{csv_path.stem}", spawn=False)
    rr.save(str(rrd_path))

    start_s = rows[0]["timestamp_start_s"]

    logged_norms: list[float] = []
    logged_errors: list[float] = []

    for row in rows:
        t_rel = row["timestamp_start_s"] - start_s
        if args.max_duration_s is not None and t_rel > args.max_duration_s:
            break

        rr.set_time("time", duration=t_rel)

        rr.log("gyro_bias/x_radps",    rr.Scalars(row["gyro_bias_est_x_radps"]))
        rr.log("gyro_bias/y_radps",    rr.Scalars(row["gyro_bias_est_y_radps"]))
        rr.log("gyro_bias/z_radps",    rr.Scalars(row["gyro_bias_est_z_radps"]))
        rr.log("gyro_bias/norm_radps", rr.Scalars(row["gyro_bias_est_norm_radps"]))
        rr.log("gyro_bias/zero_radps", rr.Scalars(0.0))
        rr.log("orientation_error/error_angle_deg", rr.Scalars(row["error_angle_deg"]))

        logged_norms.append(row["gyro_bias_est_norm_radps"])
        logged_errors.append(row["error_angle_deg"])

    n = len(logged_norms)
    if n == 0:
        print("Warning: no rows logged (max-duration-s too short?)")
        return

    duration_s = rows[n - 1]["timestamp_start_s"] - start_s

    print(f"  Rows logged:                {n}")
    print(f"  Duration covered:           {duration_s:.1f} s")
    print(f"  Bias norm  [min/mean/max]:  "
          f"{min(logged_norms):.4f} / {sum(logged_norms)/n:.4f} / {max(logged_norms):.4f}  rad/s")
    print(f"  Error angle [min/mean/max]: "
          f"{min(logged_errors):.2f} / {sum(logged_errors)/n:.2f} / {max(logged_errors):.2f}  deg")
    print(f"  Saved: {rrd_path}")


if __name__ == "__main__":
    main()
