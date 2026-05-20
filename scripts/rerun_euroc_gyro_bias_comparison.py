"""
Compare raw vs bias-corrected gyro-only propagation against EuRoC GT in Rerun.

Two separate 3D views, one error plot:
  raw_comparison/ground_truth/body_frame    — blue  x-axis  (Raw Gyro vs GT panel)
  raw_comparison/raw_gyro/body_frame        — red   x-axis
  corrected_comparison/ground_truth/body_frame  — blue  x-axis  (Bias-Corrected vs GT panel)
  corrected_comparison/bias_corrected/body_frame — green x-axis

Scalar error channels:
  orientation_error/raw_geodesic_deg
  orientation_error/bias_corrected_geodesic_deg

Both estimated series share the same reference pair (first raw row) for
relative-orientation comparison — identical convention to
scripts/rerun_euroc_gyro_propagation.py.

Usage:
    uv run python scripts/rerun_euroc_gyro_bias_comparison.py \\
      configs/datasets/euroc_mh01.yaml \\
      results/imu/MH_01_easy_gyro_orientations.csv \\
      results/imu/MH_01_easy_gyro_orientations_bias_corrected.csv \\
      --dataset-root "$HOME/datasets" \\
      --frame-stride 5 \\
      --max-duration-s 180 \\
      --output results/rerun/MH_01_easy_gyro_bias_comparison.rrd
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
    GroundTruthSample,
    load_config,
    read_groundtruth_csv,
    resolve_sequence_root,
)

_EXPECTED_COLS = {
    "timestamp_s", "r00", "r01", "r02",
    "r10", "r11", "r12",
    "r20", "r21", "r22",
}

_MAIN_LEN = 1.0
_AUX_LEN  = 0.45

_GT_X_COLOR    = [  0,   0, 255]
_GT_AUX_COLOR  = [ 90,  90,  90]
_RAW_X_COLOR   = [220,  60,   0]
_RAW_AUX_COLOR = [170, 170, 170]
_COR_X_COLOR   = [  0, 180,  80]
_COR_AUX_COLOR = [170, 170, 170]


# ---------------------------------------------------------------------------
# Helpers — same convention as rerun_euroc_gyro_propagation.py
# ---------------------------------------------------------------------------

def _quat_to_mat(w: float, x: float, y: float, z: float) -> np.ndarray:
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x)],
        [2*(x*z - w*y),      2*(y*z + w*x),       1 - 2*(x*x + y*y)],
    ])


def _build_gt_index(
    samples: list[GroundTruthSample],
) -> tuple[list[float], list[np.ndarray]]:
    times: list[float] = []
    mats:  list[np.ndarray] = []
    for s in samples:
        norm = math.sqrt(s.q_w**2 + s.q_x**2 + s.q_y**2 + s.q_z**2)
        if norm < 1e-10:
            continue
        w, x, y, z = s.q_w/norm, s.q_x/norm, s.q_y/norm, s.q_z/norm
        times.append(s.timestamp_s)
        mats.append(_quat_to_mat(w, x, y, z))
    return times, mats


def _nearest_gt_mat(
    ts: float,
    gt_times: list[float],
    gt_mats: list[np.ndarray],
) -> np.ndarray:
    idx = bisect.bisect_left(gt_times, ts)
    if idx == 0:
        return gt_mats[0]
    if idx == len(gt_times):
        return gt_mats[-1]
    if ts - gt_times[idx - 1] <= gt_times[idx] - ts:
        return gt_mats[idx - 1]
    return gt_mats[idx]


def _geodesic_deg(R_err: np.ndarray) -> float:
    cos_angle = (np.trace(R_err) - 1.0) / 2.0
    return math.acos(max(-1.0, min(1.0, float(cos_angle)))) * 180.0 / math.pi


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def _load_orientations(csv_path: Path) -> list[tuple[float, list[float]]]:
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
# Rerun helpers
# ---------------------------------------------------------------------------

def _log_frame(
    entity: str,
    R_rel: np.ndarray,
    x_color: list[int],
    aux_color: list[int],
) -> None:
    rr.log(
        entity,
        rr.Arrows3D(
            origins=[[0, 0, 0]] * 3,
            vectors=[
                R_rel[:, 0] * _MAIN_LEN,
                R_rel[:, 1] * _AUX_LEN,
                R_rel[:, 2] * _AUX_LEN,
            ],
            colors=[x_color, aux_color, aux_color],
        ),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare raw vs bias-corrected gyro propagation in Rerun."
    )
    parser.add_argument(
        "config",
        help="EuRoC sequence config YAML",
    )
    parser.add_argument(
        "raw_csv",
        help="Raw orientation CSV from export_gyro_propagation",
    )
    parser.add_argument(
        "corrected_csv",
        help="Bias-corrected orientation CSV from export_gyro_propagation --gyro-bias",
    )
    parser.add_argument(
        "--dataset-root", metavar="PATH",
        help="Parent directory containing euroc/<sequence>/",
    )
    parser.add_argument(
        "--frame-stride", type=int, default=5, metavar="N",
        help="Log every Nth orientation row (default: 5)",
    )
    parser.add_argument(
        "--max-duration-s", type=float, default=None, metavar="FLOAT",
        help="Truncate at this relative time in seconds (default: full)",
    )
    parser.add_argument(
        "--output", metavar="PATH",
        help="Output .rrd path (default: results/rerun/<sequence>_gyro_bias_comparison.rrd)",
    )
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
    gt_times, gt_mats = _build_gt_index(gt_samples)

    try:
        raw_rows  = _load_orientations(Path(args.raw_csv))
        corr_rows = _load_orientations(Path(args.corrected_csv))
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"Error: {e}")

    if not raw_rows or not corr_rows:
        sys.exit("Error: one or both orientation CSVs are empty")

    if len(raw_rows) != len(corr_rows):
        sys.exit(
            f"Error: row count mismatch — raw {len(raw_rows)}, "
            f"corrected {len(corr_rows)}. Both CSVs must come from the same IMU input."
        )

    if args.output:
        rrd_path = Path(args.output)
    else:
        rrd_path = Path("results/rerun") / f"{sequence}_gyro_bias_comparison.rrd"
    rrd_path.parent.mkdir(parents=True, exist_ok=True)

    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(name="Raw Gyro vs GT",       origin="raw_comparison"),
                rrb.Spatial3DView(name="Bias-Corrected vs GT", origin="corrected_comparison"),
            ),
            rrb.TimeSeriesView(name="Orientation Error", origin="orientation_error"),
            row_shares=[3, 2],
        ),
    )

    rr.init(f"slam_core/{sequence}/gyro_bias_comparison", spawn=False)
    rr.save(str(rrd_path), default_blueprint=blueprint)

    # Reference pair: first raw row matched to nearest GT.
    # Both series use this pair so their relative errors are comparable.
    first_ts, first_mat = raw_rows[0]
    R_est_ref = np.array(first_mat).reshape(3, 3)
    R_gt_ref  = _nearest_gt_mat(first_ts, gt_times, gt_mats)
    start_s   = first_ts

    raw_errors:  list[float] = []
    corr_errors: list[float] = []
    n_logged = 0

    for i, ((ts_raw, mat_raw), (_, mat_cor)) in enumerate(zip(raw_rows, corr_rows)):
        t_rel = ts_raw - start_s
        if args.max_duration_s is not None and t_rel > args.max_duration_s:
            break

        R_raw = np.array(mat_raw).reshape(3, 3)
        R_cor = np.array(mat_cor).reshape(3, 3)
        R_gt  = _nearest_gt_mat(ts_raw, gt_times, gt_mats)

        R_raw_rel = R_est_ref.T @ R_raw
        R_cor_rel = R_est_ref.T @ R_cor
        R_gt_rel  = R_gt_ref.T  @ R_gt

        err_raw  = _geodesic_deg(R_gt_rel.T @ R_raw_rel)
        err_corr = _geodesic_deg(R_gt_rel.T @ R_cor_rel)

        raw_errors.append(err_raw)
        corr_errors.append(err_corr)

        rr.set_time("time", duration=t_rel)

        rr.log("orientation_error/raw_geodesic_deg",            rr.Scalars(err_raw))
        rr.log("orientation_error/bias_corrected_geodesic_deg", rr.Scalars(err_corr))

        if i % args.frame_stride == 0:
            _log_frame("raw_comparison/ground_truth/body_frame",        R_gt_rel,  _GT_X_COLOR,  _GT_AUX_COLOR)
            _log_frame("raw_comparison/raw_gyro/body_frame",            R_raw_rel, _RAW_X_COLOR, _RAW_AUX_COLOR)
            _log_frame("corrected_comparison/ground_truth/body_frame",  R_gt_rel,  _GT_X_COLOR,  _GT_AUX_COLOR)
            _log_frame("corrected_comparison/bias_corrected/body_frame", R_cor_rel, _COR_X_COLOR, _COR_AUX_COLOR)
            n_logged += 1

    if not raw_errors:
        sys.exit("Error: no rows within the requested duration")

    mean_raw  = sum(raw_errors)  / len(raw_errors)
    mean_corr = sum(corr_errors) / len(corr_errors)

    print(f"  Sequence:                     {sequence}")
    print(f"  Scalar rows logged:           {len(raw_errors)}")
    print(f"  Body frames logged:           {n_logged}  (stride {args.frame_stride})")
    if args.max_duration_s is not None:
        print(f"  Max duration:                 {args.max_duration_s} s")
    print(f"  Raw   error [mean/final/max]: "
          f"{mean_raw:.2f} / {raw_errors[-1]:.2f} / {max(raw_errors):.2f} deg")
    print(f"  Corr  error [mean/final/max]: "
          f"{mean_corr:.2f} / {corr_errors[-1]:.2f} / {max(corr_errors):.2f} deg")
    print(f"  Saved: {rrd_path}")


if __name__ == "__main__":
    main()
