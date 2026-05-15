"""
EuRoC debug logger. Reads one sequence and logs stereo images + IMU to Rerun.

Usage:
    uv run python scripts/rerun_euroc_debug.py [config_yaml] [--dataset-root PATH]
                                               [--max-frames N] [--image-stride N]
                                               [--imu-stride N] [--full]

Default config: configs/datasets/euroc_mh01.yaml
Output (debug): results/rerun/<sequence>_debug.rrd   (default)
Output (full):  results/rerun/<sequence>_full.rrd    (--full)

Sampling defaults (debug mode):
    --max-frames 300    cap on stereo pairs logged
    --image-stride 5    log every 5th stereo pair
    --imu-stride 10     log every 10th IMU sample
    --full              disable all sampling and log the entire sequence

Dataset root resolution order:
    1. --dataset-root CLI argument
    2. SLAM_CORE_DATASETS environment variable
    3. dataset_root field in config yaml
    4. Error if none of the above are set
"""

import argparse
import csv
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import yaml


# ---------------------------------------------------------------------------
# Config and path resolution
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_sequence_root(dataset_root_arg: str | None, cfg: dict) -> Path:
    # Priority: CLI arg > env var > config field
    dataset_root = (
        dataset_root_arg
        or os.environ.get("SLAM_CORE_DATASETS")
        or cfg.get("dataset_root")
    )
    if not dataset_root:
        sys.exit(
            "Error: dataset root not found. Provide it via one of:\n"
            "  1. --dataset-root /path/to/datasets\n"
            "  2. export SLAM_CORE_DATASETS=/path/to/datasets\n"
            "  3. dataset_root: /path/to/datasets  (in config yaml)"
        )
    root = Path(dataset_root) / "euroc" / cfg["sequence"]
    if not root.exists():
        sys.exit(
            f"Error: sequence directory not found: {root}\n"
            f"Expected layout: <dataset_root>/euroc/{cfg['sequence']}/"
        )
    return root


# ---------------------------------------------------------------------------
# IMU/Camera reading
# ---------------------------------------------------------------------------

def read_imu_csv(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"Error: IMU CSV not found: {path}")
    rows = []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for line in reader:
            ts_ns, wx, wy, wz, ax, ay, az = (x.strip() for x in line[:7])
            rows.append({
                "timestamp_s": float(ts_ns) * 1e-9,
                "gyro": (float(wx), float(wy), float(wz)),
                "accel": (float(ax), float(ay), float(az)),
            })
    return rows


def validate_imu(rows: list[dict]) -> None:
    errors = 0
    prev_t = None
    for i, r in enumerate(rows):
        vals = [r["timestamp_s"], *r["gyro"], *r["accel"]]
        if not all(math.isfinite(v) for v in vals):
            print(f"  [warn] IMU row {i}: non-finite value")
            errors += 1
        if prev_t is not None and r["timestamp_s"] <= prev_t:
            print(f"  [warn] IMU row {i}: non-monotonic timestamp")
            errors += 1
        prev_t = r["timestamp_s"]
    if errors:
        print(f"  {errors} IMU validation warning(s)")
    else:
        print(f"  IMU: {len(rows)} samples, all valid")



def read_cam_csv(path: Path) -> list[tuple[float, str]]:
    if not path.exists():
        sys.exit(f"Error: camera CSV not found: {path}")
    entries = []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for line in reader:
            ts_ns, filename = line[0].strip(), line[1].strip()
            entries.append((float(ts_ns) * 1e-9, filename))
    return entries


# ---------------------------------------------------------------------------
# Rerun logging
# ---------------------------------------------------------------------------

def set_rerun_time(t_s: float, start_s: float) -> None:
    t_rel_s = t_s - start_s
    rr.set_time("time", duration=t_rel_s)


def log_imu(rows: list[dict], start_s: float) -> None:
    for r in rows:
        gx, gy, gz = r["gyro"]
        ax, ay, az = r["accel"]
        gyro_norm          = math.sqrt(gx**2 + gy**2 + gz**2)
        accel_norm         = math.sqrt(ax**2 + ay**2 + az**2)
        accel_norm_minus_g = accel_norm - 9.81
        set_rerun_time(r["timestamp_s"], start_s)
        rr.log("imu/gyro/x",                        rr.Scalars(gx))
        rr.log("imu/gyro/y",                        rr.Scalars(gy))
        rr.log("imu/gyro/z",                        rr.Scalars(gz))
        rr.log("imu/gyro/norm_radps",               rr.Scalars(gyro_norm))
        rr.log("imu/accel/x",                       rr.Scalars(ax))
        rr.log("imu/accel/y",                       rr.Scalars(ay))
        rr.log("imu/accel/z",                       rr.Scalars(az))
        rr.log("imu/accel/norm_mps2",               rr.Scalars(accel_norm))
        rr.log("imu/accel/norm_minus_gravity_mps2",  rr.Scalars(accel_norm_minus_g))


def log_stereo_images(
    cam0: list[tuple[float, str]],
    cam1: list[tuple[float, str]],
    cam0_dir: Path,
    cam1_dir: Path,
    start_s: float,
    stride: int = 1,
    max_frames: int | None = None,
) -> int:
    cam1_by_ts = {ts: fn for ts, fn in cam1}
    logged = 0
    for i, (ts, fn0) in enumerate(cam0):
        if i % stride != 0:
            continue
        if max_frames is not None and logged >= max_frames:
            break
        fn1 = cam1_by_ts.get(ts)
        if fn1 is None:
            continue
        img0 = cv2.imread(str(cam0_dir / fn0), cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(str(cam1_dir / fn1), cv2.IMREAD_GRAYSCALE)
        if img0 is None or img1 is None:
            continue
        set_rerun_time(ts, start_s)
        rr.log("camera/cam0", rr.Image(img0))
        rr.log("camera/cam1", rr.Image(img1))
        logged += 1
    return logged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Log a EuRoC sequence to Rerun.")
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/datasets/euroc_mh01.yaml",
        help="Path to sequence config yaml (default: configs/datasets/euroc_mh01.yaml)",
    )
    parser.add_argument(
        "--dataset-root",
        metavar="PATH",
        help="Parent directory containing euroc/<sequence>/. "
             "Overrides SLAM_CORE_DATASETS and config dataset_root.",
    )
    parser.add_argument(
        "--max-frames", type=int, default=300, metavar="N",
        help="Max stereo pairs to log in debug mode (default: 300).",
    )
    parser.add_argument(
        "--image-stride", type=int, default=5, metavar="N",
        help="Log every Nth stereo pair in debug mode (default: 5).",
    )
    parser.add_argument(
        "--imu-stride", type=int, default=10, metavar="N",
        help="Log every Nth IMU sample in debug mode (default: 10).",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Disable all sampling and log the entire sequence.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"Error: config not found: {config_path}")

    cfg = load_config(config_path)
    seq_root = resolve_sequence_root(args.dataset_root, cfg)
    sequence = cfg["sequence"]

    print(f"Sequence: {sequence}")
    print(f"Root:     {seq_root}")

    # Read data
    imu_rows = read_imu_csv(seq_root / cfg["imu_csv"])
    validate_imu(imu_rows)

    cam0_entries = read_cam_csv(seq_root / cfg["cam0_csv"])
    cam1_entries = read_cam_csv(seq_root / cfg["cam1_csv"])
    cam0_dir = seq_root / cfg["cam0_images"]
    cam1_dir = seq_root / cfg["cam1_images"]

    # Sampling parameters
    if args.full:
        imu_to_log = imu_rows
        image_stride, max_frames = 1, None
        mode = "full"
    else:
        imu_to_log = imu_rows[::args.imu_stride]
        image_stride, max_frames = args.image_stride, args.max_frames
        mode = "debug"

    # Output path
    output_dir = Path("results/rerun")
    output_dir.mkdir(parents=True, exist_ok=True)
    rrd_path = output_dir / f"{sequence}_{mode}.rrd"

    # Initialise Rerun
    rr.init(f"slam_core/{sequence}", spawn=False)
    rr.save(str(rrd_path))

    # Sequence-relative time origin from first IMU sample
    start_s = imu_rows[0]["timestamp_s"] if imu_rows else 0.0

    # Log
    log_imu(imu_to_log, start_s)
    n_stereo = log_stereo_images(
        cam0_entries, cam1_entries, cam0_dir, cam1_dir, start_s,
        stride=image_stride, max_frames=max_frames,
    )

    # Summary
    if imu_rows:
        duration = imu_rows[-1]["timestamp_s"] - imu_rows[0]["timestamp_s"]
        print(f"  Duration:  {duration:.2f} s")
    print(f"  IMU:       {len(imu_rows):6d} read,  {len(imu_to_log):6d} logged")
    print(f"  Stereo:    {len(cam0_entries):6d} read,  {n_stereo:6d} logged")
    print(f"  Saved: {rrd_path}")


if __name__ == "__main__":
    main()
