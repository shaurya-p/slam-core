"""
EuRoC debug logger. Reads one sequence and logs stereo images + IMU to Rerun.

Usage:
    uv run python scripts/rerun_euroc_debug.py [config_yaml]

Default config: configs/datasets/euroc_mh01.yaml
Output:         results/rerun/<sequence>.rrd
"""

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


def resolve_sequence_root(cfg: dict) -> Path:
    env_key = "SLAM_CORE_DATASETS"
    datasets_root = os.environ.get(env_key)
    if not datasets_root:
        sys.exit(
            f"Error: environment variable {env_key} is not set.\n"
            f"Set it to your datasets directory, e.g.:\n"
            f"  export {env_key}=/path/to/datasets"
        )
    root = Path(datasets_root) / "euroc" / cfg["sequence"]
    if not root.exists():
        sys.exit(
            f"Error: sequence directory not found: {root}\n"
            f"Expected layout: ${env_key}/euroc/{cfg['sequence']}/"
        )
    return root


# ---------------------------------------------------------------------------
# IMU reading and validation
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


# ---------------------------------------------------------------------------
# Camera CSV reading
# ---------------------------------------------------------------------------

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

def log_imu(rows: list[dict]) -> None:
    for r in rows:
        t = r["timestamp_s"]
        gx, gy, gz = r["gyro"]
        ax, ay, az = r["accel"]
        rr.set_time_seconds("timestamp", t)
        rr.log("imu/gyro/x",  rr.Scalar(gx))
        rr.log("imu/gyro/y",  rr.Scalar(gy))
        rr.log("imu/gyro/z",  rr.Scalar(gz))
        rr.log("imu/accel/x", rr.Scalar(ax))
        rr.log("imu/accel/y", rr.Scalar(ay))
        rr.log("imu/accel/z", rr.Scalar(az))


def log_stereo_images(
    cam0: list[tuple[float, str]],
    cam1: list[tuple[float, str]],
    cam0_dir: Path,
    cam1_dir: Path,
) -> int:
    # Build lookup from timestamp -> filename for cam1
    cam1_by_ts = {ts: fn for ts, fn in cam1}
    logged = 0
    for ts, fn0 in cam0:
        fn1 = cam1_by_ts.get(ts)
        if fn1 is None:
            continue
        img0 = cv2.imread(str(cam0_dir / fn0), cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(str(cam1_dir / fn1), cv2.IMREAD_GRAYSCALE)
        if img0 is None or img1 is None:
            continue
        rr.set_time_seconds("timestamp", ts)
        rr.log("camera/cam0", rr.Image(img0))
        rr.log("camera/cam1", rr.Image(img1))
        logged += 1
    return logged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("configs/datasets/euroc_mh01.yaml")
    if not config_path.exists():
        sys.exit(f"Error: config not found: {config_path}")

    cfg = load_config(config_path)
    seq_root = resolve_sequence_root(cfg)
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

    # Output path
    output_dir = Path("results/rerun")
    output_dir.mkdir(parents=True, exist_ok=True)
    rrd_path = output_dir / f"{sequence}.rrd"

    # Initialise Rerun
    rr.init(f"slam_core/{sequence}", spawn=False)
    rr.save(str(rrd_path))

    # Log
    log_imu(imu_rows)
    n_stereo = log_stereo_images(cam0_entries, cam1_entries, cam0_dir, cam1_dir)

    # Timing diagnostics
    if imu_rows:
        duration = imu_rows[-1]["timestamp_s"] - imu_rows[0]["timestamp_s"]
        print(f"  Duration: {duration:.2f} s")
    print(f"  IMU samples logged:    {len(imu_rows)}")
    print(f"  Stereo pairs logged:   {n_stereo}")
    print(f"  Saved: {rrd_path}")


if __name__ == "__main__":
    main()
