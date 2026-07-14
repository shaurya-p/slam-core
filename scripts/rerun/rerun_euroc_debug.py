"""
EuRoC debug logger. Reads one sequence and logs stereo images + IMU to Rerun.

Usage:
    uv run python scripts/rerun/rerun_euroc_debug.py [config_yaml] [--dataset-root PATH]
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
import math
import sys
from pathlib import Path

import cv2
import rerun as rr

from slam_core_tools.datasets.euroc import (
    ImuSample,
    StereoPair,
    associate_stereo_pairs,
    load_config,
    read_cam_csv,
    read_imu_csv,
    resolve_sequence_root,
    validate_imu,
)
from slam_core_tools.viz import set_time_since


# ---------------------------------------------------------------------------
# Rerun logging
# ---------------------------------------------------------------------------

def log_imu(samples: list[ImuSample], start_s: float) -> None:
    for i, s in enumerate(samples):
        gx, gy, gz = s.gyro_radps
        ax, ay, az = s.accel_mps2
        gyro_norm          = math.sqrt(gx**2 + gy**2 + gz**2)
        accel_norm         = math.sqrt(ax**2 + ay**2 + az**2)
        accel_norm_minus_g = accel_norm - 9.81
        if i == 0:
            print(f"  [imu_derived sample 0]  gyro_norm={gyro_norm:.4f} rad/s"
                  f"  accel_norm={accel_norm:.4f} m/s²"
                  f"  norm_minus_g={accel_norm_minus_g:.4f} m/s²")
        set_time_since(s.timestamp_s, start_s)
        rr.log("imu/gyro/x",                                 rr.Scalars(gx))
        rr.log("imu/gyro/y",                                 rr.Scalars(gy))
        rr.log("imu/gyro/z",                                 rr.Scalars(gz))
        rr.log("imu/accel/x",                                rr.Scalars(ax))
        rr.log("imu/accel/y",                                rr.Scalars(ay))
        rr.log("imu/accel/z",                                rr.Scalars(az))
        rr.log("imu_derived/gyro_norm_radps",                rr.Scalars(gyro_norm))
        rr.log("imu_derived/accel_norm_mps2",                rr.Scalars(accel_norm))
        rr.log("imu_derived/accel_norm_minus_gravity_mps2",  rr.Scalars(accel_norm_minus_g))


def log_stereo_images(
    pairs: list[StereoPair],
    cam0_dir: Path,
    cam1_dir: Path,
    start_s: float,
    stride: int = 1,
    max_frames: int | None = None,
) -> tuple[int, float | None, float | None]:
    logged = 0
    t_first: float | None = None
    t_last:  float | None = None
    for i, pair in enumerate(pairs):
        if i % stride != 0:
            continue
        if max_frames is not None and logged >= max_frames:
            break
        img0 = cv2.imread(str(cam0_dir / pair.filename_cam0), cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(str(cam1_dir / pair.filename_cam1), cv2.IMREAD_GRAYSCALE)
        if img0 is None or img1 is None:
            continue
        set_time_since(pair.timestamp_s, start_s)
        rr.log("camera/cam0", rr.Image(img0))
        rr.log("camera/cam1", rr.Image(img1))
        if t_first is None:
            t_first = pair.timestamp_s
        t_last = pair.timestamp_s
        logged += 1
    return logged, t_first, t_last


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

    # Config and path resolution
    config_path = Path(args.config)
    try:
        cfg = load_config(config_path)
        seq_root = resolve_sequence_root(args.dataset_root, cfg)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"Error: {e}")

    sequence = cfg["sequence"]
    print(f"Sequence: {sequence}")
    print(f"Root:     {seq_root}")

    # Read and validate IMU
    try:
        imu_samples = read_imu_csv(seq_root / cfg["imu_csv"])
    except FileNotFoundError as e:
        sys.exit(f"Error: {e}")

    imu_warnings = validate_imu(imu_samples)
    for w in imu_warnings:
        print(f"  [warn] IMU {w}")
    if imu_warnings:
        print(f"  {len(imu_warnings)} IMU validation warning(s)")
    else:
        print(f"  IMU: {len(imu_samples)} samples, all valid")

    # Read camera CSVs and associate stereo pairs
    try:
        cam0_frames = read_cam_csv(seq_root / cfg["cam0_csv"])
        cam1_frames = read_cam_csv(seq_root / cfg["cam1_csv"])
    except FileNotFoundError as e:
        sys.exit(f"Error: {e}")

    stereo_pairs = associate_stereo_pairs(cam0_frames, cam1_frames)
    cam0_dir = seq_root / cfg["cam0_images"]
    cam1_dir = seq_root / cfg["cam1_images"]

    # Sampling parameters
    if args.full:
        imu_to_log = imu_samples
        image_stride, max_frames = 1, None
        mode = "full"
    else:
        imu_to_log = imu_samples[::args.imu_stride]
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
    start_s = imu_samples[0].timestamp_s if imu_samples else 0.0

    # Log stereo first to obtain the logged timestamp window
    n_stereo, cam_t_first, cam_t_last = log_stereo_images(
        stereo_pairs, cam0_dir, cam1_dir, start_s,
        stride=image_stride, max_frames=max_frames,
    )

    # In debug mode, align IMU to the camera window
    if not args.full:
        if cam_t_first is None or cam_t_last is None:
            sys.exit(
                "Error: no stereo frames were logged. "
                "Cannot determine camera time window for IMU alignment.\n"
                "Check that the dataset paths are correct and stereo CSVs are non-empty."
            )
        imu_to_log = [
            s for s in imu_to_log
            if cam_t_first <= s.timestamp_s <= cam_t_last
        ]

    log_imu(imu_to_log, start_s)

    # Summary
    def rel(t: float) -> float:
        return t - start_s

    if cam_t_first is not None and cam_t_last is not None:
        print(f"  Camera:  {len(stereo_pairs):6d} read,  {n_stereo:6d} logged"
              f"  [{rel(cam_t_first):.2f} – {rel(cam_t_last):.2f} s]")
    else:
        print(f"  Camera:  {len(stereo_pairs):6d} read,  {n_stereo:6d} logged")
    imu_t_first = imu_to_log[0].timestamp_s  if imu_to_log else None
    imu_t_last  = imu_to_log[-1].timestamp_s if imu_to_log else None
    if imu_t_first is not None and imu_t_last is not None:
        print(f"  IMU:     {len(imu_samples):6d} read,  {len(imu_to_log):6d} logged"
              f"  [{rel(imu_t_first):.2f} – {rel(imu_t_last):.2f} s]")
    else:
        print(f"  IMU:     {len(imu_samples):6d} read,  {len(imu_to_log):6d} logged")
    print(f"  Saved: {rrd_path}")


if __name__ == "__main__":
    main()
