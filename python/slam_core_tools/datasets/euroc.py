"""
EuRoC MAV dataset loading utilities.

Units:
    timestamp_s  — seconds
    gyro_radps   — rad/s, body frame
    accel_mps2   — m/s², body frame
"""

import bisect
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ImuSample:
    timestamp_s: float                       # seconds
    gyro_radps:  tuple[float, float, float]  # rad/s, body frame (x, y, z)
    accel_mps2:  tuple[float, float, float]  # m/s², body frame (x, y, z)


@dataclass(frozen=True)
class CameraFrame:
    timestamp_s: float  # seconds
    filename:    str    # relative to image directory


@dataclass(frozen=True)
class StereoPair:
    timestamp_s:   float  # seconds; cam0 == cam1 for EuRoC synchronized stereo
    filename_cam0: str    # relative to cam0 image directory
    filename_cam1: str    # relative to cam1 image directory


@dataclass(frozen=True)
class GroundTruthSample:
    timestamp_s: float  # seconds
    p_x: float          # position in reference frame R (world) [m]
    p_y: float
    p_z: float
    q_w: float          # quaternion scalar part (q_RS_w, Hamilton convention)
    q_x: float          # quaternion vector part
    q_y: float
    q_z: float
    v_x: float          # velocity in reference frame R [m/s]
    v_y: float
    v_z: float


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_sequence_root(dataset_root_arg: str | None, cfg: dict) -> Path:
    """Resolve the EuRoC sequence root directory.

    Priority: explicit argument > SLAM_CORE_DATASETS env var > config field.

    Raises:
        ValueError: if no dataset root is found via any source.
        FileNotFoundError: if the resolved sequence directory does not exist.
    """
    dataset_root = (
        dataset_root_arg
        or os.environ.get("SLAM_CORE_DATASETS")
        or cfg.get("dataset_root")
    )
    if not dataset_root:
        raise ValueError(
            "Dataset root not found. Provide it via one of:\n"
            "  1. --dataset-root /path/to/datasets\n"
            "  2. export SLAM_CORE_DATASETS=/path/to/datasets\n"
            "  3. dataset_root: /path/to/datasets  (in config yaml)"
        )
    root = Path(dataset_root) / "euroc" / cfg["sequence"]
    if not root.exists():
        raise FileNotFoundError(
            f"Sequence directory not found: {root}\n"
            f"Expected layout: <dataset_root>/euroc/{cfg['sequence']}/"
        )
    return root


def read_imu_csv(path: Path) -> list[ImuSample]:
    """Parse a EuRoC IMU CSV into ImuSample records.

    Raises:
        FileNotFoundError: if the CSV does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"IMU CSV not found: {path}")
    samples = []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for line in reader:
            ts_ns, wx, wy, wz, ax, ay, az = (x.strip() for x in line[:7])
            samples.append(ImuSample(
                timestamp_s=float(ts_ns) * 1e-9,
                gyro_radps=(float(wx), float(wy), float(wz)),
                accel_mps2=(float(ax), float(ay), float(az)),
            ))
    return samples


def validate_imu(samples: list[ImuSample]) -> list[str]:
    """Check IMU samples for non-finite values and non-monotonic timestamps.

    Returns a list of warning strings, one per anomalous row.
    Empty list means all samples are valid.
    Does not print or call sys.exit.
    """
    warnings: list[str] = []
    prev_t = None
    for i, s in enumerate(samples):
        vals = [s.timestamp_s, *s.gyro_radps, *s.accel_mps2]
        if not all(math.isfinite(v) for v in vals):
            warnings.append(f"row {i}: non-finite value")
        if prev_t is not None and s.timestamp_s <= prev_t:
            warnings.append(f"row {i}: non-monotonic timestamp")
        prev_t = s.timestamp_s
    return warnings


def read_cam_csv(path: Path) -> list[CameraFrame]:
    """Parse a EuRoC camera CSV into CameraFrame records.

    Raises:
        FileNotFoundError: if the CSV does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Camera CSV not found: {path}")
    frames = []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for line in reader:
            ts_ns, filename = line[0].strip(), line[1].strip()
            frames.append(CameraFrame(
                timestamp_s=float(ts_ns) * 1e-9,
                filename=filename,
            ))
    return frames


def read_groundtruth_csv(path: Path) -> list[GroundTruthSample]:
    """Parse a EuRoC ground-truth CSV into GroundTruthSample records.

    Column layout (by index):
      0:   timestamp_ns
      1-3: p_RS_R_x/y/z — position in reference frame R [m]
      4-7: q_RS_w/x/y/z — quaternion, Hamilton convention (w scalar first)
      8-10: v_RS_R_x/y/z — velocity in reference frame R [m/s]
      11+: biases (ignored)

    Raises:
        FileNotFoundError: if the CSV does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Ground-truth CSV not found: {path}")
    samples = []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for line in reader:
            cols = [x.strip() for x in line]
            if len(cols) < 11:
                continue
            samples.append(GroundTruthSample(
                timestamp_s=float(cols[0]) * 1e-9,
                p_x=float(cols[1]),
                p_y=float(cols[2]),
                p_z=float(cols[3]),
                q_w=float(cols[4]),
                q_x=float(cols[5]),
                q_y=float(cols[6]),
                q_z=float(cols[7]),
                v_x=float(cols[8]),
                v_y=float(cols[9]),
                v_z=float(cols[10]),
            ))
    return samples


def nearest_gt_sample(
    ts: float,
    samples: list[GroundTruthSample],
) -> GroundTruthSample | None:
    """Return the GroundTruthSample with the nearest timestamp to ts.

    Uses bisect nearest-neighbor. Clamps to the first or last sample if ts
    is before the first or after the last timestamp.
    Returns None for an empty list.
    """
    if not samples:
        return None
    times = [s.timestamp_s for s in samples]
    idx = bisect.bisect_left(times, ts)
    if idx == 0:
        return samples[0]
    if idx >= len(samples):
        return samples[-1]
    if ts - times[idx - 1] <= times[idx] - ts:
        return samples[idx - 1]
    return samples[idx]


def associate_stereo_pairs(
    cam0: list[CameraFrame],
    cam1: list[CameraFrame],
) -> list[StereoPair]:
    """Match cam0 and cam1 frames by timestamp.

    Returns StereoPairs in cam0 order, only for timestamps present in both cameras.
    """
    cam1_by_ts = {f.timestamp_s: f.filename for f in cam1}
    pairs = []
    for frame in cam0:
        fn1 = cam1_by_ts.get(frame.timestamp_s)
        if fn1 is not None:
            pairs.append(StereoPair(
                timestamp_s=frame.timestamp_s,
                filename_cam0=frame.filename,
                filename_cam1=fn1,
            ))
    return pairs
