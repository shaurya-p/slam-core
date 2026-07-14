"""
Shared helpers for visualization scripts.

Conventions:
    R_W_B        — rotation mapping body frame B into world frame W
    Quaternions  — Hamilton convention, w scalar first
    Timestamps   — float64 seconds
    Rotations    — 3x3 numpy arrays, row-major CSV storage (r00..r22)
"""

import bisect
import csv
import math
import sys
from pathlib import Path

import numpy as np
import rerun as rr

from slam_core_tools.datasets.euroc import (
    GroundTruthSample,
    load_config,
    read_groundtruth_csv,
    resolve_sequence_root,
)

# EuRoC ground-truth CSV path relative to the sequence root.
GT_REL_PATH = "mav0/state_groundtruth_estimate0/data.csv"

# Body-frame axis triads: x-axis in a series color, y/z as shorter gray context.
GT_X_COLOR    = [0, 0, 255]      # ground truth forward (x): blue
GT_AUX_COLOR  = [90, 90, 90]     # ground truth y/z: dark gray
EST_AUX_COLOR = [170, 170, 170]  # estimated y/z: light gray
# One x-axis color per estimated series (raw, corrected, ...).
SERIES_X_COLORS = [
    [255, 0, 0],     # red
    [0, 180, 80],    # green
    [220, 60, 0],    # orange
    [160, 0, 200],   # purple
]
AUX_SCALE = 0.45  # y/z arrow length as a fraction of the x arrow length

ORIENTATION_COLS = frozenset({
    "timestamp_s",
    "r00", "r01", "r02",
    "r10", "r11", "r12",
    "r20", "r21", "r22",
})

STATE_COLS = ORIENTATION_COLS | frozenset({
    "p_x", "p_y", "p_z",
    "v_x", "v_y", "v_z",
    "q_w", "q_x", "q_y", "q_z",
})


def quat_to_mat(w: float, x: float, y: float, z: float) -> np.ndarray:
    """3x3 rotation matrix from unit quaternion (Hamilton convention, w scalar first)."""
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y)],
        [2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x)],
        [2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y)],
    ])


def rpy_from_mat(R: np.ndarray) -> tuple[float, float, float]:
    """ZYX Euler angles (rad) from a 3x3 rotation matrix.

    roll  = atan2(R[2,1], R[2,2])
    pitch = asin(clamp(-R[2,0], -1, 1))
    yaw   = atan2(R[1,0], R[0,0])
    """
    roll  = math.atan2(float(R[2, 1]), float(R[2, 2]))
    pitch = math.asin(max(-1.0, min(1.0, float(-R[2, 0]))))
    yaw   = math.atan2(float(R[1, 0]), float(R[0, 0]))
    return roll, pitch, yaw


def geodesic_deg(R_err: np.ndarray) -> float:
    """SO(3) geodesic angle in degrees from an error rotation matrix."""
    cos_angle = (np.trace(R_err) - 1.0) / 2.0
    return math.acos(max(-1.0, min(1.0, float(cos_angle)))) * 180.0 / math.pi


def build_gt_index(
    samples: list[GroundTruthSample],
) -> tuple[list[float], list[np.ndarray]]:
    """Sorted timestamps and rotation matrices for nearest-neighbor lookup.

    Each quaternion is normalized before conversion; near-zero-norm
    quaternions are dropped.
    """
    times: list[float] = []
    mats:  list[np.ndarray] = []
    for s in samples:
        norm = math.sqrt(s.q_w**2 + s.q_x**2 + s.q_y**2 + s.q_z**2)
        if norm < 1e-10:
            continue
        times.append(s.timestamp_s)
        mats.append(quat_to_mat(s.q_w/norm, s.q_x/norm, s.q_y/norm, s.q_z/norm))
    return times, mats


def nearest_by_index(ts: float, samples: list, times: list[float]):
    """Nearest element of samples by absolute timestamp difference.

    times must be the sorted timestamps of samples, built once by the caller.
    Ties resolve to the earlier sample; clamps at both ends.
    """
    idx = bisect.bisect_left(times, ts)
    if idx == 0:
        return samples[0]
    if idx >= len(samples):
        return samples[-1]
    if ts - times[idx - 1] <= times[idx] - ts:
        return samples[idx - 1]
    return samples[idx]


def load_csv_rows(csv_path: Path, expected_cols: frozenset, kind: str) -> list[dict]:
    """Read a CSV with a header into float-valued row dicts.

    Raises:
        FileNotFoundError: if csv_path does not exist.
        ValueError: if expected_cols is not a subset of the header.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"{kind} CSV not found: {csv_path}")
    rows: list[dict] = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not expected_cols.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"{kind} CSV missing required columns.\n"
                f"Expected subset: {sorted(expected_cols)}\n"
                f"Got: {reader.fieldnames}"
            )
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def load_orientation_csv(csv_path: Path) -> list[tuple[float, np.ndarray]]:
    """Read a CSV exported by export_gyro_propagation.

    Returns a list of (timestamp_s, R_W_B) with R_W_B as a 3x3 numpy array.
    """
    rows = load_csv_rows(csv_path, ORIENTATION_COLS, "Orientation")
    return [(row["timestamp_s"], mat_from_row(row)) for row in rows]


def load_state_csv(csv_path: Path) -> list[dict]:
    """Read a CSV exported by export_imu_state_propagation into row dicts."""
    return load_csv_rows(csv_path, STATE_COLS, "State")


def mat_from_row(row: dict) -> np.ndarray:
    """3x3 rotation matrix from a row dict with r00..r22 columns."""
    return np.array([
        [row["r00"], row["r01"], row["r02"]],
        [row["r10"], row["r11"], row["r12"]],
        [row["r20"], row["r21"], row["r22"]],
    ])


def load_sequence_gt(
    config_path: Path,
    dataset_root: str | None,
) -> tuple[dict, Path, list[GroundTruthSample]]:
    """Resolve a sequence config and load its ground truth.

    Returns (cfg, seq_root, gt_samples). Exits with a message on failure —
    intended for script main() use only.
    """
    try:
        cfg      = load_config(config_path)
        seq_root = resolve_sequence_root(dataset_root, cfg)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"Error: {e}")

    gt_csv = seq_root / GT_REL_PATH
    try:
        gt_samples = read_groundtruth_csv(gt_csv)
    except FileNotFoundError:
        sys.exit(
            f"Error: Ground-truth CSV not found: {gt_csv}\n"
            f"Expected EuRoC path relative to sequence root: {GT_REL_PATH}"
        )
    return cfg, seq_root, gt_samples


def set_time_since(t_s: float, start_s: float) -> None:
    """Set the Rerun timeline to t_s relative to start_s (seconds)."""
    rr.set_time("time", duration=t_s - start_s)


def log_axis_triad(
    entity: str,
    R: np.ndarray,
    x_color: list[int],
    aux_color: list[int],
    length: float = 1.0,
    position: list[float] | None = None,
) -> None:
    """Log a body-frame axis triad: full-length x arrow, shorter y/z arrows.

    R columns are the body x/y/z axes expressed in the display frame.
    """
    if position is None:
        position = [0.0, 0.0, 0.0]
    rr.log(
        entity,
        rr.Arrows3D(
            origins=[position, position, position],
            vectors=[
                R[:, 0] * length,
                R[:, 1] * length * AUX_SCALE,
                R[:, 2] * length * AUX_SCALE,
            ],
            colors=[x_color, aux_color, aux_color],
        ),
    )
