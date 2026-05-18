# Benchmark Plan

No benchmarks have been run yet. This document records the intended approach for when the system is ready.

## Datasets

### EuRoC MAV (primary)

- Sequences: MH_01 through MH_05, V1_01 through V2_03.
- Camera: stereo 20 Hz, IMU 200 Hz.
- Ground truth: Vicon / Leica.
- Use cases: indoor, varying texture, varying motion.

### TUM-VI (future)

- Sequences: corridor, room, outdoors subsets.
- Camera: fisheye stereo 20 Hz, IMU 200 Hz.
- Ground truth: motion capture + laser.
- No loaders or scripts exist yet.

### KITTI (optional, future)

- Sequences: 00–10 for odometry.
- Driving domain only — for generalization testing.

## Metrics

Metrics applicable once a working trajectory estimator exists:

| Metric | Description |
|---|---|
| ATE | Absolute Trajectory Error (RMSE, meters) |
| RPE | Relative Pose Error (translation + rotation) |
| Latency | Frontend + backend processing time per frame |

The following metrics require feature tracking, optimization, covariance estimation, or loop closure — none of which are implemented yet:

- Track quality (mean track length, reprojection error histogram)
- Post-optimization residuals
- Covariance vs. ground truth error
- Loop closure trajectory correction

## Reproducibility

- Experiments run via script with pinned dataset paths and config files.
- Configs committed under `configs/`.
- Results stored under `results/` (git-ignored).
- Dependencies managed via `uv`. No Docker or conda environment is set up yet.
- Summary tables and plots will be committed to `docs/blog_assets/` when available.
