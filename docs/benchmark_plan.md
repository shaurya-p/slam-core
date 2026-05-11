# Benchmark Plan

## Datasets

### EuRoC MAV (Primary)

- Sequences: MH_01 through MH_05, V1_01 through V2_03.
- Camera: stereo 20 Hz, IMU 200 Hz.
- Ground truth: Vicon / Leica.
- Use cases: indoor, varying texture, varying motion.

### TUM-VI (Primary)

- Sequences: corridor, room, outdoors subsets.
- Camera: fisheye stereo 20 Hz, IMU 200 Hz.
- Ground truth: motion capture + laser.
- Use cases: narrow corridors, rotation-heavy, long sequences.

### KITTI (Optional)

- Sequences: 00–10 for odometry.
- Driving domain only — for generalization testing.
- Use sparingly; not the primary target domain.

## Metrics

| Metric | Description |
|---|---|
| ATE | Absolute Trajectory Error (RMSE, meters) |
| RPE | Relative Pose Error (translation + rotation) |
| Latency | Frontend + backend processing time per frame |
| Track quality | Mean track length, outlier ratio, reprojection error histogram |
| Residuals | Post-optimization reprojection and IMU residual distributions |
| Covariance | Estimated uncertainty vs. ground truth error |
| Loop closure | Trajectory error before and after loop closure on looping sequences |

## Reproducibility

- All experiments run via script with pinned dataset paths and config files.
- Results stored in `outputs/` (git-ignored; large files).
- Summary tables and plots committed to `docs/blog_assets/`.
- Config files committed to `config/` (to be created during implementation).
- Docker or conda environment for reproducible dependency management.

## Reporting

- Per-sequence results tables.
- Trajectory plots overlaid on ground truth.
- Failure case annotations linked to `docs/failure_atlas.md`.
