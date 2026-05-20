# slam-core

A geometry-first Visual-Inertial SLAM project. The goal is a clean, testable implementation built in disciplined stages, with Rerun-based visual debugging and reproducible experiments on EuRoC MAV.

## Current state

- SO(3) / SE(3) geometry utilities (`skew`, `exp_so3`, `log_so3`, `is_valid_rotation`, SE3 compose/inverse/apply)
- IMU measurement type with validation
- Gyro-only SO(3) orientation propagation (`R_W_B_next = R_W_B * exp_so3(gyro_radps * dt_s)`)
- Bias-corrected gyro propagation
- Full `ImuState` (position, velocity, orientation, gyro/accel biases) with single-step propagation using gyro, accelerometer, and gravity
- EuRoC dataset loading utilities (IMU, camera, stereo pairs, ground truth with position/velocity/quaternion)
- C++ export tools for orientation and full IMU state CSVs; `--init-from-gt` initializes from GT; `--gyro-bias` / `--accel-bias` supply offline-validated constant biases
- Offline GT-based bias diagnostics: `evaluate_gyro_bias_from_gt`, `evaluate_accel_bias_from_gt` estimate per-window gyro and accel bias from GT; MH_01_easy results show ~0.078 rad/s gyro bias (z-dominated) and ~0.16 m/s² accel bias norm
- Offline bias-corrected validation: supplying GT-derived biases reduces gyro-only mean orientation error from ~108° to ~7° over 180 s, and full IMU dead-reckoning position error from ~50 km to ~400 m at 60 s — demonstrating that pure inertial navigation still requires visual constraints
- Visualization: Rerun scripts for debug, gyro propagation, gyro/accel bias comparison, and IMU state error dashboard; Matplotlib scripts for scalar bias analysis; organized under `scripts/rerun/` and `scripts/plots/`

## Future work

IMU preintegration, gravity-aligned initialization without GT, camera projection/reprojection, visual odometry, visual-inertial factor graph, loop closure.

## Docs

- [Architecture](docs/architecture.md)
- [EuRoC dataset setup](docs/datasets/euroc.md)
- [Benchmark plan](docs/benchmark_plan.md)
- [Design decisions](docs/design_decisions.md)
- [Failure atlas](docs/failure_atlas.md)
