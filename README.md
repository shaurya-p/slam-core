# slam-core

A geometry-first Visual-Inertial SLAM project. The goal is a clean, testable implementation built in disciplined stages, with Rerun-based visual debugging and reproducible experiments on EuRoC MAV.

## Current state

- SO(3) / SE(3) geometry utilities (`skew`, `exp_so3`, `log_so3`, `is_valid_rotation`, SE3 compose/inverse/apply)
- IMU measurement type with validation
- Gyro-only SO(3) orientation propagation (`R_W_B_next = R_W_B * exp_so3(gyro_radps * dt_s)`)
- Bias-corrected gyro propagation
- Full `ImuState` (position, velocity, orientation, gyro/accel biases) with single-step propagation using gyro, accelerometer, and gravity
- EuRoC dataset loading utilities (IMU, camera, stereo pairs, ground truth with position/velocity/quaternion)
- C++ export tools for orientation CSVs and full IMU state CSVs; `--init-from-gt` initializes state from GT at the start of GT coverage
- Rerun debug visualization: stereo images, raw IMU, derived IMU channels
- Rerun gyro propagation visualization: estimated vs GT 3D orientation axes, SO(3) geodesic error, RPY error components
- Rerun IMU state visualization: estimated and GT 3D trajectories and body frames, position/velocity scalar plots, position/velocity/orientation error vs GT

## Future work

Gyro bias evaluation from GT, accel bias and gravity consistency evaluation, gravity-aligned initialization without GT, IMU preintegration, camera projection/reprojection, visual odometry, visual-inertial factor graph, loop closure.

## Docs

- [Architecture](docs/architecture.md)
- [EuRoC dataset setup](docs/datasets/euroc.md)
- [Benchmark plan](docs/benchmark_plan.md)
- [Design decisions](docs/design_decisions.md)
- [Failure atlas](docs/failure_atlas.md)
