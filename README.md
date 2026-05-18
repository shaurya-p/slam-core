# slam-core

A geometry-first Visual-Inertial SLAM project. The goal is a clean, testable implementation built in disciplined stages, with Rerun-based visual debugging and reproducible experiments on EuRoC MAV.

## Current state

- SO(3) / SE(3) geometry utilities (`skew`, `so3_exp`, `so3_log`, `is_valid_rotation`, SE3 compose/inverse/apply)
- IMU measurement type with validation
- Gyro-only SO(3) orientation propagation (`R_W_B_next = R_W_B * so3_exp(gyro_radps * dt_s)`)
- EuRoC dataset loading utilities (IMU, camera, stereo pairs, ground truth)
- Rerun debug visualization: stereo images, raw IMU, derived IMU channels
- Rerun gyro propagation visualization: estimated vs GT 3D orientation axes, SO(3) geodesic error, RPY error components
- C++ export tool for orientation CSVs

## Future work

IMU bias handling, gravity initialization, full IMU state propagation, IMU preintegration, camera projection/reprojection, visual odometry, visual-inertial factor graph, loop closure.

## Docs

- [Architecture](docs/architecture.md)
- [EuRoC dataset setup](docs/datasets/euroc.md)
- [Benchmark plan](docs/benchmark_plan.md)
- [Design decisions](docs/design_decisions.md)
- [Failure atlas](docs/failure_atlas.md)
