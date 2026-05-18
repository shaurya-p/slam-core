# Architecture

## Implemented

### Geometry (`src/geometry/`, `include/slam_core/geometry/`)

- `skew`, `so3_exp`, `so3_log`, `is_valid_rotation`
- `SE3::compose`, `SE3::inverse`, `SE3::apply`

### IMU (`src/imu/`, `include/slam_core/imu/`)

- `ImuMeasurement` — timestamped gyro + accel with finite-value validation
- `propagate_gyro(R_W_B, gyro_radps, dt_s)` — zeroth-order-hold SO(3) integration

### Python tooling (`python/slam_core_tools/datasets/euroc`)

- EuRoC CSV loaders: `ImuSample`, `CameraFrame`, `StereoPair`, `GroundTruthSample`
- `read_imu_csv`, `read_cam_csv`, `associate_stereo_pairs`, `read_groundtruth_csv`
- `load_config`, `resolve_sequence_root`, `validate_imu`

### Tools (`tools/`)

- `export_gyro_propagation` — reads EuRoC IMU CSV, integrates orientation, writes row-major `R_W_B` CSV

### Visualization (`scripts/`)

- `rerun_euroc_debug.py` — stereo images, raw IMU, derived IMU channels
- `rerun_euroc_gyro_propagation.py` — estimated vs GT 3D orientation arrows, geodesic error, RPY error components

## Key conventions

- Poses: `T_A_B` maps points from frame B into frame A. `T_A_C = T_A_B.compose(T_B_C)`.
- IMU: body frame. `R_W_B` maps body frame B into world frame W.
- Camera: Z-forward, X-right, Y-down (OpenCV convention).
- Gravity: −Z in world frame.
- Time: float64 seconds throughout.
- Gyro: rad/s, body frame. Accel: m/s², body frame.

## Planned direction

The next goal is to complete the IMU foundation before moving to camera or factor-graph work:

1. Gyro bias handling — bias-corrected orientation propagation
2. Gravity initialization / alignment — accelerometer-based world-frame alignment
3. Full IMU state propagation — position, velocity, orientation, biases
4. IMU preintegration — between-keyframe preintegrated measurements for use as IMU factors
5. Visual-inertial factor graph — reprojection factors + IMU factors (requires 1–4)

Feature detection and tracking, fixed-lag VIO backend, GTSAM/iSAM2 pose graph, loop closure, and ROS2 integration are not yet started.
