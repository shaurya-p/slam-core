# Architecture

## Implemented

### Geometry (`src/geometry/`, `include/slam_core/geometry/`)

- `skew`, `exp_so3`, `log_so3`, `is_valid_rotation`
- `SE3::compose`, `SE3::inverse`, `SE3::apply`

### IMU (`src/imu/`, `include/slam_core/imu/`)

- `ImuMeasurement` — timestamped gyro + accel with finite-value validation
- `propagate_gyro(R_W_B, gyro_radps, dt_s)` — zeroth-order-hold SO(3) integration
- `propagate_gyro_biascorrected(R_W_B, gyro_radps, gyro_bias_radps, dt_s)` — bias-subtracted gyro integration
- `ImuState` — `timestamp_s`, `R_W_B`, `p_W_B`, `v_W_B`, `gyro_bias_radps`, `accel_bias_mps2`
- `propagate_imu_state(state, meas, gravity_W, dt_s)` — single-step propagation: bias-corrected gyro, bias-corrected accel rotated into world and gravity-compensated, biases carried forward unchanged

### Python tooling (`python/slam_core_tools/datasets/euroc`)

- EuRoC CSV loaders: `ImuSample`, `CameraFrame`, `StereoPair`, `GroundTruthSample`
- `read_imu_csv`, `read_cam_csv`, `associate_stereo_pairs`, `read_groundtruth_csv`
- `load_config`, `resolve_sequence_root`, `validate_imu`
- `GroundTruthSample` fields: `timestamp_s`, `p_x/y/z`, `q_w/x/y/z`, `v_x/y/z`

### Tools (`tools/`)

- `export_gyro_propagation` — reads EuRoC IMU CSV, integrates orientation, writes row-major `R_W_B` CSV
- `export_imu_state_propagation` — reads EuRoC IMU CSV, propagates full `ImuState`, writes state CSV; `--init-from-gt` initializes position/velocity/orientation from GT at the first IMU timestamp inside GT coverage

### Visualization (`scripts/`)

- `rerun_euroc_debug.py` — stereo images, raw IMU, derived IMU channels
- `rerun_euroc_gyro_propagation.py` — estimated vs GT 3D orientation arrows, geodesic error, RPY error components
- `rerun_euroc_imu_state_propagation.py` — estimated and GT 3D trajectories and body frames, position/velocity scalar plots, position/velocity/orientation error vs GT

## Key conventions

- Poses: `T_A_B` maps points from frame B into frame A. `T_A_C = T_A_B.compose(T_B_C)`.
- IMU: body frame. `R_W_B` maps body frame B into world frame W.
- Camera: Z-forward, X-right, Y-down (OpenCV convention).
- Gravity: −Z in world frame.
- Time: float64 seconds throughout.
- Gyro: rad/s, body frame. Accel: m/s², body frame.

## Planned direction

The IMU propagation foundation is complete. The next focus is bias characterization and initialization before preintegration:

1. Gyro bias evaluation from GT — quantify orientation drift contribution
2. Accel bias / gravity consistency evaluation — measure residual world-frame acceleration over static segments
3. Gravity-aligned initialization without GT — accelerometer-based world-frame alignment
4. IMU preintegration — between-keyframe preintegrated measurements for use as IMU factors
5. Visual-inertial factor graph — reprojection factors + IMU factors (requires 1–4)

Feature detection and tracking, fixed-lag VIO backend, GTSAM/iSAM2 pose graph, loop closure, and ROS2 integration are not yet started.
