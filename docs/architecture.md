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

- `export_gyro_propagation` — reads EuRoC IMU CSV, integrates orientation, writes row-major `R_W_B` CSV; `--gyro-bias bx by bz` applies offline bias correction (offline validation only)
- `export_imu_state_propagation` — reads EuRoC IMU CSV, propagates full `ImuState`, writes state CSV; `--init-from-gt` initializes position/velocity/orientation from GT; `--gyro-bias` / `--accel-bias` supply offline-validated constant biases (offline validation only)
- `evaluate_gyro_bias_from_gt` — estimates gyro bias per non-overlapping window by comparing raw gyro integration against GT relative rotation; sign convention: `omega_meas = omega_true + bias`
- `evaluate_accel_bias_from_gt` — estimates accel bias per window using GT velocity change and GT orientation at each IMU timestamp; `--gravity-z` sets the world-frame gravity value (default −9.81 m/s²)

### Visualization (`scripts/`)

Scripts are organized into two subdirectories:

- `scripts/rerun/` — Rerun recordings for spatial visualization and error dashboards
- `scripts/plots/` — Matplotlib PNG outputs for scalar analysis

Rerun scripts: `rerun_euroc_debug.py` (stereo + raw IMU), `rerun_euroc_gyro_propagation.py` (gyro-only orientation vs GT), `rerun_euroc_gyro_bias_comparison.py` (raw vs bias-corrected gyro orientation, two 3D panels), `rerun_euroc_gyro_bias_eval.py` (per-window gyro bias scalar channels), `rerun_euroc_imu_state_propagation.py` (full IMU state trajectories and error vs GT), `rerun_euroc_imu_state_bias_comparison.py` (raw vs bias-corrected IMU state error dashboard).

Matplotlib scripts: `plot_gyro_bias_impact.py` (raw vs bias-corrected orientation drift), `plot_accel_bias_eval.py` (accel bias components and velocity error over time).

## Key conventions

- Poses: `T_A_B` maps points from frame B into frame A. `T_A_C = T_A_B.compose(T_B_C)`.
- IMU: body frame. `R_W_B` maps body frame B into world frame W.
- Camera: Z-forward, X-right, Y-down (OpenCV convention).
- Gravity: −Z in world frame.
- Time: float64 seconds throughout.
- Gyro: rad/s, body frame. Accel: m/s², body frame.

## Planned direction

IMU propagation and offline bias diagnostics are complete. The next focus is preintegration and visual constraints:

1. IMU preintegration — preintegrate IMU measurements between keyframe timestamps for use as IMU factors
2. Gravity-aligned initialization without GT — accelerometer-based world-frame alignment
3. Camera models — projection, distortion, reprojection error
4. Feature detection and tracking
5. Visual-inertial factor graph — reprojection + IMU preintegration factors

Fixed-lag VIO backend, GTSAM/iSAM2 pose graph, loop closure, and ROS2 integration are not yet started.
