# Architecture

Design and conventions. For what is currently implemented and what comes
next, see [project_state.md](project_state.md).

## Module boundaries

```
include/slam_core/ + src/     C++ library (namespace slam_core::*)
  geometry/    SO(3)/SE(3) primitives; depends only on Eigen
  imu/         measurement type, state propagation, preintegration,
               initialization; depends on geometry
  camera/      projection models; depends on geometry
  io/          dataset CSV readers (EuRoC); depends on imu types
tools/         CLI executables over the library; no estimation math of
               their own — parsing, argument handling, and CSV writing only
python/slam_core_tools/
  datasets/    EuRoC CSV loaders (pure Python, no C++ dependency)
  viz/         shared helpers for visualization scripts (palettes, GT
               indexing, CSV loading, Rerun logging)
scripts/       thin CLIs over slam_core_tools
  rerun/       interactive 3D recordings and error dashboards
  plots/       Matplotlib PNGs for scalar analysis
tests/         GoogleTest suites (C++); python/tests/ holds pytest suites
```

Rules of thumb:

- Estimation math lives in the C++ library and nowhere else. Python
  visualizes exported CSVs; it never recomputes propagation.
- Library code reports failure with exceptions; tools catch at `main`,
  print to stderr, and return a non-zero exit code.
- Invalid data is never silently ignored: readers count skipped rows,
  tools log them.

## Key conventions

- Poses: `T_A_B` maps points from frame B into frame A.
  `T_A_C = T_A_B.compose(T_B_C)`.
- IMU: body frame B. `R_W_B` maps body frame B into world frame W.
- Camera: Z-forward, X-right, Y-down (OpenCV convention).
- Gravity: `gravity_W = [0, 0, -9.81]` m/s² (−Z in world frame).
- Time: float64 seconds throughout; EuRoC nanosecond timestamps are
  converted on read (`ns * 1e-9`).
- Gyro: rad/s, body frame. Accel: m/s², body frame.
- Bias sign convention: `omega_meas = omega_true + bias`;
  `accel_meas = accel_true + bias`. Correction subtracts the bias.
- Quaternions: Hamilton convention, w scalar first. EuRoC GT quaternion is
  `q_W_B`; `Eigen::Quaterniond(q_w, q_x, q_y, q_z).toRotationMatrix()`
  yields `R_W_B`.
- Rotation integration is right-multiplied (body-frame rates):
  `R_W_B_next = R_W_B * exp_so3((gyro - bias) * dt)`.
- Error-state / covariance ordering (fixed now for future factor work):
  Forster et al. `[δθ, δv, δp]`, extended per keyframe with
  `[δb_g, δb_a]`.

## Library surface

### geometry

`skew`, `exp_so3`, `log_so3`, `is_valid_rotation`, and `SE3` with
`compose` / `inverse` / `apply`.

### imu

- `ImuMeasurement` — timestamped body-frame gyro + accel with validation
  helpers.
- `propagate_gyro(R_W_B, gyro_radps, dt_s, gyro_bias_radps = 0)` —
  zeroth-order-hold SO(3) integration; zero default bias is the raw case.
- `ImuState` + `propagate_imu_state(state, meas, gravity_W, dt_s)` — full
  state propagation (bias-corrected gyro and accel, gravity-compensated).
- `PreintegratedImu` with `integrate`, `integrate_sequence`,
  `integrate_window` — body-frame deltas between keyframe timestamps;
  gravity excluded, handled at the factor level. Covariance and bias
  Jacobians are future work.
- `estimate_R_W_B_from_stationary` — gravity alignment from a stationary
  accelerometer window (roll/pitch only; yaw unobservable).

### camera

`PinholeCamera` (no distortion): `project`, `unproject_to_bearing`,
`reprojection_error`.

### io

`read_euroc_imu_csv`, `read_euroc_gt_csv` (returning `EurocGtSample` with
`p_W_B`, normalized `q_W_B`, `v_W_B`), and binary-search `nearest_gt`.
Shared by all tools; the Python package has equivalent loaders for
visualization.

## Tools

| Tool | Purpose |
|---|---|
| `export_gyro_propagation` | gyro-only orientation CSV; optional `--gyro-bias` |
| `export_imu_state_propagation` | full-state dead-reckoning CSV; `--init-from-gt` / `--init-from-stationary`; optional biases |
| `evaluate_gyro_bias_from_gt` | per-window gyro bias from GT relative rotation |
| `evaluate_accel_bias_from_gt` | per-window accel bias from GT velocity change |

All are offline validation tools; none represent a runtime estimator.
