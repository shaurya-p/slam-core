# slam-core Project State

`slam-core` is a career-defining public GitHub repo for a geometry-first Visual-Inertial SLAM system built in disciplined stages: geometry → IMU propagation → preintegration → camera models → tracking → VO → VIO → optimization → loop closure. Visualization is a first-class support layer for debugging, validation, and reproducible experiments on EuRoC MAV.

## Engineering principles

- Keep code simple, readable, and professional.
- Prefer small, clean, behavior-preserving patches.
- Be explicit about frames, units, and timestamps.
- Add meaningful tests for meaningful behavior.
- Avoid fake complexity and learning-repo tone.
- Do not introduce GTSAM, Ceres, Sophus, ROS2, Docker, pybind11, learned models, or full preintegration until the foundation is ready.

## Current stack

- CMake (>= 3.24), C++17, Eigen (3.4–5.x), GoogleTest via FetchContent
- Python tooling with `uv`; pytest for loader tests
- Rerun for debug visualization
- EuRoC MAV as the primary benchmark dataset
- Apache-2.0 license; GitHub Actions CI (macOS + Ubuntu: build with
  warnings-as-errors, ctest, pytest, pinned clang-format check)

## Current implemented foundation

### Geometry (`src/geometry/`)

- `skew`, `exp_so3`, `log_so3`, `is_valid_rotation`
- `SE3::compose`, `SE3::inverse`, `SE3::apply`

### Conventions

All frame/unit/timestamp/rotation conventions live in
[architecture.md](architecture.md). The load-bearing ones: `T_A_B` maps B
into A; `R_W_B` body-to-world; `gravity_W = [0, 0, -9.81]` m/s²; float64
seconds; `omega_meas = omega_true + bias`; EuRoC GT quaternion is Hamilton
w-first `q_W_B`; error-state ordering is Forster `[δθ, δv, δp]` + biases.

### IMU (`src/imu/`)

- `ImuMeasurement` — `timestamp_s`, `gyro_radps`, `accel_mps2`; `is_finite`, `has_strictly_increasing_timestamp`
- `propagate_gyro(R_W_B, gyro_radps, dt_s, gyro_bias_radps = 0)` — `R_W_B_next = R_W_B * exp_so3((gyro_radps - gyro_bias_radps) * dt_s)`; zero default bias is the raw case; throws on non-positive or non-finite `dt_s`
- `ImuState` — `timestamp_s`, `R_W_B`, `p_W_B`, `v_W_B`, `gyro_bias_radps`, `accel_bias_mps2`
- `propagate_imu_state(state, meas, gravity_W, dt_s)` — bias-corrected gyro and accel; accel rotated to world frame and gravity-compensated; biases carried forward unchanged

### IMU preintegration (`src/imu/preintegration.hpp`)

- `PreintegratedImu` — stores `delta_R` (SO3), `delta_v`, `delta_p`, `delta_t_s`; default-constructed identity/zero; `reset()` restores identity/zero
- `integrate(preint, meas, gyro_bias, accel_bias, dt_s)` — single-step zeroth-order hold; bias-corrected gyro and accel; deltas accumulated in starting body/keyframe frame; gravity excluded; throws on invalid input
- `integrate_sequence(measurements, gyro_bias, accel_bias)` — contiguous timestamped sequence; N measurements → N−1 steps; dt derived from adjacent timestamps
- `integrate_window(stream, t_start_s, t_end_s, gyro_bias, accel_bias)` — selects contained measurements where `t_start_s <= timestamp_s <= t_end_s`; no boundary interpolation; `result.delta_t_s` may be smaller than `t_end_s − t_start_s`; caller must ensure stream brackets desired keyframe timestamps
- All functions throw `std::invalid_argument` on non-finite values, non-positive dt, non-increasing timestamps, or too few measurements
- Covariance, Jacobians, residuals, and optimizer integration are future work

### EuRoC CSV IO (`src/io/`)

- `read_euroc_imu_csv`, `read_euroc_gt_csv` (`EurocGtSample`: `p_W_B`, normalized `q_W_B`, `v_W_B`, `R_W_B()`), binary-search `nearest_gt`
- Shared by all four tools; errors via exceptions; skipped-row counts surfaced to callers

**Current test count: 113/113 C++ (ctest), 6/6 Python (pytest).**

## EuRoC / tooling status

### Dataset utilities (`python/slam_core_tools/datasets/euroc`)

`ImuSample`, `CameraFrame`, `StereoPair`, `GroundTruthSample` (`timestamp_s`, `p_x/y/z` [m], `q_w/x/y/z`, `v_x/y/z` [m/s]); `load_config`, `resolve_sequence_root`, `read_imu_csv`, `validate_imu`, `read_cam_csv`, `associate_stereo_pairs`, `read_groundtruth_csv`. Installed via Hatchling (`uv sync`, no `sys.path` hacks).

### Export tools (`tools/`)

| Tool | Description |
|---|---|
| `export_gyro_propagation` | EuRoC IMU → `timestamp_s, r00..r22` (row-major `R_W_B`); `--gyro-bias bx by bz` for offline bias correction |
| `export_imu_state_propagation` | EuRoC IMU → full state CSV (`timestamp_s`, `p/v/q/R/biases`); `--init-from-gt` seeds `R/p/v` from nearest GT; `--gyro-bias` / `--accel-bias` apply offline-validated biases |
| `evaluate_gyro_bias_from_gt` | Per-window gyro bias: `bias = −log_SO3(R_err) / dt` where `R_err = R_rel_imu.T * R_rel_gt` |
| `evaluate_accel_bias_from_gt` | Per-window accel bias: `solve(M, Δv_err)` where `M = Σ R_W_B_gt_i * dt_i`; uses GT orientation per IMU sample |

### Visualization / plotting (`scripts/`)

Shared helpers live in `slam_core_tools.viz` (palettes, GT indexing, CSV loading, Rerun logging).

Rerun scripts (`scripts/rerun/`): `rerun_euroc_debug.py` (stereo + raw IMU), `rerun_euroc_gyro_propagation.py` (gyro vs GT orientation; N CSVs with `--labels` for raw-vs-corrected comparison), `rerun_euroc_gyro_bias_eval.py` (per-window gyro bias scalars), `rerun_euroc_imu_state_propagation.py` (IMU state trajectories + GT error; N CSVs with `--labels` for comparison dashboards).

Matplotlib scripts (`scripts/plots/`): `plot_gyro_bias_impact.py` (orientation drift comparison), `plot_accel_bias_eval.py` (accel bias components + velocity error).

## Offline IMU bias diagnostics (complete)

GT-based bias estimation. Diagnostic only — not a runtime estimator.

**MH_01_easy results (180 s, GT-initialized).** Estimates stable across 1 s and 5 s windows:

- Gyro bias: `[−0.00333, 0.02130, 0.07808]` rad/s (z-dominated, ~0.08 rad/s norm)
- Accel bias: `[−0.02756, 0.13727, 0.07670]` m/s² (y-dominated, ~0.16 m/s² norm)

| Metric | Raw | Offline bias-corrected |
|---|---|---|
| Mean gyro-only orientation error (180 s) | ~108° | ~7° |
| Position error at ~60 s (GT-init) | ~50,000 m | ~400 m |

Bias correction dramatically reduces drift. At ~400 m in 60 s, pure inertial dead-reckoning still fails for localization — visual constraints are necessary.

**These bias values are GT-derived and manually supplied. The runtime system does not estimate bias online yet.**

## Useful commands

```bash
uv sync
cmake --build build -j$(sysctl -n hw.ncpu)
ctest --test-dir build --output-on-failure

# EuRoC debug
uv run python scripts/rerun/rerun_euroc_debug.py configs/datasets/euroc_mh01.yaml \
  --dataset-root "$HOME/datasets" --max-frames 300 --image-stride 2 --imu-stride 2

# Gyro propagation vs GT (add a second CSV + --labels for raw-vs-corrected)
./build/tools/export_gyro_propagation \
  "$HOME/datasets/euroc/MH_01_easy/mav0/imu0/data.csv" \
  results/imu/MH_01_easy_gyro_orientations.csv
uv run python scripts/rerun/rerun_euroc_gyro_propagation.py \
  configs/datasets/euroc_mh01.yaml results/imu/MH_01_easy_gyro_orientations.csv \
  --dataset-root "$HOME/datasets" --frame-stride 5 --max-duration-s 30

# IMU state propagation vs GT
./build/tools/export_imu_state_propagation \
  "$HOME/datasets/euroc/MH_01_easy/mav0/imu0/data.csv" \
  results/imu/MH_01_easy_imu_state.csv --init-from-gt
uv run python scripts/rerun/rerun_euroc_imu_state_propagation.py \
  results/imu/MH_01_easy_imu_state.csv \
  --config configs/datasets/euroc_mh01.yaml --dataset-root "$HOME/datasets" \
  --frame-stride 50 --max-duration-s 30 \
  --output results/rerun/MH_01_easy_imu_state_vs_gt.rrd

# Offline bias-corrected IMU state comparison (same script, two CSVs)
./build/tools/export_imu_state_propagation \
  "$HOME/datasets/euroc/MH_01_easy/mav0/imu0/data.csv" \
  results/imu/MH_01_easy_imu_state_bias_corrected.csv \
  --init-from-gt \
  --gyro-bias -0.00332713 0.0212966 0.0780824 \
  --accel-bias -0.027562 0.137274 0.076698
uv run python scripts/rerun/rerun_euroc_imu_state_propagation.py \
  results/imu/MH_01_easy_imu_state.csv \
  results/imu/MH_01_easy_imu_state_bias_corrected.csv \
  --labels raw bias_corrected \
  --config configs/datasets/euroc_mh01.yaml --dataset-root "$HOME/datasets" \
  --max-duration-s 180 \
  --output results/rerun/MH_01_easy_imu_state_bias_comparison.rrd
```

## Current observations

- EuRoC GT starts later than the IMU CSV; `--init-from-gt` skips IMU rows before GT coverage begins to ensure the initial state matches a valid GT sample.
- The SO(3) gyro convention `R_next = R_W_B * exp_so3(gyro_B * dt)` (body-frame gyro, right-multiply) is confirmed correct against GT.
- EuRoC quaternion `q_W_B` with `Eigen::Quaterniond(w,x,y,z).toRotationMatrix()` → `R_W_B` is confirmed consistent with the project convention.

## Next direction

IMU propagation, offline bias diagnostics, the preintegration foundation, gravity-aligned stationary initialization, and the undistorted pinhole camera model are complete. Repo hardening (license, CI, shared IO, shared viz helpers) landed 2026-07. The next focus is the factor-graph backbone:

1. **Factor-graph foundation** — on-manifold state blocks and a small Levenberg–Marquardt optimizer over Eigen; SO(3) right Jacobians; numeric-vs-analytic Jacobian test harness
2. **Preintegration covariance + bias Jacobians** — extend `PreintegratedImu` for use as an IMU factor
3. **IMU + prior + reprojection factors** — validated on EuRoC with GT priors and synthetic landmark tracks before any frontend exists
4. **Feature detection and tracking**, then sliding-window VIO

## Project state update rule

This file should only be edited after a meaningful milestone is complete, and only when explicitly requested by the project owner.

Do not update this file for small intermediate edits, minor bug fixes, formatting changes, or incomplete work.
