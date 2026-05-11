# Architecture

## System Overview

slam-core is a modular Visual-Inertial SLAM pipeline. Components are layered to allow independent development and testing.

## Layers

### 1. Input

- Calibrated monocular or stereo camera.
- IMU at high frequency (100–400 Hz).
- Inputs synchronized, timestamped, and validated at system boundary.

### 2. Geometric Frontend

- Feature detection and tracking (e.g., FAST + LK optical flow or descriptor matching).
- Feature track manager: per-feature lifetime, reprojection bookkeeping.
- Outlier rejection via RANSAC on essential matrix or homography.

### 3. IMU Preintegration

- On-manifold preintegration (Forster et al.).
- Bias tracking.
- Covariance propagation.
- Preintegrated measurement exposed as a factor input.

### 4. Fixed-Lag VIO Backend

- Sliding-window estimator.
- IMU preintegration factors + visual reprojection factors.
- Marginalization of old states.

### 5. GTSAM / iSAM2 Pose Graph Layer

- Loop closure integration via pose graph.
- iSAM2 incremental updates.
- Loop candidates provided externally (see Learned Modules).

### 6. Learned Modules (Optional)

- Learned modules are measurement/candidate providers only.
- They do not directly mutate estimator state.
- Examples: depth priors, place recognition for loop closure candidates.
- All outputs logged with latency, confidence, accept/reject status.

### 7. Visualization

- Rerun for trajectory, landmarks, residuals, covariance.
- Visual outputs logged for debugging and publication.

### 8. Deployment (Future)

- ROS2 integration.
- Real-time mode.

## Key Conventions

- Poses: T_world_body (SE(3)).
- IMU: body frame.
- Camera: OpenCV convention (Z-forward, X-right, Y-down).
- Gravity: -Z in world frame.
- Time: float64 seconds.
