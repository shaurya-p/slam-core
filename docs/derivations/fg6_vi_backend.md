# FG-6: Full visual-inertial backend on synthetic tracks

Closes the factor-graph backbone: every FG-1..FG-5 component in one graph
on real EuRoC IMU data, before any frontend exists.

## Why synthetic tracks

Landmarks are unprojected through GT poses and the *real* cam0
calibration, then observed with Gaussian pixel noise. This isolates the
backend: any error is the estimator's, not a tracker's. When the real
frontend lands, it replaces exactly one function — the track source.

## Observability (the point of the exercise)

Graph: per-keyframe `(R_W_B, v_W_B, p_W_B, b_g, b_a)` + landmarks; IMU,
reprojection, and bias-walk factors; priors on the **first pose only**.

- **Gauge:** a pure VI system has 4 unobservable DoF — global position
  and yaw. The first-pose prior pins all 6; the extra 2 (roll/pitch)
  are already observable through gravity, so the prior is merely
  consistent with the estimate there.
- **No far-end anchor:** unlike the IMU-only chain (FG-4), the last
  keyframe has no prior — landmark reobservations hold the far end.
- **Scale** comes from the IMU (accelerometer over known Δt); monocular
  vision alone would leave it free.
- **Depth needs parallax:** landmarks observed only while the MAV hovers
  have unobservable depth and slide along their bearing indefinitely at
  flat cost — found the hard way; a minimum-parallax gate (bearing swing
  ≥ 1.5° across observations) is required at landmark acceptance. The
  real frontend will need the same gate.

## Results (MH_01_easy, 20 s, 40 keyframes, 150 landmarks, σ_px = 1)

| Quantity | Init | Optimized |
|---|---|---|
| keyframe position RMSE | 0.27 m (perturbed) | 3.3 mm |
| landmark RMSE | 0.3 m (perturbed) | 0.16 m |
| gyro bias | 0 | [−0.0028, 0.0210, 0.0792] rad/s |

The recovered gyro bias matches both the FG-4 IMU-only chain and the
offline GT evaluator — three independent estimation routes agreeing on
the same physical quantity is the strongest cross-validation this repo
has.

`tests/test_vi_graph.cpp` pins the observability contract; the tool
(`optimize_vi_chain`) reproduces the numbers above deterministically
(fixed seed).

## What is NOT here yet

Frontend tracking (KLT), triangulation-based landmark initialization,
robust losses for outliers, distortion handling, sliding-window
marginalization, and sparsity exploitation in the solver (dense LM is
fine at 40 keyframes; it will not be at 400).
