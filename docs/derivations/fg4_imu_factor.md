# FG-4: Preintegrated IMU factor

Builds on [fg1_manifold_lm.md](fg1_manifold_lm.md) and
[fg3_preintegration_covariance.md](fg3_preintegration_covariance.md).
Reference: Forster TRO'17 §VII.

## Residuals

Keyframes i, j with states (R_W_B, v_W_B, p_W_B), gravity g_W, interval
Δt = delta_t_s, and bias-corrected deltas ΔR̃, Δṽ, Δp̃ (FG-3 helpers at
δb = b_i − b_lin):

```
r_ΔR = log_so3(ΔR̃ᵀ · R_iᵀ R_j)                       ∈ R³ (rad)
r_Δv = R_iᵀ (v_j − v_i − g_W Δt) − Δṽ                ∈ R³ (m/s)
r_Δp = R_iᵀ (p_j − p_i − v_i Δt − ½ g_W Δt²) − Δp̃    ∈ R³ (m)
```

Gravity, excluded during integration, re-enters here analytically — that
is the entire point of preintegration: the deltas are pose-independent,
so keyframe states can move during optimization without re-integrating.

Whitening: r_w = Lᵀ r with Σ⁻¹ = L Lᵀ from the FG-3 covariance, computed
once per factor. ‖r_w‖² = rᵀ Σ⁻¹ r.

## Jacobians (right perturbation; nonzero blocks only)

With E = ΔR̃ᵀR_iᵀR_j, r_R = log(E):

```
∂r_ΔR/∂δθ_i = −J_r⁻¹(r_R)·R_jᵀR_i        ∂r_ΔR/∂δθ_j = J_r⁻¹(r_R)
∂r_ΔR/∂δb_g = −J_r⁻¹(r_R)·Eᵀ·J_r(D_R δb_g)·D_R
∂r_Δv/∂δθ_i = [R_iᵀ(v_j−v_i−gΔt)]×       ∂r_Δv/∂v_i = −R_iᵀ   ∂r_Δv/∂v_j = R_iᵀ
∂r_Δv/∂δb_g = −D_vg                       ∂r_Δv/∂δb_a = −D_va
∂r_Δp/∂δθ_i = [R_iᵀ(p_j−p_i−v_iΔt−½gΔt²)]×
∂r_Δp/∂p_i = −R_iᵀ   ∂r_Δp/∂p_j = R_iᵀ   ∂r_Δp/∂v_i = −R_iᵀΔt
∂r_Δp/∂δb_g = −D_pg                       ∂r_Δp/∂δb_a = −D_pa
```

The δb_g rotation Jacobian chains through the correction retraction
ΔR̃ = ΔR·exp(D_R δb_g): perturbing δb_g moves ΔR̃ by
J_r(D_R δb_g)·D_R·δ, transported through the log by −J_r⁻¹(r_R)·Eᵀ
(using exp(w)·A = A·exp(Aᵀw)). All blocks are whitened by Lᵀ.

Bias random walk between keyframes: r = [(b_g,j−b_g,i)/σ_g; (b_a,j−b_a,i)/σ_a],
σ = walk_density·√Δt.

## EuRoC result (optimize_imu_chain, MH_01_easy)

20 s, 40 keyframes at 0.5 s, biases initialized to zero, GT priors on
both end poses/velocities only:

- position RMSE: dead reckoning 293 m → optimized 0.19 m
- recovered mean gyro bias [−0.0028, 0.0209, 0.0792] rad/s vs the
  offline GT-derived estimate [−0.0033, 0.0213, 0.0781] — the graph
  discovers in 9 LM iterations what previously required a dedicated
  evaluator with full GT access.
