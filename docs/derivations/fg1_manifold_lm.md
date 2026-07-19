# FG-1: On-manifold least squares — conventions and derivations

Reference note for the `optim/` module and the FG-1 factors. Written to be
re-derivable: each result states the convention it depends on.

## Perturbation convention

All rotation variables use the **right (local) perturbation**:

```
R ← R · exp_so3(δθ),   δθ ∈ R³ (rad), expressed in the body frame
```

Consequence: Jacobians of any residual w.r.t. δθ are derivatives *through
this retraction*, evaluated at δθ = 0. Euclidean variables are additive.
This matches the error-state ordering already fixed for the estimator
(Forster `[δθ, δv, δp]` + biases; see design decision #5).

## SO(3) right Jacobian

The exponential map is not additive: `exp(w + dw) ≠ exp(w)·exp(dw)`.
The right Jacobian `J_r` is the correction:

```
exp_so3(w + dw) ≈ exp_so3(w) · exp_so3(J_r(w) · dw)

J_r(w)   = I − (1−cos θ)/θ² [w]× + (θ − sin θ)/θ³ [w]×²,   θ = ‖w‖
J_r⁻¹(w) = I + ½[w]× + (1/θ² − (1+cos θ)/(2θ sin θ)) [w]×²
```

The identity used everywhere for log-type residuals:

```
log_so3(exp_so3(w) · exp_so3(dw)) ≈ w + J_r⁻¹(w) · dw          (small dw)
```

Numerics: Taylor branches below θ = 1e-5 (`J_r ≈ I − ½W + ⅙W²`,
`J_r⁻¹ ≈ I + ½W + 1/12·W²`); the `J_r⁻¹` quadratic coefficient tends to
1/θ² as θ → π. Domain θ < π.

## Gauss-Newton / Levenberg-Marquardt

Cost `F(x) = ½ Σ_f ‖r_f(x)‖²`. Linearize each residual through the
retraction: `r(x ⊞ δ) ≈ r + J δ`. Normal equations:

```
H δ = −b,   H = Σ JᵀJ,   b = Σ Jᵀr
```

LM damps with Marquardt scaling `(H + λ·diag(H)) δ = −b`: λ→0 gives
Gauss-Newton, λ→∞ gives scaled gradient descent. Accept the step iff cost
decreases (then λ ← λ/10), else roll back and λ ← 10λ. Convergence: step
norm or relative cost decrease below tolerance. A vanishing step at a
local optimum must terminate *before* the cost check — a zero step never
strictly decreases cost.

## FG-1 factor derivations (identity information)

**SO(3) prior** — `r = log_so3(R_priorᵀ R)`:
perturb `R → R·exp(δθ)`; `r(δθ) = log(exp(r)·exp(δθ)) ≈ r + J_r⁻¹(r)·δθ`,
so `∂r/∂δθ = J_r⁻¹(r)`.

**Point alignment** — `r = R·p_B + t − q_A`:
`R·exp(δθ)·p_B ≈ R(I + [δθ]×)p_B = R·p_B − R·[p_B]×·δθ`, so
`∂r/∂δθ = −R·[p_B]×`, `∂r/∂δt = I`.

**Relative pose** (poses `T_W_Bi`, `T_W_Bj`; measurement `R_meas = R_iᵀR_j`,
`t_meas = R_iᵀ(t_j − t_i)`):

- `r_R = log_so3(R_measᵀ R_iᵀ R_j)`. Push perturbations through using
  `exp(w)·A = A·exp(Aᵀw)` for `A ∈ SO(3)`:
  `∂r_R/∂δθ_j = J_r⁻¹(r_R)`, `∂r_R/∂δθ_i = −J_r⁻¹(r_R)·R_jᵀR_i`.
- `r_t = R_iᵀ(t_j − t_i) − t_meas`. From
  `exp(−δθ_i)·v ≈ v + [v]×·δθ_i`:
  `∂r_t/∂δθ_i = [R_iᵀ(t_j−t_i)]×`, `∂r_t/∂δt_i = −R_iᵀ`,
  `∂r_t/∂δt_j = R_iᵀ`, `∂r_t/∂δθ_j = 0`.

Every Jacobian above is validated against central-difference numeric
differentiation in `tests/test_optim.cpp` via `tests/numeric_jacobian.hpp`.

## Reading

- Forster, Carlone, Dellaert, Scaramuzza, *On-Manifold Preintegration for
  Real-Time Visual-Inertial Odometry* (TRO 2017) — §III and Appendix A for
  J_r and the perturbation calculus.
- Solà, Deray, Atchuthan, *A micro Lie theory for state estimation in
  robotics* (arXiv:1812.01537) — compact reference for all identities used
  here.
