# FG-3: Preintegration covariance and bias Jacobians

Extends the FG-1 conventions ([fg1_manifold_lm.md](fg1_manifold_lm.md)).
Reference: Forster et al., TRO 2017, §V–VI and Appendix B.

## Error state and noise model

Error state per preintegration interval, Forster ordering (decision #5):

```
e = [dθ, dv, dp] ∈ R⁹
dθ: right rotation error, ΔR_true = ΔR̂ · exp_so3(dθ)   (rad)
dv, dp: additive                                        (m/s, m)
```

Measurement noise is white with continuous-time densities σ_g
[rad/s/√Hz], σ_a [m/s²/√Hz] (exactly the EuRoC `sensor.yaml` fields).
Per integration step of length dt the discrete per-axis variance is
σ²/dt — density² × bandwidth, with bandwidth 1/dt.

## Covariance propagation

Per step, with pre-update ΔR_k, corrected ω̂, â, and W = [·]×:

```
Σ_{k+1} = A Σ_k Aᵀ + (σ_g²/dt) B_g B_gᵀ + (σ_a²/dt) B_a B_aᵀ

A = [ exp(ω̂dt)ᵀ        0      0 ]      B_g = [ J_r(ω̂dt)·dt ]   B_a = [    0     ]
    [ −ΔR_k[â]×dt       I      0 ]            [      0      ]         [ ΔR_k·dt  ]
    [ −½ΔR_k[â]×dt²    I·dt    I ]            [      0      ]         [ ½ΔR_k·dt² ]
```

Row meanings: dθ evolves in the *local* frame (hence exp(ω̂dt)ᵀ transports
it through the step, and J_r maps gyro noise into it); rotation error
tilts the accumulated acceleration (the −ΔR[â]× couplings); dp integrates
dv. Every RHS term uses the pre-update state — updating deltas first is
the classic implementation bug.

## Bias-correction Jacobians

Deltas were integrated at linearization biases (b_g, b_a). For a bias
update (δb_g, δb_a), first-order corrections (recursions, pre-update RHS):

```
∂ΔR/∂b_g: D_R ← exp(ω̂dt)ᵀ·D_R − J_r(ω̂dt)·dt          (log-space)
∂Δv/∂b_a: D_va ← D_va − ΔR_k·dt
∂Δv/∂b_g: D_vg ← D_vg − ΔR_k[â]×·D_R·dt
∂Δp/∂b_a: D_pa ← D_pa + D_va·dt − ½ΔR_k·dt²
∂Δp/∂b_g: D_pg ← D_pg + D_vg·dt − ½ΔR_k[â]×·D_R·dt²

ΔR(b_g+δb_g) ≈ ΔR·exp(D_R·δb_g)
Δv(b+δb)    ≈ Δv + D_vg·δb_g + D_va·δb_a     (Δp analogous)
```

Note ∂ΔR/∂b_a ≡ 0: accelerometer bias cannot rotate anything — the tests
assert this numerically.

## Validation (tests/test_preintegration.cpp)

- All five Jacobians vs central-difference re-integration on a 1 s
  time-varying trajectory; corrected deltas track exact re-integration to
  O(‖δb‖²) and beat uncorrected by >10×.
- Monte-Carlo consistency: 500 noisy integrations; mean NEES
  eᵀΣ⁻¹e ≈ 9 (dim) within sampling error, per-axis variance ratios ≈ 1.
  NEES is the strongest check — it validates magnitudes *and*
  cross-correlations, in the exact error convention above.
