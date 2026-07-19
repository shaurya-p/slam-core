# FG-5: Reprojection factor

Builds on [fg1_manifold_lm.md](fg1_manifold_lm.md). Camera convention:
Z-forward, X-right, Y-down; pixels from the top-left.

## Projection chain

Body pose (R_W_B, p_W_B), fixed extrinsics T_B_C (EuRoC `T_BS`:
p_B = R_B_C·p_C + t_B_C), world landmark p_W_L:

```
p_B = R_W_Bᵀ (p_W_L − p_W_B)          world -> body
p_C = R_B_Cᵀ (p_B − t_B_C)            body  -> camera
π(p_C) = [fx·X/Z + cx, fy·Y/Z + cy]   camera -> pixels
r = (observed_px − π(p_C)) / σ_px ∈ R²
```

## Jacobians

Pinhole:  ∂π/∂p_C = [ fx/Z  0  −fx·X/Z² ; 0  fy/Z  −fy·Y/Z² ] — the 1/Z
and 1/Z² structure is why far points constrain rotation but barely
constrain translation.

Chain rule with J_π ≔ −(1/σ)·∂π/∂p_C (minus from observed − predicted):

```
∂r/∂p_W_L = J_π · R_C_B · R_W_Bᵀ  = J_π · R_C_W
∂r/∂p_W_B = −J_π · R_C_W
∂r/∂δθ    = J_π · R_C_B · [p_B]×      (right perturbation)
```

The δθ block: R_W_B ← R_W_B·exp(δθ) gives
p_B(δ) = exp(−δθ)·p_B ≈ p_B + [p_B]×·δθ. Note the landmark and body
position Jacobians are equal and opposite — moving the world or moving
the camera is the same observation, which is exactly the gauge freedom
priors must pin.

## Behind-camera policy

`try_project` (non-throwing) gates evaluation: Z ≤ z_min returns a large
constant residual with zero Jacobians. LM's cost comparison then rejects
any step that pushes a point behind the camera — equivalent to Ceres's
"evaluation failed" step rejection — without exceptions unwinding the
solve. Consequence: a state *initialized* behind the camera has zero
gradient and cannot recover; initialization must guarantee positive
depth (cheirality).

## Validation

tests/test_reprojection_factor.cpp: zero residual at consistent
geometry; all Jacobian blocks vs numeric differentiation; behind-camera
policy; and a mini bundle adjustment (3 poses × 12 landmarks, two poses
pinned for gauge + scale) recovering the perturbed pose and all
landmarks from noise-free pixels.

Robust (Huber) loss is deliberately deferred until real outliers exist
(frontend tracking); synthetic tracks are outlier-free.
