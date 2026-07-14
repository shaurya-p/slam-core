# Design Decisions

Public-facing record of engineering decisions and tradeoffs.

| # | Decision | Rationale | Tradeoffs | Date |
|---|---|---|---|---|
| 1 | Bias sign convention: `omega_meas = omega_true + bias`, correction subtracts the bias. | Matches the standard IMU error model and the Forster preintegration papers; one convention everywhere avoids sign bugs at module seams. | None once fixed; mixing conventions with external code requires care. | 2026-05 |
| 2 | Zeroth-order hold for all IMU integration: the measurement at `t_i` is applied constant over `[t_i, t_{i+1}]`. | Simplest deterministic scheme; error is second-order in dt at 200 Hz; midpoint schemes complicate exact test cases. | Slightly worse accuracy than midpoint/RK4 for aggressive motion. | 2026-05 |
| 3 | `integrate_window` uses contained measurements only — no boundary interpolation; `delta_t_s` may undershoot the requested window. | Keeps the preintegration exact over actual samples and trivially testable; interpolation policy is a caller/factor concern. | Callers must ensure the IMU stream brackets keyframe timestamps. | 2026-06 |
| 4 | Library code throws (`std::invalid_argument` / `std::runtime_error`); tools catch at `main` and exit non-zero. No `std::exit` outside `main`. | Library callers decide failure policy; exceptions carry context; tools keep CLI behavior. | Slightly more boilerplate in each tool's `main`. | 2026-07 |
| 5 | Error-state / covariance ordering fixed as Forster `[δθ, δv, δp]` (+ `[δb_g, δb_a]` per keyframe) ahead of any factor-graph work. | The preintegration covariance, bias Jacobians, and IMU factor all share one block layout; deciding late means rewriting tests. | Committed before the optimizer exists; changing later is expensive. | 2026-07 |
| 6 | GT nearest-neighbor lookup: binary search by absolute timestamp difference, ties resolve to the earlier sample. | Two divergent implementations (linear scan, `lower_bound`) had crept into tools; unified on the O(log n) one and verified byte-identical outputs on MH_01_easy. | None. | 2026-07 |
| 7 | Python never recomputes estimation math; scripts visualize CSVs exported by C++ tools. | One source of truth for numerics; viz bugs cannot masquerade as estimator improvements. | Requires an export step before visualizing. | 2026-05 |
| 8 | clang-format pinned to one exact version (19.1.7 via uvx) locally and in CI. | Formatting differs across clang-format releases; an unpinned check breaks on toolchain upgrades. | Contributors need uv (already required for the Python tooling). | 2026-07 |
