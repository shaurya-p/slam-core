#pragma once

#include <functional>
#include <string>

#include "slam_core/optim/problem.hpp"

namespace slam_core::optim {

// State of one Levenberg-Marquardt outer iteration, passed to the
// iteration callback after the step has been accepted or rejected.
// Variable states are current at call time (rejected steps are already
// rolled back), so callbacks may read them directly for logging.
struct LmIterationSummary {
    int    iteration;      // 0-based outer iteration index
    double cost;           // cost after this iteration
    double lambda;         // damping used for the attempted step
    double step_norm;      // ‖delta‖ of the attempted step
    bool   step_accepted;  // false: cost did not decrease, state rolled back
};

struct LmOptions {
    int    max_iterations = 50;
    double initial_lambda = 1e-4;
    double lambda_up      = 10.0;   // multiplier after a rejected step
    double lambda_down    = 0.1;    // multiplier after an accepted step
    double max_lambda     = 1e12;   // give up when lambda exceeds this
    double cost_tolerance = 1e-12;  // converged: relative cost decrease below this
    double step_tolerance = 1e-12;  // converged: ‖delta‖ below this

    std::function<void(const LmIterationSummary&)> iteration_callback;
};

struct LmResult {
    bool        converged    = false;
    int         iterations   = 0;  // outer iterations performed
    double      initial_cost = 0.0;
    double      final_cost   = 0.0;
    std::string message;
};

// Dense Levenberg-Marquardt with Marquardt scaling:
//   (H + lambda * diag(H)) delta = -b,   x ← x ⊞ delta if cost decreases.
// Deterministic; solves via LDLT. Diagonal entries are floored at 1e-12
// so pure-gradient directions stay bounded.
LmResult optimize(Problem& problem, const LmOptions& options = {});

}  // namespace slam_core::optim
