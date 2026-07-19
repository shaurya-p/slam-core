#include "slam_core/optim/levenberg_marquardt.hpp"

#include <cmath>
#include <vector>

#include <Eigen/Cholesky>

namespace slam_core::optim {

namespace {

// Snapshot of all variable states, for rolling back rejected steps.
std::vector<Eigen::VectorXd> save_state(const Problem& problem) {
    std::vector<Eigen::VectorXd> state;
    state.reserve(problem.variables().size());
    for (const Variable* v : problem.variables()) state.push_back(v->value());
    return state;
}

void restore_state(Problem& problem, const std::vector<Eigen::VectorXd>& state) {
    for (std::size_t i = 0; i < state.size(); ++i) {
        problem.variables()[i]->set_value(state[i]);
    }
}

}  // namespace

LmResult optimize(Problem& problem, const LmOptions& options) {
    LmResult result;
    result.initial_cost = problem.cost();
    result.final_cost   = result.initial_cost;

    if (problem.total_tangent_dim() == 0 || problem.factors().empty()) {
        result.converged = true;
        result.message   = "nothing to optimize";
        return result;
    }

    double          lambda = options.initial_lambda;
    double          cost   = result.initial_cost;
    Eigen::MatrixXd H;
    Eigen::VectorXd b;

    for (int iter = 0; iter < options.max_iterations; ++iter) {
        const double cost_before_iter = cost;
        problem.build_normal_equations(H, b);

        // Marquardt scaling with a floored diagonal so directions with tiny
        // curvature stay bounded.
        Eigen::VectorXd diag = H.diagonal().cwiseMax(1e-12);

        bool   accepted  = false;
        double step_norm = 0.0;

        while (lambda <= options.max_lambda) {
            Eigen::MatrixXd H_damped = H;
            H_damped.diagonal() += lambda * diag;

            const Eigen::VectorXd delta = H_damped.ldlt().solve(-b);
            step_norm                   = delta.norm();
            if (!delta.allFinite()) {
                lambda *= options.lambda_up;
                continue;
            }

            // At a (local) optimum the step vanishes before it can decrease
            // the cost; converge on step size rather than escalating lambda.
            if (step_norm < options.step_tolerance) {
                result.iterations = iter + 1;
                result.final_cost = cost;
                result.converged  = true;
                result.message    = "step norm below tolerance";
                if (options.iteration_callback) {
                    options.iteration_callback({iter, cost, lambda, step_norm, false});
                }
                return result;
            }

            const std::vector<Eigen::VectorXd> backup = save_state(problem);
            problem.apply_step(delta);
            const double new_cost = problem.cost();

            if (std::isfinite(new_cost) && new_cost < cost) {
                cost     = new_cost;
                lambda   = std::max(lambda * options.lambda_down, 1e-15);
                accepted = true;
                break;
            }
            restore_state(problem, backup);
            lambda *= options.lambda_up;
        }

        result.iterations = iter + 1;
        result.final_cost = cost;

        if (options.iteration_callback) {
            options.iteration_callback({iter, cost, lambda, step_norm, accepted});
        }

        if (!accepted) {
            result.message = "lambda exceeded max_lambda without cost decrease";
            return result;
        }

        if (step_norm < options.step_tolerance) {
            result.converged = true;
            result.message   = "step norm below tolerance";
            return result;
        }
        const double relative_decrease =
            (cost_before_iter - cost) / std::max(cost_before_iter, 1e-300);
        if (relative_decrease < options.cost_tolerance) {
            result.converged = true;
            result.message   = "relative cost decrease below tolerance";
            return result;
        }
    }

    result.message = "max iterations reached";
    return result;
}

}  // namespace slam_core::optim
