#pragma once

#include <vector>

#include <Eigen/Core>

#include "slam_core/optim/factor.hpp"

namespace slam_core::testing {

// Central-difference numeric Jacobians of a factor's residual with respect
// to each connected variable's tangent space, under that variable's retract
// convention. Variable states are saved and restored via value()/set_value(),
// so the factor is left unchanged.
inline std::vector<Eigen::MatrixXd> numeric_jacobians(const optim::Factor& factor,
                                                      double               h = 1e-6) {
    std::vector<Eigen::MatrixXd> result;
    const int                    m = factor.residual_dim();

    for (optim::Variable* var : factor.variables()) {
        const int             n      = var->tangent_dim();
        const Eigen::VectorXd backup = var->value();
        Eigen::MatrixXd       J(m, n);

        for (int i = 0; i < n; ++i) {
            Eigen::VectorXd delta = Eigen::VectorXd::Zero(n);

            delta(i) = h;
            var->retract(delta);
            const Eigen::VectorXd r_plus = factor.residual();
            var->set_value(backup);

            delta(i) = -h;
            var->retract(delta);
            const Eigen::VectorXd r_minus = factor.residual();
            var->set_value(backup);

            J.col(i) = (r_plus - r_minus) / (2.0 * h);
        }
        result.push_back(J);
    }
    return result;
}

// Max absolute difference between analytic and numeric Jacobians across all
// connected variables.
inline double max_jacobian_error(const optim::Factor& factor, double h = 1e-6) {
    const std::vector<Eigen::MatrixXd> analytic = factor.jacobians();
    const std::vector<Eigen::MatrixXd> numeric  = numeric_jacobians(factor, h);
    double                             max_err  = 0.0;
    for (std::size_t k = 0; k < analytic.size(); ++k) {
        max_err = std::max(max_err, (analytic[k] - numeric[k]).cwiseAbs().maxCoeff());
    }
    return max_err;
}

}  // namespace slam_core::testing
