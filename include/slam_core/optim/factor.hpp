#pragma once

#include <vector>

#include <Eigen/Core>

#include "slam_core/optim/variable.hpp"

namespace slam_core::optim {

// One residual block r(x) ∈ R^residual_dim() connecting one or more
// variables. Cost contribution: ½ ‖r‖².
//
// Noise weighting is the factor's responsibility: residual() and
// jacobians() must return pre-whitened quantities (FG-1 factors use
// identity information; whitening arrives with the IMU factor).
//
// jacobians()[k] = ∂r/∂delta_k ∈ R^{m×n_k}: the derivative of the residual
// with respect to the k-th connected variable's tangent-space increment,
// evaluated at delta = 0 for the current variable states, under that
// variable's retract convention (So3Variable: R ← R·exp_so3(delta)).
class Factor {
public:
    virtual ~Factor() = default;

    virtual int residual_dim() const = 0;

    virtual Eigen::VectorXd              residual() const  = 0;
    virtual std::vector<Eigen::MatrixXd> jacobians() const = 0;

    const std::vector<Variable*>& variables() const { return variables_; }

protected:
    // Throws std::invalid_argument if any variable pointer is null.
    explicit Factor(std::vector<Variable*> variables);

    std::vector<Variable*> variables_;
};

}  // namespace slam_core::optim
