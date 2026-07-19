#pragma once

#include <Eigen/Core>

#include "slam_core/optim/factor.hpp"
#include "slam_core/optim/variable.hpp"

namespace slam_core::factors {

// Prior on an SO(3) variable.
//
// r = log_so3(R_priorᵀ · R) / sigma ∈ R³ (rad / sigma)
// ∂r/∂δθ = J_r⁻¹(r·sigma) / sigma   (right perturbation R ← R·exp_so3(δθ))
//
// sigma: isotropic standard deviation (rad); 1.0 = identity information.
class So3PriorFactor final : public optim::Factor {
public:
    So3PriorFactor(optim::So3Variable*    variable,
                   const Eigen::Matrix3d& R_prior,
                   double                 sigma = 1.0);

    int                          residual_dim() const override { return 3; }
    Eigen::VectorXd              residual() const override;
    std::vector<Eigen::MatrixXd> jacobians() const override;

private:
    const optim::So3Variable* var_;
    Eigen::Matrix3d           R_prior_;
    double                    sigma_;
};

// Prior on a Euclidean variable.
//
// r = (v - v_prior) / sigma,  ∂r/∂δv = I / sigma.
// sigma: isotropic standard deviation (variable units); 1.0 = identity info.
class VectorPriorFactor final : public optim::Factor {
public:
    VectorPriorFactor(optim::VectorVariable* variable,
                      const Eigen::VectorXd& prior,
                      double                 sigma = 1.0);

    int                          residual_dim() const override;
    Eigen::VectorXd              residual() const override;
    std::vector<Eigen::MatrixXd> jacobians() const override;

private:
    const optim::VectorVariable* var_;
    Eigen::VectorXd              prior_;
    double                       sigma_;
};

}  // namespace slam_core::factors
