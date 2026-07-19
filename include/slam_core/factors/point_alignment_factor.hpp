#pragma once

#include <Eigen/Core>

#include "slam_core/optim/factor.hpp"
#include "slam_core/optim/variable.hpp"

namespace slam_core::factors {

// One point correspondence for rigid alignment: estimate T_A_B = (R, t)
// from pairs (p_B, q_A) with q_A ≈ R · p_B + t.
//
// r = (R · p_B + t) - q_A ∈ R³ (m)
// ∂r/∂δθ = -R · [p_B]×   (right perturbation R ← R·exp_so3(δθ))
// ∂r/∂δt = I
//
// Identity information (unwhitened).
class PointAlignmentFactor final : public optim::Factor {
public:
    PointAlignmentFactor(optim::So3Variable*    rotation,
                         optim::VectorVariable* translation,  // size 3
                         const Eigen::Vector3d& p_B,
                         const Eigen::Vector3d& q_A);

    int                          residual_dim() const override { return 3; }
    Eigen::VectorXd              residual() const override;
    std::vector<Eigen::MatrixXd> jacobians() const override;

private:
    const optim::So3Variable*    rot_;
    const optim::VectorVariable* trans_;
    Eigen::Vector3d              p_B_;
    Eigen::Vector3d              q_A_;
};

}  // namespace slam_core::factors
