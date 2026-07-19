#pragma once

#include <Eigen/Core>

#include "slam_core/optim/factor.hpp"
#include "slam_core/optim/variable.hpp"

namespace slam_core::factors {

// Relative pose measurement between poses i and j (both T_W_B):
// measured rotation R_meas = R_iᵀ R_j and body-i-frame translation
// t_meas = R_iᵀ (t_j - t_i).
//
// Residuals (6 total: rotation stacked over translation):
//   r_R = log_so3(R_measᵀ · R_iᵀ · R_j) ∈ R³ (rad)
//   r_t = R_iᵀ (t_j - t_i) - t_meas    ∈ R³ (m)
//
// Jacobians (right perturbation R ← R·exp_so3(δθ)):
//   ∂r_R/∂δθ_i = -J_r⁻¹(r_R) · R_jᵀ R_i      ∂r_R/∂δθ_j = J_r⁻¹(r_R)
//   ∂r_t/∂δθ_i = [R_iᵀ (t_j - t_i)]×          ∂r_t/∂δθ_j = 0
//   ∂r_t/∂δt_i = -R_iᵀ                        ∂r_t/∂δt_j = R_iᵀ
//
// Variable order: (R_i, t_i, R_j, t_j). Identity information (unwhitened).
class RelativePoseFactor final : public optim::Factor {
public:
    RelativePoseFactor(optim::So3Variable*    rotation_i,
                       optim::VectorVariable* translation_i,  // size 3
                       optim::So3Variable*    rotation_j,
                       optim::VectorVariable* translation_j,  // size 3
                       const Eigen::Matrix3d& R_meas,
                       const Eigen::Vector3d& t_meas);

    int                          residual_dim() const override { return 6; }
    Eigen::VectorXd              residual() const override;
    std::vector<Eigen::MatrixXd> jacobians() const override;

private:
    const optim::So3Variable*    rot_i_;
    const optim::VectorVariable* trans_i_;
    const optim::So3Variable*    rot_j_;
    const optim::VectorVariable* trans_j_;
    Eigen::Matrix3d              R_meas_;
    Eigen::Vector3d              t_meas_;
};

}  // namespace slam_core::factors
