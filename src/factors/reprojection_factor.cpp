#include "slam_core/factors/reprojection_factor.hpp"

#include <cmath>
#include <stdexcept>

#include "slam_core/geometry/so3.hpp"

namespace slam_core::factors {

ReprojectionFactor::ReprojectionFactor(optim::So3Variable*          R_W_B,
                                       optim::VectorVariable*       p_W_B,
                                       optim::VectorVariable*       p_W_L,
                                       const camera::PinholeCamera& cam,
                                       const Eigen::Matrix3d&       R_B_C,
                                       const Eigen::Vector3d&       t_B_C,
                                       const Eigen::Vector2d&       observed_px,
                                       double                       sigma_px)
    : Factor({R_W_B, p_W_B, p_W_L}),
      R_W_B_(R_W_B),
      p_W_B_(p_W_B),
      p_W_L_(p_W_L),
      cam_(cam),
      R_B_C_(R_B_C),
      t_B_C_(t_B_C),
      observed_px_(observed_px),
      sigma_px_(sigma_px) {
    if (p_W_B->tangent_dim() != 3 || p_W_L->tangent_dim() != 3) {
        throw std::invalid_argument("ReprojectionFactor: p_W_B and p_W_L must be size 3");
    }
    if (!geometry::is_valid_rotation(R_B_C_)) {
        throw std::invalid_argument("ReprojectionFactor: R_B_C is not a valid rotation");
    }
    if (!t_B_C_.allFinite() || !observed_px_.allFinite()) {
        throw std::invalid_argument("ReprojectionFactor: non-finite extrinsics or observation");
    }
    if (!std::isfinite(sigma_px_) || sigma_px_ <= 0.0) {
        throw std::invalid_argument("ReprojectionFactor: sigma_px must be positive and finite");
    }
}

Eigen::Vector3d ReprojectionFactor::point_in_camera() const {
    const Eigen::Vector3d p_B = R_W_B_->R().transpose() * (p_W_L_->vec() - p_W_B_->vec());
    return R_B_C_.transpose() * (p_B - t_B_C_);
}

Eigen::VectorXd ReprojectionFactor::residual() const {
    const Eigen::Vector3d p_C = point_in_camera();
    Eigen::Vector2d       predicted;
    if (!cam_.try_project(p_C, predicted, kZMin)) {
        return Eigen::Vector2d::Constant(kInfeasibleCost);
    }
    return (observed_px_ - predicted) / sigma_px_;
}

std::vector<Eigen::MatrixXd> ReprojectionFactor::jacobians() const {
    const Eigen::Vector3d p_C = point_in_camera();
    if (!(p_C.z() > kZMin)) {
        return {Eigen::MatrixXd::Zero(2, 3), Eigen::MatrixXd::Zero(2, 3),
                Eigen::MatrixXd::Zero(2, 3)};
    }

    // r = (observed - pi(p_C)) / sigma  =>  J = -(1/sigma) * dpi/dp_C * dp_C/dx
    const Eigen::Matrix<double, 2, 3> J_pi = -cam_.project_jacobian(p_C) / sigma_px_;

    const Eigen::Matrix3d R_C_B = R_B_C_.transpose();
    const Eigen::Vector3d p_B   = R_W_B_->R().transpose() * (p_W_L_->vec() - p_W_B_->vec());
    const Eigen::Matrix3d R_C_W = R_C_B * R_W_B_->R().transpose();

    // Right perturbation R_W_B <- R_W_B exp(dtheta): dp_B/ddtheta = [p_B]x.
    return {J_pi * R_C_B * geometry::skew(p_B), J_pi * (-R_C_W), J_pi * R_C_W};
}

}  // namespace slam_core::factors
