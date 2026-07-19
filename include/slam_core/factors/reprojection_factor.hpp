#pragma once

#include <Eigen/Core>

#include "slam_core/camera/pinhole_camera.hpp"
#include "slam_core/optim/factor.hpp"
#include "slam_core/optim/variable.hpp"

namespace slam_core::factors {

// Reprojection of a world landmark into a body-mounted pinhole camera.
//
// Connected variables, in order:
//   R_W_B (SO3), p_W_B (3)  — body pose
//   p_W_L (3)               — landmark position in world frame (m)
//
// Fixed camera extrinsics T_B_C (EuRoC sensor.yaml T_BS convention):
//   p_B = R_B_C · p_C + t_B_C
//
// Projection chain:
//   p_B = R_W_Bᵀ (p_W_L − p_W_B),   p_C = R_B_Cᵀ (p_B − t_B_C)
//   r   = (observed_px − π(p_C)) / sigma_px ∈ R²
// (matches PinholeCamera::reprojection_error: observed − predicted).
//
// Behind-camera policy: if p_C.z() <= z_min the factor returns a large
// constant residual with zero Jacobians. LM then rejects any step that
// pushes the point behind the camera, without exceptions unwinding the
// solve. A state initialized behind the camera has zero gradient and will
// not recover — the caller must initialize with positive depth.
//
// Throws std::invalid_argument on invalid extrinsics/sigma/observation.
class ReprojectionFactor final : public optim::Factor {
public:
    ReprojectionFactor(optim::So3Variable*          R_W_B,
                       optim::VectorVariable*       p_W_B,
                       optim::VectorVariable*       p_W_L,
                       const camera::PinholeCamera& cam,
                       const Eigen::Matrix3d&       R_B_C,
                       const Eigen::Vector3d&       t_B_C,
                       const Eigen::Vector2d&       observed_px,
                       double                       sigma_px = 1.0);

    int                          residual_dim() const override { return 2; }
    Eigen::VectorXd              residual() const override;
    std::vector<Eigen::MatrixXd> jacobians() const override;

private:
    Eigen::Vector3d point_in_camera() const;

    const optim::So3Variable*    R_W_B_;
    const optim::VectorVariable* p_W_B_;
    const optim::VectorVariable* p_W_L_;
    camera::PinholeCamera        cam_;
    Eigen::Matrix3d              R_B_C_;
    Eigen::Vector3d              t_B_C_;
    Eigen::Vector2d              observed_px_;
    double                       sigma_px_;

    static constexpr double kZMin           = 1e-6;
    static constexpr double kInfeasibleCost = 1e6;  // px, per component
};

}  // namespace slam_core::factors
