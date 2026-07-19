#pragma once

#include <Eigen/Core>

#include "slam_core/imu/preintegration.hpp"
#include "slam_core/optim/factor.hpp"
#include "slam_core/optim/variable.hpp"

namespace slam_core::factors {

// Preintegrated IMU factor between keyframes i and j (Forster TRO'17).
//
// Connected variables, in order:
//   R_W_Bi (SO3), v_W_Bi (3), p_W_Bi (3),
//   R_W_Bj (SO3), v_W_Bj (3), p_W_Bj (3),
//   bg_i (3, rad/s), ba_i (3, m/s²)
//
// The preintegration was integrated at linearization biases
// (bg_lin, ba_lin); the factor applies the first-order bias correction
// for the current bias estimates, so bias variables can move without
// re-integration.
//
// Unwhitened residuals (9: [r_dR, r_dv, r_dp], Forster ordering), with
// dt = preint.delta_t_s and corrected deltas dR~, dv~, dp~:
//   r_dR = log_so3(dR~ᵀ · R_iᵀ R_j)                              (rad)
//   r_dv = R_iᵀ (v_j − v_i − g_W·dt) − dv~                       (m/s)
//   r_dp = R_iᵀ (p_j − p_i − v_i·dt − ½ g_W·dt²) − dp~           (m)
//
// Residual and Jacobians are whitened by the square-root information of
// preint.covariance (computed once at construction).
//
// Throws std::invalid_argument if preint.covariance is not positive
// definite (integrate with nonzero ImuNoiseParams), delta_t_s <= 0, or
// gravity_W is non-finite.
class ImuFactor final : public optim::Factor {
public:
    ImuFactor(optim::So3Variable*          R_i,
              optim::VectorVariable*       v_i,
              optim::VectorVariable*       p_i,
              optim::So3Variable*          R_j,
              optim::VectorVariable*       v_j,
              optim::VectorVariable*       p_j,
              optim::VectorVariable*       bg_i,
              optim::VectorVariable*       ba_i,
              const imu::PreintegratedImu& preint,
              const Eigen::Vector3d&       gravity_W,
              const Eigen::Vector3d&       bg_lin,
              const Eigen::Vector3d&       ba_lin);

    int                          residual_dim() const override { return 9; }
    Eigen::VectorXd              residual() const override;
    std::vector<Eigen::MatrixXd> jacobians() const override;

private:
    Eigen::Vector3d delta_bg() const;
    Eigen::Vector3d delta_ba() const;

    const optim::So3Variable*    R_i_;
    const optim::VectorVariable* v_i_;
    const optim::VectorVariable* p_i_;
    const optim::So3Variable*    R_j_;
    const optim::VectorVariable* v_j_;
    const optim::VectorVariable* p_j_;
    const optim::VectorVariable* bg_i_;
    const optim::VectorVariable* ba_i_;

    imu::PreintegratedImu       preint_;
    Eigen::Vector3d             gravity_W_;
    Eigen::Vector3d             bg_lin_;
    Eigen::Vector3d             ba_lin_;
    Eigen::Matrix<double, 9, 9> sqrt_info_;  // Lᵀ with covariance⁻¹ = L·Lᵀ
};

// Gaussian random-walk factor between the bias estimates of consecutive
// keyframes: r = [(bg_j − bg_i)/σ_g; (ba_j − ba_i)/σ_a] ∈ R⁶, where
// σ = walk_density · sqrt(dt) (continuous-time walk densities, EuRoC
// sensor.yaml convention: rad/s²/sqrt(Hz), m/s³/sqrt(Hz)).
//
// Variables in order: bg_i, ba_i, bg_j, ba_j (each size 3).
class BiasRandomWalkFactor final : public optim::Factor {
public:
    BiasRandomWalkFactor(optim::VectorVariable* bg_i,
                         optim::VectorVariable* ba_i,
                         optim::VectorVariable* bg_j,
                         optim::VectorVariable* ba_j,
                         double                 gyro_walk_density,
                         double                 accel_walk_density,
                         double                 dt_s);

    int                          residual_dim() const override { return 6; }
    Eigen::VectorXd              residual() const override;
    std::vector<Eigen::MatrixXd> jacobians() const override;

private:
    const optim::VectorVariable* bg_i_;
    const optim::VectorVariable* ba_i_;
    const optim::VectorVariable* bg_j_;
    const optim::VectorVariable* ba_j_;
    double                       sigma_g_;  // rad/s
    double                       sigma_a_;  // m/s²
};

}  // namespace slam_core::factors
