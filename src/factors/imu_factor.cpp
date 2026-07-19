#include "slam_core/factors/imu_factor.hpp"

#include <cmath>
#include <stdexcept>

#include <Eigen/Cholesky>
#include <Eigen/LU>

#include "slam_core/geometry/so3.hpp"

namespace slam_core::factors {

namespace {

void check_size3(const optim::VectorVariable* v, const char* name) {
    if (v->tangent_dim() != 3) {
        throw std::invalid_argument(std::string("ImuFactor: ") + name + " must be size 3");
    }
}

}  // namespace

ImuFactor::ImuFactor(optim::So3Variable*          R_i,
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
                     const Eigen::Vector3d&       ba_lin)
    : Factor({R_i, v_i, p_i, R_j, v_j, p_j, bg_i, ba_i}),
      R_i_(R_i),
      v_i_(v_i),
      p_i_(p_i),
      R_j_(R_j),
      v_j_(v_j),
      p_j_(p_j),
      bg_i_(bg_i),
      ba_i_(ba_i),
      preint_(preint),
      gravity_W_(gravity_W),
      bg_lin_(bg_lin),
      ba_lin_(ba_lin) {
    check_size3(v_i, "v_i");
    check_size3(p_i, "p_i");
    check_size3(v_j, "v_j");
    check_size3(p_j, "p_j");
    check_size3(bg_i, "bg_i");
    check_size3(ba_i, "ba_i");
    if (!(preint_.delta_t_s > 0.0) || !std::isfinite(preint_.delta_t_s)) {
        throw std::invalid_argument("ImuFactor: preint.delta_t_s must be positive");
    }
    if (!gravity_W_.allFinite() || !bg_lin_.allFinite() || !ba_lin_.allFinite()) {
        throw std::invalid_argument("ImuFactor: non-finite gravity or linearization bias");
    }

    const Eigen::Matrix<double, 9, 9>             info = preint_.covariance.inverse();
    const Eigen::LLT<Eigen::Matrix<double, 9, 9>> llt(info);
    if (llt.info() != Eigen::Success || !info.allFinite()) {
        throw std::invalid_argument(
            "ImuFactor: preint.covariance is not positive definite; "
            "integrate with nonzero ImuNoiseParams");
    }
    sqrt_info_ = llt.matrixL().transpose();
}

Eigen::Vector3d ImuFactor::delta_bg() const {
    return bg_i_->vec() - bg_lin_;
}
Eigen::Vector3d ImuFactor::delta_ba() const {
    return ba_i_->vec() - ba_lin_;
}

Eigen::VectorXd ImuFactor::residual() const {
    const double          dt  = preint_.delta_t_s;
    const Eigen::Matrix3d R_i = R_i_->R();

    const Eigen::Matrix3d dR = imu::delta_R_corrected(preint_, delta_bg());
    const Eigen::Vector3d dv = imu::delta_v_corrected(preint_, delta_bg(), delta_ba());
    const Eigen::Vector3d dp = imu::delta_p_corrected(preint_, delta_bg(), delta_ba());

    Eigen::Matrix<double, 9, 1> r;
    r.segment<3>(0) = geometry::log_so3(dR.transpose() * R_i.transpose() * R_j_->R());
    r.segment<3>(3) = R_i.transpose() * (v_j_->vec() - v_i_->vec() - gravity_W_ * dt) - dv;
    r.segment<3>(6) = R_i.transpose() * (p_j_->vec() - p_i_->vec() - v_i_->vec() * dt -
                                         0.5 * gravity_W_ * dt * dt) -
                      dp;
    return sqrt_info_ * r;
}

std::vector<Eigen::MatrixXd> ImuFactor::jacobians() const {
    const double          dt  = preint_.delta_t_s;
    const Eigen::Matrix3d R_i = R_i_->R();
    const Eigen::Matrix3d R_j = R_j_->R();

    const Eigen::Vector3d dbg = delta_bg();
    const Eigen::Matrix3d dR  = imu::delta_R_corrected(preint_, dbg);

    const Eigen::Matrix3d E      = dR.transpose() * R_i.transpose() * R_j;
    const Eigen::Vector3d r_R    = geometry::log_so3(E);
    const Eigen::Matrix3d Jr_inv = geometry::right_jacobian_so3_inverse(r_R);

    const Eigen::Vector3d v_term = R_i.transpose() * (v_j_->vec() - v_i_->vec() - gravity_W_ * dt);
    const Eigen::Vector3d p_term = R_i.transpose() * (p_j_->vec() - p_i_->vec() - v_i_->vec() * dt -
                                                      0.5 * gravity_W_ * dt * dt);

    // Bias correction enters delta_R through exp(D_R·dbg); the chain rule
    // through that retraction contributes J_r(D_R·dbg)·D_R.
    const Eigen::Matrix3d J_corr =
        geometry::right_jacobian_so3(preint_.d_delta_R_d_bg * dbg) * preint_.d_delta_R_d_bg;

    Eigen::MatrixXd J_Ri   = Eigen::MatrixXd::Zero(9, 3);
    J_Ri.block<3, 3>(0, 0) = -Jr_inv * R_j.transpose() * R_i;
    J_Ri.block<3, 3>(3, 0) = geometry::skew(v_term);
    J_Ri.block<3, 3>(6, 0) = geometry::skew(p_term);

    Eigen::MatrixXd J_vi   = Eigen::MatrixXd::Zero(9, 3);
    J_vi.block<3, 3>(3, 0) = -R_i.transpose();
    J_vi.block<3, 3>(6, 0) = -R_i.transpose() * dt;

    Eigen::MatrixXd J_pi   = Eigen::MatrixXd::Zero(9, 3);
    J_pi.block<3, 3>(6, 0) = -R_i.transpose();

    Eigen::MatrixXd J_Rj   = Eigen::MatrixXd::Zero(9, 3);
    J_Rj.block<3, 3>(0, 0) = Jr_inv;

    Eigen::MatrixXd J_vj   = Eigen::MatrixXd::Zero(9, 3);
    J_vj.block<3, 3>(3, 0) = R_i.transpose();

    Eigen::MatrixXd J_pj   = Eigen::MatrixXd::Zero(9, 3);
    J_pj.block<3, 3>(6, 0) = R_i.transpose();

    Eigen::MatrixXd J_bg   = Eigen::MatrixXd::Zero(9, 3);
    J_bg.block<3, 3>(0, 0) = -Jr_inv * E.transpose() * J_corr;
    J_bg.block<3, 3>(3, 0) = -preint_.d_delta_v_d_bg;
    J_bg.block<3, 3>(6, 0) = -preint_.d_delta_p_d_bg;

    Eigen::MatrixXd J_ba   = Eigen::MatrixXd::Zero(9, 3);
    J_ba.block<3, 3>(3, 0) = -preint_.d_delta_v_d_ba;
    J_ba.block<3, 3>(6, 0) = -preint_.d_delta_p_d_ba;

    return {sqrt_info_ * J_Ri, sqrt_info_ * J_vi, sqrt_info_ * J_pi, sqrt_info_ * J_Rj,
            sqrt_info_ * J_vj, sqrt_info_ * J_pj, sqrt_info_ * J_bg, sqrt_info_ * J_ba};
}

BiasRandomWalkFactor::BiasRandomWalkFactor(optim::VectorVariable* bg_i,
                                           optim::VectorVariable* ba_i,
                                           optim::VectorVariable* bg_j,
                                           optim::VectorVariable* ba_j,
                                           double                 gyro_walk_density,
                                           double                 accel_walk_density,
                                           double                 dt_s)
    : Factor({bg_i, ba_i, bg_j, ba_j}), bg_i_(bg_i), ba_i_(ba_i), bg_j_(bg_j), ba_j_(ba_j) {
    if (!std::isfinite(dt_s) || dt_s <= 0.0) {
        throw std::invalid_argument("BiasRandomWalkFactor: dt_s must be positive and finite");
    }
    if (!std::isfinite(gyro_walk_density) || gyro_walk_density <= 0.0 ||
        !std::isfinite(accel_walk_density) || accel_walk_density <= 0.0) {
        throw std::invalid_argument(
            "BiasRandomWalkFactor: walk densities must be positive and finite");
    }
    sigma_g_ = gyro_walk_density * std::sqrt(dt_s);
    sigma_a_ = accel_walk_density * std::sqrt(dt_s);
}

Eigen::VectorXd BiasRandomWalkFactor::residual() const {
    Eigen::Matrix<double, 6, 1> r;
    r.segment<3>(0) = (bg_j_->vec() - bg_i_->vec()) / sigma_g_;
    r.segment<3>(3) = (ba_j_->vec() - ba_i_->vec()) / sigma_a_;
    return r;
}

std::vector<Eigen::MatrixXd> BiasRandomWalkFactor::jacobians() const {
    Eigen::MatrixXd J_bgi   = Eigen::MatrixXd::Zero(6, 3);
    J_bgi.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity() / sigma_g_;
    Eigen::MatrixXd J_bai   = Eigen::MatrixXd::Zero(6, 3);
    J_bai.block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity() / sigma_a_;
    Eigen::MatrixXd J_bgj   = Eigen::MatrixXd::Zero(6, 3);
    J_bgj.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() / sigma_g_;
    Eigen::MatrixXd J_baj   = Eigen::MatrixXd::Zero(6, 3);
    J_baj.block<3, 3>(3, 0) = Eigen::Matrix3d::Identity() / sigma_a_;
    return {J_bgi, J_bai, J_bgj, J_baj};
}

}  // namespace slam_core::factors
