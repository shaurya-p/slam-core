#include "slam_core/factors/relative_pose_factor.hpp"

#include <stdexcept>

#include "slam_core/geometry/so3.hpp"

namespace slam_core::factors {

RelativePoseFactor::RelativePoseFactor(optim::So3Variable*    rotation_i,
                                       optim::VectorVariable* translation_i,
                                       optim::So3Variable*    rotation_j,
                                       optim::VectorVariable* translation_j,
                                       const Eigen::Matrix3d& R_meas,
                                       const Eigen::Vector3d& t_meas)
    : Factor({rotation_i, translation_i, rotation_j, translation_j}),
      rot_i_(rotation_i),
      trans_i_(translation_i),
      rot_j_(rotation_j),
      trans_j_(translation_j),
      R_meas_(R_meas),
      t_meas_(t_meas) {
    if (translation_i->tangent_dim() != 3 || translation_j->tangent_dim() != 3) {
        throw std::invalid_argument("RelativePoseFactor: translations must be size 3");
    }
    if (!geometry::is_valid_rotation(R_meas_)) {
        throw std::invalid_argument("RelativePoseFactor: R_meas is not a valid rotation");
    }
    if (!t_meas_.allFinite()) {
        throw std::invalid_argument("RelativePoseFactor: t_meas is not finite");
    }
}

Eigen::VectorXd RelativePoseFactor::residual() const {
    Eigen::VectorXd r(6);
    r.head<3>() = geometry::log_so3(R_meas_.transpose() * rot_i_->R().transpose() * rot_j_->R());
    r.tail<3>() = rot_i_->R().transpose() * (trans_j_->vec() - trans_i_->vec()) - t_meas_;
    return r;
}

std::vector<Eigen::MatrixXd> RelativePoseFactor::jacobians() const {
    const Eigen::Matrix3d R_i     = rot_i_->R();
    const Eigen::Matrix3d R_j     = rot_j_->R();
    const Eigen::Vector3d r_R     = geometry::log_so3(R_meas_.transpose() * R_i.transpose() * R_j);
    const Eigen::Matrix3d Jr_inv  = geometry::right_jacobian_so3_inverse(r_R);
    const Eigen::Vector3d t_i_rel = R_i.transpose() * (trans_j_->vec() - trans_i_->vec());

    Eigen::MatrixXd J_Ri = Eigen::MatrixXd::Zero(6, 3);
    J_Ri.topRows<3>()    = -Jr_inv * R_j.transpose() * R_i;
    J_Ri.bottomRows<3>() = geometry::skew(t_i_rel);

    Eigen::MatrixXd J_ti = Eigen::MatrixXd::Zero(6, 3);
    J_ti.bottomRows<3>() = -R_i.transpose();

    Eigen::MatrixXd J_Rj = Eigen::MatrixXd::Zero(6, 3);
    J_Rj.topRows<3>()    = Jr_inv;

    Eigen::MatrixXd J_tj = Eigen::MatrixXd::Zero(6, 3);
    J_tj.bottomRows<3>() = R_i.transpose();

    return {J_Ri, J_ti, J_Rj, J_tj};
}

}  // namespace slam_core::factors
