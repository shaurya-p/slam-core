#include "slam_core/factors/point_alignment_factor.hpp"

#include <stdexcept>

#include "slam_core/geometry/so3.hpp"

namespace slam_core::factors {

PointAlignmentFactor::PointAlignmentFactor(optim::So3Variable*    rotation,
                                           optim::VectorVariable* translation,
                                           const Eigen::Vector3d& p_B,
                                           const Eigen::Vector3d& q_A)
    : Factor({rotation, translation}), rot_(rotation), trans_(translation), p_B_(p_B), q_A_(q_A) {
    if (translation->tangent_dim() != 3) {
        throw std::invalid_argument("PointAlignmentFactor: translation must be size 3");
    }
    if (!p_B_.allFinite() || !q_A_.allFinite()) {
        throw std::invalid_argument("PointAlignmentFactor: non-finite point");
    }
}

Eigen::VectorXd PointAlignmentFactor::residual() const {
    return rot_->R() * p_B_ + trans_->vec() - q_A_;
}

std::vector<Eigen::MatrixXd> PointAlignmentFactor::jacobians() const {
    return {-rot_->R() * geometry::skew(p_B_), Eigen::Matrix3d::Identity()};
}

}  // namespace slam_core::factors
