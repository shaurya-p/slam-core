#include "slam_core/geometry/se3.hpp"

namespace slam_core::geometry {

SE3 SE3::compose(const SE3& other) const {
    return SE3{R * other.R, R * other.t + t};
}

SE3 SE3::inverse() const {
    const Eigen::Matrix3d R_inv = R.transpose();
    return SE3{R_inv, -(R_inv * t)};
}

Eigen::Vector3d SE3::apply(const Eigen::Vector3d& p) const {
    return R * p + t;
}

}  // namespace slam_core::geometry
