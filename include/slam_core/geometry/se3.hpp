#pragma once

#include <Eigen/Core>

namespace slam_core::geometry {

// T_A_B: rigid transform that maps points from frame B into frame A.
// p_A = R * p_B + t
//
// Composition:  T_A_C = T_A_B.compose(T_B_C)
// Inverse:      T_B_A = T_A_B.inverse()
// Apply:        p_A   = T_A_B.apply(p_B)
struct SE3 {
    Eigen::Matrix3d R;  // rotation:    R_A_B
    Eigen::Vector3d t;  // translation: t expressed in frame A

    SE3 compose(const SE3& other) const;
    SE3 inverse() const;
    Eigen::Vector3d apply(const Eigen::Vector3d& p) const;
};

}  // namespace slam_core::geometry
