#pragma once

#include <Eigen/Core>

namespace slam_core::geometry {

// w in R^3: axis-angle vector, ||w|| = rotation angle (rad), axis = w / ||w||

Eigen::Matrix3d skew(const Eigen::Vector3d& w);

// Rodrigues exponential map: w -> R in SO(3).
// Uses first-order approximation when ||w|| < 1e-8.
Eigen::Matrix3d exp_so3(const Eigen::Vector3d& w);

// SO(3) logarithm: R -> axis-angle w.
// acos input is clamped to [-1, 1]. Stable near theta = 0 and theta = pi.
Eigen::Vector3d log_so3(const Eigen::Matrix3d& R);

// Returns true if R is orthonormal with det(R) ≈ +1 within tol.
bool is_valid_rotation(const Eigen::Matrix3d& R, double tol = 1e-6);

}  // namespace slam_core::geometry
