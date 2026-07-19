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

// Right Jacobian of SO(3). Maps additive tangent increments to local
// (right-multiplied) group increments:
//   exp_so3(w + dw) ≈ exp_so3(w) * exp_so3(J_r(w) * dw)
//
// J_r(w) = I - (1 - cos θ)/θ² [w]× + (θ - sin θ)/θ³ [w]×²,  θ = ||w||
// Uses the Taylor expansion I - ½[w]× + ⅙[w]×² when θ < 1e-5.
Eigen::Matrix3d right_jacobian_so3(const Eigen::Vector3d& w);

// Inverse of the right Jacobian:
//   log_so3(exp_so3(w) * exp_so3(dw)) ≈ w + J_r⁻¹(w) * dw
//
// J_r⁻¹(w) = I + ½[w]× + (1/θ² - (1 + cos θ)/(2 θ sin θ)) [w]×²
// Uses the Taylor expansion I + ½[w]× + 1/12 [w]×² when θ < 1e-5, and the
// θ → π limit of the [w]×² coefficient when sin θ ≈ 0. Valid for θ < π.
Eigen::Matrix3d right_jacobian_so3_inverse(const Eigen::Vector3d& w);

}  // namespace slam_core::geometry
