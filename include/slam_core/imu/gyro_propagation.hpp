#pragma once

#include <Eigen/Core>

namespace slam_core::imu {

// Single gyro integration step on SO(3).
//
// R_W_B_next = R_W_B * exp_so3(gyro_radps * dt_s)
//
// R_W_B      — rotation mapping body frame B into world frame W
// gyro_radps — angular velocity in the body frame (rad/s)
// dt_s       — time step in seconds; must be > 0 and finite
//
// Throws std::invalid_argument if dt_s <= 0 or dt_s is not finite.
Eigen::Matrix3d propagate_gyro(
    const Eigen::Matrix3d& R_W_B,
    const Eigen::Vector3d& gyro_radps,
    double dt_s);

// Single gyro integration step on SO(3) with explicit gyro bias correction.
//
// R_W_B_next = R_W_B * exp_so3((gyro_radps - gyro_bias_radps) * dt_s)
//
// R_W_B           — rotation mapping body frame B into world frame W
// gyro_radps      — angular velocity in the body frame (rad/s)
// gyro_bias_radps — body-frame gyro bias (rad/s)
// dt_s            — time step in seconds; must be > 0 and finite
//
// Throws std::invalid_argument if dt_s <= 0 or dt_s is not finite.
Eigen::Matrix3d propagate_gyro_bias_corrected(
    const Eigen::Matrix3d& R_W_B,
    const Eigen::Vector3d& gyro_radps,
    const Eigen::Vector3d& gyro_bias_radps,
    double dt_s);

}  // namespace slam_core::imu
