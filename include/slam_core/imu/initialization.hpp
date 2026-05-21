#pragma once

#include <Eigen/Core>
#include <vector>

#include "slam_core/imu/imu_measurement.hpp"

namespace slam_core::imu {

// Estimates R_W_B from a short stationary IMU window using accelerometer measurements.
//
// At rest: accel_body ≈ −R_B_W * gravity_W (specific force in body frame).
// Returns the shortest-arc rotation R_W_B such that R_W_B * mean(accel_body) ≈ −gravity_W.
//
// Constrains roll and pitch only. Yaw is unobservable from gravity alone.
//
// Throws std::invalid_argument on:
//   - fewer than 2 measurements
//   - any non-finite measurement
//   - non-finite or near-zero gravity_W
//   - near-zero mean accelerometer vector
Eigen::Matrix3d estimate_R_W_B_from_stationary(
    const std::vector<ImuMeasurement>& measurements,
    const Eigen::Vector3d& gravity_W = Eigen::Vector3d(0.0, 0.0, -9.81));

}  // namespace slam_core::imu
