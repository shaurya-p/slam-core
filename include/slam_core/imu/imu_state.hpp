#pragma once

#include <Eigen/Core>

#include "slam_core/imu/imu_measurement.hpp"

namespace slam_core::imu {

// Full IMU navigation state.
//
// R_W_B         — rotation mapping body frame B into world frame W
// p_W_B         — body position expressed in world frame (m)
// v_W_B         — body velocity expressed in world frame (m/s)
// gyro_bias_radps  — body-frame gyro bias (rad/s)
// accel_bias_mps2  — body-frame accelerometer bias (m/s²)
struct ImuState {
    double          timestamp_s;
    Eigen::Matrix3d R_W_B;
    Eigen::Vector3d p_W_B;
    Eigen::Vector3d v_W_B;
    Eigen::Vector3d gyro_bias_radps;
    Eigen::Vector3d accel_bias_mps2;
};

// Single-step forward propagation of IMU state.
//
// Propagation model (zeroth-order hold):
//   omega_B = measurement.gyro_radps - state.gyro_bias_radps
//   a_B     = measurement.accel_mps2 - state.accel_bias_mps2
//   a_W     = state.R_W_B * a_B + gravity_W
//
//   R_next  = state.R_W_B * exp_so3(omega_B * dt_s)
//   v_next  = state.v_W_B + a_W * dt_s
//   p_next  = state.p_W_B + state.v_W_B * dt_s + 0.5 * a_W * dt_s^2
//
// Biases are copied forward unchanged.
// gravity_W — gravity acceleration in world frame, e.g. [0, 0, -9.81] m/s²
//
// Throws std::invalid_argument if:
//   - dt_s is not positive and finite
//   - any field of state, measurement, or gravity_W is not finite
ImuState propagate_imu_state(
    const ImuState&        state,
    const ImuMeasurement&  measurement,
    const Eigen::Vector3d& gravity_W,
    double                 dt_s);

}  // namespace slam_core::imu
