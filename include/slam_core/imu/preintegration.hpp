#pragma once

#include <vector>

#include <Eigen/Core>

#include "slam_core/imu/imu_measurement.hpp"

namespace slam_core::imu {

// Body-frame IMU preintegration between two keyframe timestamps.
//
// Integrates raw IMU measurements after subtracting constant biases.
// Gravity is not included — it is handled analytically at the factor level.
//
// Zeroth-order hold update order per step (dt_s):
//   omega     = gyro_radps  - gyro_bias_radps
//   a         = accel_mps2  - accel_bias_mps2
//   delta_p  += delta_v * dt + 0.5 * (delta_R * a) * dt^2
//   delta_v  += delta_R * a * dt
//   delta_R   = delta_R * exp_so3(omega * dt)
//   delta_t_s += dt
struct PreintegratedImu {
    Eigen::Matrix3d delta_R   = Eigen::Matrix3d::Identity();
    Eigen::Vector3d delta_v   = Eigen::Vector3d::Zero();
    Eigen::Vector3d delta_p   = Eigen::Vector3d::Zero();
    double          delta_t_s = 0.0;

    void reset();
};

// Integrate one IMU measurement into preint.
//
// Throws std::invalid_argument if:
//   - dt_s is not positive and finite
//   - measurement, gyro_bias_radps, or accel_bias_mps2 contain non-finite values
void integrate(
    PreintegratedImu&      preint,
    const ImuMeasurement&  measurement,
    const Eigen::Vector3d& gyro_bias_radps,
    const Eigen::Vector3d& accel_bias_mps2,
    double                 dt_s);

// Integrate a contiguous sequence of timestamped IMU measurements into a fresh
// PreintegratedImu. dt for each step is derived from adjacent timestamps.
// N measurements produce N-1 integration steps (zeroth-order hold).
//
// Throws std::invalid_argument if:
//   - measurements has fewer than 2 elements
//   - any measurement contains non-finite values
//   - gyro_bias_radps or accel_bias_mps2 contain non-finite values
//   - any adjacent dt is not positive and finite (non-increasing timestamps)
PreintegratedImu integrate_sequence(
    const std::vector<ImuMeasurement>& measurements,
    const Eigen::Vector3d&             gyro_bias_radps,
    const Eigen::Vector3d&             accel_bias_mps2);

// Integrate all measurements from stream whose timestamps fall within
// [t_start_s, t_end_s] (inclusive on both ends).
//
// Boundary policy: contained measurements only — no interpolation is performed
// at either boundary. Measurements outside the window are excluded entirely.
// As a result, result.delta_t_s may be smaller than (t_end_s - t_start_s) if
// no measurements land exactly on the requested boundary times.
//
// Throws std::invalid_argument if:
//   - gyro_bias_radps or accel_bias_mps2 contain non-finite values
//   - t_start_s or t_end_s are non-finite
//   - t_start_s >= t_end_s
//   - stream is empty
//   - any measurement in stream contains non-finite values
//   - stream timestamps are not strictly increasing
//   - fewer than 2 measurements fall within [t_start_s, t_end_s]
PreintegratedImu integrate_window(
    const std::vector<ImuMeasurement>& stream,
    double                             t_start_s,
    double                             t_end_s,
    const Eigen::Vector3d&             gyro_bias_radps,
    const Eigen::Vector3d&             accel_bias_mps2);

}  // namespace slam_core::imu
