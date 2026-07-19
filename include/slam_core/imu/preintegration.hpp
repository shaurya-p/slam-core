#pragma once

#include <vector>

#include <Eigen/Core>

#include "slam_core/imu/imu_measurement.hpp"

namespace slam_core::imu {

// Continuous-time IMU noise densities (EuRoC sensor.yaml convention).
// Discretized internally per step as sigma_d^2 = sigma^2 / dt.
// Zero densities are valid: covariance then stays zero.
struct ImuNoiseParams {
    double gyro_noise_density  = 0.0;  // rad/s/sqrt(Hz)
    double accel_noise_density = 0.0;  // m/s^2/sqrt(Hz)
};

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
//
// covariance: 9x9 of the error state [dtheta, dv, dp] (Forster ordering,
// design decision #5), propagated from ImuNoiseParams. dtheta is the local
// (right) rotation error: delta_R_true = delta_R * exp_so3(dtheta).
//
// Bias-correction Jacobians (about the linearization biases passed to
// integrate): d_delta_R_d_bg is log-space —
//   delta_R(bg + dbg) ≈ delta_R(bg) * exp_so3(d_delta_R_d_bg * dbg)
// while dv/dp Jacobians are additive. Use the delta_*_corrected helpers.
struct PreintegratedImu {
    Eigen::Matrix3d delta_R   = Eigen::Matrix3d::Identity();
    Eigen::Vector3d delta_v   = Eigen::Vector3d::Zero();
    Eigen::Vector3d delta_p   = Eigen::Vector3d::Zero();
    double          delta_t_s = 0.0;

    Eigen::Matrix<double, 9, 9> covariance = Eigen::Matrix<double, 9, 9>::Zero();

    Eigen::Matrix3d d_delta_R_d_bg = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d d_delta_v_d_bg = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d d_delta_v_d_ba = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d d_delta_p_d_bg = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d d_delta_p_d_ba = Eigen::Matrix3d::Zero();

    void reset();
};

// First-order bias-corrected deltas, for re-linearizing factors without
// re-integrating. delta_bg / delta_ba are deviations from the biases the
// preintegration was integrated with.
Eigen::Matrix3d delta_R_corrected(const PreintegratedImu& preint, const Eigen::Vector3d& delta_bg);
Eigen::Vector3d delta_v_corrected(const PreintegratedImu& preint,
                                  const Eigen::Vector3d&  delta_bg,
                                  const Eigen::Vector3d&  delta_ba);
Eigen::Vector3d delta_p_corrected(const PreintegratedImu& preint,
                                  const Eigen::Vector3d&  delta_bg,
                                  const Eigen::Vector3d&  delta_ba);

// Integrate one IMU measurement into preint. Deltas, covariance, and bias
// Jacobians are all propagated; with default (zero) noise the covariance
// stays zero and only deltas/Jacobians evolve.
//
// Throws std::invalid_argument if:
//   - dt_s is not positive and finite
//   - measurement, gyro_bias_radps, or accel_bias_mps2 contain non-finite values
//   - noise densities are negative or non-finite
void integrate(PreintegratedImu&      preint,
               const ImuMeasurement&  measurement,
               const Eigen::Vector3d& gyro_bias_radps,
               const Eigen::Vector3d& accel_bias_mps2,
               double                 dt_s,
               const ImuNoiseParams&  noise = ImuNoiseParams{});

// Integrate a contiguous sequence of timestamped IMU measurements into a fresh
// PreintegratedImu. dt for each step is derived from adjacent timestamps.
// N measurements produce N-1 integration steps (zeroth-order hold).
//
// Throws std::invalid_argument if:
//   - measurements has fewer than 2 elements
//   - any measurement contains non-finite values
//   - gyro_bias_radps or accel_bias_mps2 contain non-finite values
//   - any adjacent dt is not positive and finite (non-increasing timestamps)
PreintegratedImu integrate_sequence(const std::vector<ImuMeasurement>& measurements,
                                    const Eigen::Vector3d&             gyro_bias_radps,
                                    const Eigen::Vector3d&             accel_bias_mps2,
                                    const ImuNoiseParams&              noise = ImuNoiseParams{});

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
PreintegratedImu integrate_window(const std::vector<ImuMeasurement>& stream,
                                  double                             t_start_s,
                                  double                             t_end_s,
                                  const Eigen::Vector3d&             gyro_bias_radps,
                                  const Eigen::Vector3d&             accel_bias_mps2,
                                  const ImuNoiseParams&              noise = ImuNoiseParams{});

}  // namespace slam_core::imu
