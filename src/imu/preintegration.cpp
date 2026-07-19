#include "slam_core/imu/preintegration.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

#include "slam_core/geometry/so3.hpp"

namespace slam_core::imu {

void PreintegratedImu::reset() {
    delta_R   = Eigen::Matrix3d::Identity();
    delta_v   = Eigen::Vector3d::Zero();
    delta_p   = Eigen::Vector3d::Zero();
    delta_t_s = 0.0;

    covariance = Eigen::Matrix<double, 9, 9>::Zero();

    d_delta_R_d_bg = Eigen::Matrix3d::Zero();
    d_delta_v_d_bg = Eigen::Matrix3d::Zero();
    d_delta_v_d_ba = Eigen::Matrix3d::Zero();
    d_delta_p_d_bg = Eigen::Matrix3d::Zero();
    d_delta_p_d_ba = Eigen::Matrix3d::Zero();
}

Eigen::Matrix3d delta_R_corrected(const PreintegratedImu& preint, const Eigen::Vector3d& delta_bg) {
    return preint.delta_R * slam_core::geometry::exp_so3(preint.d_delta_R_d_bg * delta_bg);
}

Eigen::Vector3d delta_v_corrected(const PreintegratedImu& preint,
                                  const Eigen::Vector3d&  delta_bg,
                                  const Eigen::Vector3d&  delta_ba) {
    return preint.delta_v + preint.d_delta_v_d_bg * delta_bg + preint.d_delta_v_d_ba * delta_ba;
}

Eigen::Vector3d delta_p_corrected(const PreintegratedImu& preint,
                                  const Eigen::Vector3d&  delta_bg,
                                  const Eigen::Vector3d&  delta_ba) {
    return preint.delta_p + preint.d_delta_p_d_bg * delta_bg + preint.d_delta_p_d_ba * delta_ba;
}

void integrate(PreintegratedImu&      preint,
               const ImuMeasurement&  measurement,
               const Eigen::Vector3d& gyro_bias_radps,
               const Eigen::Vector3d& accel_bias_mps2,
               double                 dt_s,
               const ImuNoiseParams&  noise) {
    if (!std::isfinite(dt_s) || dt_s <= 0.0) {
        throw std::invalid_argument("integrate: dt_s must be positive and finite");
    }
    if (!is_finite(measurement)) {
        throw std::invalid_argument("integrate: measurement contains non-finite values");
    }
    if (!gyro_bias_radps.allFinite()) {
        throw std::invalid_argument("integrate: gyro_bias_radps contains non-finite values");
    }
    if (!accel_bias_mps2.allFinite()) {
        throw std::invalid_argument("integrate: accel_bias_mps2 contains non-finite values");
    }
    if (!std::isfinite(noise.gyro_noise_density) || noise.gyro_noise_density < 0.0 ||
        !std::isfinite(noise.accel_noise_density) || noise.accel_noise_density < 0.0) {
        throw std::invalid_argument("integrate: noise densities must be finite and >= 0");
    }

    const Eigen::Vector3d omega = measurement.gyro_radps - gyro_bias_radps;
    const Eigen::Vector3d a     = measurement.accel_mps2 - accel_bias_mps2;

    // Pre-update quantities: every propagation term below is evaluated at
    // the state before this step (Forster TRO'17, eqs. 59-63).
    const Eigen::Matrix3d R      = preint.delta_R;  // delta_R_k
    const Eigen::Matrix3d a_hat  = slam_core::geometry::skew(a);
    const Eigen::Matrix3d R_step = slam_core::geometry::exp_so3(omega * dt_s);
    const Eigen::Matrix3d J_r    = slam_core::geometry::right_jacobian_so3(omega * dt_s);
    const double          dt2    = dt_s * dt_s;

    // --- covariance of [dtheta, dv, dp] ---
    // A: error-state transition; B_g/B_a: discrete noise input blocks.
    Eigen::Matrix<double, 9, 9> A = Eigen::Matrix<double, 9, 9>::Identity();
    A.block<3, 3>(0, 0)           = R_step.transpose();
    A.block<3, 3>(3, 0)           = -R * a_hat * dt_s;
    A.block<3, 3>(6, 0)           = -0.5 * R * a_hat * dt2;
    A.block<3, 3>(6, 3)           = Eigen::Matrix3d::Identity() * dt_s;

    // Discretization: sigma_d^2 = sigma^2 / dt (continuous density -> step).
    const double var_g = noise.gyro_noise_density * noise.gyro_noise_density / dt_s;
    const double var_a = noise.accel_noise_density * noise.accel_noise_density / dt_s;

    Eigen::Matrix<double, 9, 3> B_g = Eigen::Matrix<double, 9, 3>::Zero();
    B_g.block<3, 3>(0, 0)           = J_r * dt_s;
    Eigen::Matrix<double, 9, 3> B_a = Eigen::Matrix<double, 9, 3>::Zero();
    B_a.block<3, 3>(3, 0)           = R * dt_s;
    B_a.block<3, 3>(6, 0)           = 0.5 * R * dt2;

    preint.covariance = A * preint.covariance * A.transpose() + var_g * B_g * B_g.transpose() +
                        var_a * B_a * B_a.transpose();

    // --- bias-correction Jacobians (pre-update values on every RHS) ---
    preint.d_delta_p_d_ba += preint.d_delta_v_d_ba * dt_s - 0.5 * R * dt2;
    preint.d_delta_p_d_bg +=
        preint.d_delta_v_d_bg * dt_s - 0.5 * R * a_hat * preint.d_delta_R_d_bg * dt2;
    preint.d_delta_v_d_ba += -R * dt_s;
    preint.d_delta_v_d_bg += -R * a_hat * preint.d_delta_R_d_bg * dt_s;
    preint.d_delta_R_d_bg = R_step.transpose() * preint.d_delta_R_d_bg - J_r * dt_s;

    // --- deltas ---
    preint.delta_p += preint.delta_v * dt_s + 0.5 * (R * a) * dt2;
    preint.delta_v += R * a * dt_s;
    preint.delta_R = R * R_step;
    preint.delta_t_s += dt_s;
}

PreintegratedImu integrate_sequence(const std::vector<ImuMeasurement>& measurements,
                                    const Eigen::Vector3d&             gyro_bias_radps,
                                    const Eigen::Vector3d&             accel_bias_mps2,
                                    const ImuNoiseParams&              noise) {
    if (measurements.size() < 2) {
        throw std::invalid_argument(
            "integrate_sequence: need at least 2 measurements to form a dt");
    }
    if (!gyro_bias_radps.allFinite()) {
        throw std::invalid_argument(
            "integrate_sequence: gyro_bias_radps contains non-finite values");
    }
    if (!accel_bias_mps2.allFinite()) {
        throw std::invalid_argument(
            "integrate_sequence: accel_bias_mps2 contains non-finite values");
    }
    for (const auto& m : measurements) {
        if (!is_finite(m)) {
            throw std::invalid_argument(
                "integrate_sequence: measurement contains non-finite values");
        }
    }

    PreintegratedImu preint;
    for (std::size_t i = 0; i + 1 < measurements.size(); ++i) {
        const double dt = measurements[i + 1].timestamp_s - measurements[i].timestamp_s;
        // integrate() validates dt (positive, finite) and throws on failure.
        integrate(preint, measurements[i], gyro_bias_radps, accel_bias_mps2, dt, noise);
    }
    return preint;
}

PreintegratedImu integrate_window(const std::vector<ImuMeasurement>& stream,
                                  double                             t_start_s,
                                  double                             t_end_s,
                                  const Eigen::Vector3d&             gyro_bias_radps,
                                  const Eigen::Vector3d&             accel_bias_mps2,
                                  const ImuNoiseParams&              noise) {
    if (!gyro_bias_radps.allFinite()) {
        throw std::invalid_argument("integrate_window: gyro_bias_radps contains non-finite values");
    }
    if (!accel_bias_mps2.allFinite()) {
        throw std::invalid_argument("integrate_window: accel_bias_mps2 contains non-finite values");
    }
    if (!std::isfinite(t_start_s) || !std::isfinite(t_end_s)) {
        throw std::invalid_argument("integrate_window: t_start_s and t_end_s must be finite");
    }
    if (t_start_s >= t_end_s) {
        throw std::invalid_argument(
            "integrate_window: t_start_s must be strictly less than t_end_s");
    }
    if (stream.empty()) {
        throw std::invalid_argument("integrate_window: stream is empty");
    }
    for (const auto& m : stream) {
        if (!is_finite(m)) {
            throw std::invalid_argument(
                "integrate_window: stream contains a non-finite measurement");
        }
    }
    for (std::size_t i = 0; i + 1 < stream.size(); ++i) {
        if (stream[i + 1].timestamp_s <= stream[i].timestamp_s) {
            throw std::invalid_argument(
                "integrate_window: stream timestamps are not strictly increasing");
        }
    }

    std::vector<ImuMeasurement> window;
    for (const auto& m : stream) {
        if (m.timestamp_s >= t_start_s && m.timestamp_s <= t_end_s) {
            window.push_back(m);
        }
    }

    if (window.size() < 2) {
        throw std::invalid_argument(
            "integrate_window: fewer than 2 measurements in [t_start_s, t_end_s]");
    }

    return integrate_sequence(window, gyro_bias_radps, accel_bias_mps2, noise);
}

}  // namespace slam_core::imu
