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
}

void integrate(
    PreintegratedImu&      preint,
    const ImuMeasurement&  measurement,
    const Eigen::Vector3d& gyro_bias_radps,
    const Eigen::Vector3d& accel_bias_mps2,
    double                 dt_s)
{
    if (!std::isfinite(dt_s) || dt_s <= 0.0) {
        throw std::invalid_argument(
            "integrate: dt_s must be positive and finite");
    }
    if (!is_finite(measurement)) {
        throw std::invalid_argument(
            "integrate: measurement contains non-finite values");
    }
    if (!gyro_bias_radps.allFinite()) {
        throw std::invalid_argument(
            "integrate: gyro_bias_radps contains non-finite values");
    }
    if (!accel_bias_mps2.allFinite()) {
        throw std::invalid_argument(
            "integrate: accel_bias_mps2 contains non-finite values");
    }

    const Eigen::Vector3d omega = measurement.gyro_radps - gyro_bias_radps;
    const Eigen::Vector3d a     = measurement.accel_mps2 - accel_bias_mps2;

    preint.delta_p  += preint.delta_v * dt_s + 0.5 * (preint.delta_R * a) * dt_s * dt_s;
    preint.delta_v  += preint.delta_R * a * dt_s;
    preint.delta_R   = preint.delta_R * slam_core::geometry::exp_so3(omega * dt_s);
    preint.delta_t_s += dt_s;
}

PreintegratedImu integrate_sequence(
    const std::vector<ImuMeasurement>& measurements,
    const Eigen::Vector3d&             gyro_bias_radps,
    const Eigen::Vector3d&             accel_bias_mps2)
{
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
        integrate(preint, measurements[i], gyro_bias_radps, accel_bias_mps2, dt);
    }
    return preint;
}

}  // namespace slam_core::imu
