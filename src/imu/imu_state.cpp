#include "slam_core/imu/imu_state.hpp"

#include <cmath>
#include <stdexcept>

#include "slam_core/geometry/so3.hpp"

namespace slam_core::imu {

static bool imu_state_is_finite(const ImuState& s) {
    return std::isfinite(s.timestamp_s) && s.R_W_B.allFinite() && s.p_W_B.allFinite() &&
           s.v_W_B.allFinite() && s.gyro_bias_radps.allFinite() && s.accel_bias_mps2.allFinite();
}

ImuState propagate_imu_state(const ImuState&        state,
                             const ImuMeasurement&  measurement,
                             const Eigen::Vector3d& gravity_W,
                             double                 dt_s) {
    if (!std::isfinite(dt_s) || dt_s <= 0.0) {
        throw std::invalid_argument("propagate_imu_state: dt_s must be positive and finite");
    }
    if (!imu_state_is_finite(state)) {
        throw std::invalid_argument("propagate_imu_state: state contains non-finite values");
    }
    if (!is_finite(measurement)) {
        throw std::invalid_argument("propagate_imu_state: measurement contains non-finite values");
    }
    if (!gravity_W.allFinite()) {
        throw std::invalid_argument("propagate_imu_state: gravity_W contains non-finite values");
    }

    const Eigen::Vector3d omega_B = measurement.gyro_radps - state.gyro_bias_radps;
    const Eigen::Vector3d a_B     = measurement.accel_mps2 - state.accel_bias_mps2;
    const Eigen::Vector3d a_W     = state.R_W_B * a_B + gravity_W;

    ImuState next;
    next.timestamp_s     = state.timestamp_s + dt_s;
    next.R_W_B           = state.R_W_B * slam_core::geometry::exp_so3(omega_B * dt_s);
    next.v_W_B           = state.v_W_B + a_W * dt_s;
    next.p_W_B           = state.p_W_B + state.v_W_B * dt_s + 0.5 * a_W * dt_s * dt_s;
    next.gyro_bias_radps = state.gyro_bias_radps;
    next.accel_bias_mps2 = state.accel_bias_mps2;
    return next;
}

}  // namespace slam_core::imu
