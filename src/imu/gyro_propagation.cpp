#include "slam_core/imu/gyro_propagation.hpp"

#include <cmath>
#include <stdexcept>

#include "slam_core/geometry/so3.hpp"

namespace slam_core::imu {

Eigen::Matrix3d propagate_gyro(const Eigen::Matrix3d& R_W_B,
                               const Eigen::Vector3d& gyro_radps,
                               double                 dt_s,
                               const Eigen::Vector3d& gyro_bias_radps) {
    if (!std::isfinite(dt_s) || dt_s <= 0.0) {
        throw std::invalid_argument("propagate_gyro: dt_s must be positive and finite");
    }
    // gyro is body-frame, so the rotation increment is right-multiplied.
    return R_W_B * slam_core::geometry::exp_so3((gyro_radps - gyro_bias_radps) * dt_s);
}

}  // namespace slam_core::imu
