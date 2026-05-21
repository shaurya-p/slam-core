#include "slam_core/imu/initialization.hpp"

#include <Eigen/Geometry>
#include <cmath>
#include <stdexcept>

#include "slam_core/geometry/so3.hpp"

namespace slam_core::imu {

Eigen::Matrix3d estimate_R_W_B_from_stationary(
    const std::vector<ImuMeasurement>& measurements,
    const Eigen::Vector3d&             gravity_W)
{
    if (measurements.size() < 2) {
        throw std::invalid_argument(
            "estimate_R_W_B_from_stationary: need at least 2 measurements");
    }
    for (const auto& m : measurements) {
        if (!is_finite(m)) {
            throw std::invalid_argument(
                "estimate_R_W_B_from_stationary: non-finite measurement");
        }
    }
    if (!gravity_W.allFinite()) {
        throw std::invalid_argument(
            "estimate_R_W_B_from_stationary: gravity_W is non-finite");
    }
    if (gravity_W.norm() < 1e-6) {
        throw std::invalid_argument(
            "estimate_R_W_B_from_stationary: gravity_W is near-zero");
    }

    Eigen::Vector3d accel_sum = Eigen::Vector3d::Zero();
    for (const auto& m : measurements) {
        accel_sum += m.accel_mps2;
    }
    const Eigen::Vector3d accel_mean = accel_sum / static_cast<double>(measurements.size());

    if (accel_mean.norm() < 1e-3) {
        throw std::invalid_argument(
            "estimate_R_W_B_from_stationary: mean accelerometer norm is near-zero");
    }

    // At rest: accel_body ≈ −R_B_W * gravity_W, so R_W_B * accel_body ≈ −gravity_W.
    // Find shortest-arc rotation mapping a → b, where:
    //   a = accel_mean.normalized()  (body frame, source)
    //   b = (−gravity_W).normalized() (world frame, target)
    const Eigen::Vector3d a = accel_mean.normalized();
    const Eigen::Vector3d b = (-gravity_W).normalized();

    const Eigen::Vector3d v      = a.cross(b);
    const double          c      = a.dot(b);
    const double          v_norm = v.norm();

    Eigen::Matrix3d R;

    if (v_norm < 1e-8 && c > 0.0) {
        // Already aligned.
        R = Eigen::Matrix3d::Identity();
    } else if (v_norm < 1e-8 /* && c <= 0.0 */) {
        // Antiparallel: 180° rotation around a stable axis perpendicular to a.
        // Pick the world axis least parallel to a to minimise cancellation.
        Eigen::Vector3d ref;
        if (std::abs(a.x()) <= std::abs(a.y()) && std::abs(a.x()) <= std::abs(a.z())) {
            ref = Eigen::Vector3d::UnitX();
        } else if (std::abs(a.y()) <= std::abs(a.z())) {
            ref = Eigen::Vector3d::UnitY();
        } else {
            ref = Eigen::Vector3d::UnitZ();
        }
        const Eigen::Vector3d axis = (ref - a * a.dot(ref)).normalized();
        R = 2.0 * axis * axis.transpose() - Eigen::Matrix3d::Identity();
    } else {
        // Shortest-arc Rodrigues: R = I + [v]× + [v]×² / (1 + c)
        const Eigen::Matrix3d vx = geometry::skew(v);
        R = Eigen::Matrix3d::Identity() + vx + vx * vx / (1.0 + c);
    }

    if (!geometry::is_valid_rotation(R)) {
        throw std::runtime_error(
            "estimate_R_W_B_from_stationary: internal error, result is not a valid rotation");
    }

    return R;
}

}  // namespace slam_core::imu
