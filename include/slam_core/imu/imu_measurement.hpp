#pragma once

#include <Eigen/Core>

namespace slam_core::imu {

// One timestamped IMU sample from a combined accelerometer + gyroscope.
// Frame: IMU body frame (right-hand, caller defines orientation).
// Timestamp origin: caller-defined (e.g. POSIX epoch or mission start).
struct ImuMeasurement {
    double          timestamp_s;  // seconds
    Eigen::Vector3d accel_mps2;   // m/s², body frame
    Eigen::Vector3d gyro_radps;   // rad/s, body frame
};

// Returns true if all scalar fields are finite (no NaN, no Inf).
bool is_finite(const ImuMeasurement& m);

// Returns true if current.timestamp_s > previous.timestamp_s.
bool has_strictly_increasing_timestamp(const ImuMeasurement& previous,
                                       const ImuMeasurement& current);

}  // namespace slam_core::imu
