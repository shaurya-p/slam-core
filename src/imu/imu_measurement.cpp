#include "slam_core/imu/imu_measurement.hpp"

#include <cmath>

namespace slam_core::imu {

bool is_finite(const ImuMeasurement& m) {
    return std::isfinite(m.timestamp_s)
        && m.accel_mps2.allFinite()
        && m.gyro_radps.allFinite();
}

bool has_strictly_increasing_timestamp(const ImuMeasurement& previous,
                                       const ImuMeasurement& current) {
    return current.timestamp_s > previous.timestamp_s;
}

}  // namespace slam_core::imu
