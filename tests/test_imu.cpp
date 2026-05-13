#include <gtest/gtest.h>
#include <Eigen/Core>
#include <limits>

#include "slam_core/imu/imu_measurement.hpp"

using slam_core::imu::ImuMeasurement;
using slam_core::imu::is_finite;
using slam_core::imu::has_strictly_increasing_timestamp;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

ImuMeasurement valid_at(double t) {
    return ImuMeasurement{t, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
}

}  // namespace

// --- is_finite ---

TEST(ImuMeasurement, ValidFiniteReturnsTrue) {
    EXPECT_TRUE(is_finite(valid_at(1.0)));
}

TEST(ImuMeasurement, NaNTimestampReturnsFalse) {
    EXPECT_FALSE(is_finite(valid_at(kNaN)));
}

TEST(ImuMeasurement, InfAccelReturnsFalse) {
    ImuMeasurement m = valid_at(1.0);
    m.accel_mps2.x() = kInf;
    EXPECT_FALSE(is_finite(m));
}

TEST(ImuMeasurement, NaNGyroReturnsFalse) {
    ImuMeasurement m = valid_at(1.0);
    m.gyro_radps.z() = kNaN;
    EXPECT_FALSE(is_finite(m));
}

// --- has_strictly_increasing_timestamp ---

TEST(ImuTimestamp, StrictlyIncreasingPasses) {
    EXPECT_TRUE(has_strictly_increasing_timestamp(valid_at(1.0), valid_at(2.0)));
}

TEST(ImuTimestamp, EqualTimestampFails) {
    EXPECT_FALSE(has_strictly_increasing_timestamp(valid_at(1.0), valid_at(1.0)));
}

TEST(ImuTimestamp, EarlierTimestampFails) {
    EXPECT_FALSE(has_strictly_increasing_timestamp(valid_at(2.0), valid_at(1.0)));
}
