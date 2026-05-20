#include <gtest/gtest.h>
#include <Eigen/Core>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "slam_core/imu/preintegration.hpp"
#include "slam_core/geometry/so3.hpp"

using slam_core::imu::PreintegratedImu;
using slam_core::imu::ImuMeasurement;
using slam_core::imu::integrate;
using slam_core::geometry::exp_so3;

// --- helpers ---

static ImuMeasurement make_meas(
    double gx = 0.0, double gy = 0.0, double gz = 0.0,
    double ax = 0.0, double ay = 0.0, double az = 0.0)
{
    ImuMeasurement m;
    m.timestamp_s = 0.0;
    m.gyro_radps  = {gx, gy, gz};
    m.accel_mps2  = {ax, ay, az};
    return m;
}

// --- 1. default construction and reset ---

TEST(Preintegration, DefaultConstructed) {
    PreintegratedImu preint;
    EXPECT_TRUE(preint.delta_R.isApprox(Eigen::Matrix3d::Identity(), 1e-12));
    EXPECT_TRUE(preint.delta_v.isApprox(Eigen::Vector3d::Zero(), 1e-12));
    EXPECT_TRUE(preint.delta_p.isApprox(Eigen::Vector3d::Zero(), 1e-12));
    EXPECT_DOUBLE_EQ(preint.delta_t_s, 0.0);
}

TEST(Preintegration, ResetRestoresInitialState) {
    PreintegratedImu preint;
    integrate(preint, make_meas(0.1, 0.2, 0.3, 1.0, 2.0, 3.0),
              Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), 0.05);

    preint.reset();

    EXPECT_TRUE(preint.delta_R.isApprox(Eigen::Matrix3d::Identity(), 1e-12));
    EXPECT_TRUE(preint.delta_v.isApprox(Eigen::Vector3d::Zero(), 1e-12));
    EXPECT_TRUE(preint.delta_p.isApprox(Eigen::Vector3d::Zero(), 1e-12));
    EXPECT_DOUBLE_EQ(preint.delta_t_s, 0.0);
}

// --- 2. zero net motion: delta_t_s accumulates, R/v/p stay at identity/zero ---

TEST(Preintegration, ZeroNetMotionAccumulatesTime) {
    PreintegratedImu preint;
    const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
    const double dt = 0.01;
    const int steps = 100;

    for (int i = 0; i < steps; ++i) {
        integrate(preint, make_meas(), zero, zero, dt);
    }

    EXPECT_NEAR(preint.delta_t_s, steps * dt, 1e-9);
    EXPECT_TRUE(preint.delta_R.isApprox(Eigen::Matrix3d::Identity(), 1e-9));
    EXPECT_TRUE(preint.delta_v.isApprox(Eigen::Vector3d::Zero(), 1e-9));
    EXPECT_TRUE(preint.delta_p.isApprox(Eigen::Vector3d::Zero(), 1e-9));
}

// --- 3. constant acceleration, no rotation ---
//
// gyro = 0, a_B = [2, 0, 0], dt = 0.5 s
// Expected: delta_v = a*dt = [1, 0, 0]
//           delta_p = 0.5*a*dt^2 = [0.25, 0, 0]

TEST(Preintegration, ConstantAcceleration) {
    PreintegratedImu preint;
    const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
    const double ax = 2.0;
    const double dt = 0.5;

    integrate(preint, make_meas(0, 0, 0, ax, 0, 0), zero, zero, dt);

    EXPECT_TRUE(preint.delta_v.isApprox(Eigen::Vector3d(ax * dt, 0, 0), 1e-9));
    EXPECT_TRUE(preint.delta_p.isApprox(Eigen::Vector3d(0.5 * ax * dt * dt, 0, 0), 1e-9));
    EXPECT_TRUE(preint.delta_R.isApprox(Eigen::Matrix3d::Identity(), 1e-9));
}

// --- 4. constant yaw rate, no acceleration ---
//
// omega_z = pi/2 rad/s, dt = 1.0 s
// Expected: delta_R = exp_so3([0, 0, pi/2])
//           delta_v = 0, delta_p = 0

TEST(Preintegration, ConstantYawRate) {
    PreintegratedImu preint;
    const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
    const double omega_z = M_PI / 2.0;
    const double dt = 1.0;

    integrate(preint, make_meas(0, 0, omega_z), zero, zero, dt);

    const Eigen::Matrix3d expected_R = exp_so3(Eigen::Vector3d(0, 0, omega_z * dt));
    EXPECT_TRUE(preint.delta_R.isApprox(expected_R, 1e-9));
    EXPECT_TRUE(preint.delta_v.isApprox(Eigen::Vector3d::Zero(), 1e-9));
    EXPECT_TRUE(preint.delta_p.isApprox(Eigen::Vector3d::Zero(), 1e-9));
}

// --- 5. bias correction: measurement == bias -> zero net motion ---

TEST(Preintegration, BiasEqualsRawMeasurementProducesZeroMotion) {
    PreintegratedImu preint;
    const Eigen::Vector3d gyro_bias{0.1, 0.2, 0.3};
    const Eigen::Vector3d accel_bias{1.0, 2.0, 3.0};
    const double dt = 0.05;

    // measurement equals bias -> omega = 0, a = 0
    ImuMeasurement meas = make_meas(0.1, 0.2, 0.3, 1.0, 2.0, 3.0);
    for (int i = 0; i < 20; ++i) {
        integrate(preint, meas, gyro_bias, accel_bias, dt);
    }

    EXPECT_TRUE(preint.delta_R.isApprox(Eigen::Matrix3d::Identity(), 1e-9));
    EXPECT_TRUE(preint.delta_v.isApprox(Eigen::Vector3d::Zero(), 1e-9));
    EXPECT_TRUE(preint.delta_p.isApprox(Eigen::Vector3d::Zero(), 1e-9));
}

// --- 7. rotated-frame acceleration ---
//
// Validates that body-frame acceleration is rotated by the current delta_R
// before accumulating into delta_v and delta_p.

TEST(Preintegration, BodyAccelRotatedByDeltaR_YawPlusZ) {
    // Step 1: 90-deg yaw about +Z, zero accel
    //   delta_R = R_z(pi/2), delta_v = 0, delta_p = 0
    // Step 2: zero gyro, accel = [2, 0, 0] (body +X), dt = 0.5 s
    //   delta_R * [2,0,0] = R_z(pi/2)*[2,0,0] = [0, 2, 0]
    //   delta_v = [0, 1, 0]   (= [0,2,0] * 0.5)
    //   delta_p = [0, 0.25, 0] (= 0.5 * [0,2,0] * 0.25)
    PreintegratedImu preint;
    const Eigen::Vector3d zero = Eigen::Vector3d::Zero();

    integrate(preint, make_meas(0, 0, M_PI / 2.0), zero, zero, 1.0);
    integrate(preint, make_meas(0, 0, 0, 2.0, 0, 0), zero, zero, 0.5);

    EXPECT_TRUE(preint.delta_v.isApprox(Eigen::Vector3d(0.0, 1.0, 0.0), 1e-9));
    EXPECT_TRUE(preint.delta_p.isApprox(Eigen::Vector3d(0.0, 0.25, 0.0), 1e-9));
}

TEST(Preintegration, BodyAccelRotatedByDeltaR_PitchPlusY) {
    // Step 1: 90-deg pitch about +Y, zero accel
    //   delta_R = R_y(pi/2), delta_v = 0, delta_p = 0
    // Step 2: zero gyro, accel = [0, 0, 3] (body +Z), dt = 0.4 s
    //   delta_R * [0,0,3] = R_y(pi/2)*[0,0,3] = [3, 0, 0]
    //   delta_v = [1.2, 0, 0]  (= [3,0,0] * 0.4)
    //   delta_p = [0.24, 0, 0] (= 0.5 * [3,0,0] * 0.16)
    PreintegratedImu preint;
    const Eigen::Vector3d zero = Eigen::Vector3d::Zero();

    integrate(preint, make_meas(0, M_PI / 2.0, 0), zero, zero, 1.0);
    integrate(preint, make_meas(0, 0, 0, 0, 0, 3.0), zero, zero, 0.4);

    EXPECT_TRUE(preint.delta_v.isApprox(Eigen::Vector3d(1.2, 0.0, 0.0), 1e-9));
    EXPECT_TRUE(preint.delta_p.isApprox(Eigen::Vector3d(0.24, 0.0, 0.0), 1e-9));
}

// --- 6. invalid inputs throw ---

TEST(Preintegration, InvalidDtThrows) {
    PreintegratedImu preint;
    const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
    constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
    constexpr double kInf = std::numeric_limits<double>::infinity();

    EXPECT_THROW(integrate(preint, make_meas(), zero, zero, 0.0),   std::invalid_argument);
    EXPECT_THROW(integrate(preint, make_meas(), zero, zero, -0.01), std::invalid_argument);
    EXPECT_THROW(integrate(preint, make_meas(), zero, zero, kNaN),  std::invalid_argument);
    EXPECT_THROW(integrate(preint, make_meas(), zero, zero, kInf),  std::invalid_argument);
}

TEST(Preintegration, NonFiniteMeasurementThrows) {
    PreintegratedImu preint;
    const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
    constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

    {
        ImuMeasurement m = make_meas(); m.gyro_radps.z() = kNaN;
        EXPECT_THROW(integrate(preint, m, zero, zero, 0.01), std::invalid_argument);
    }
    {
        ImuMeasurement m = make_meas(); m.accel_mps2.x() = kNaN;
        EXPECT_THROW(integrate(preint, m, zero, zero, 0.01), std::invalid_argument);
    }
    {
        const Eigen::Vector3d bad_bias{0.0, kNaN, 0.0};
        EXPECT_THROW(integrate(preint, make_meas(), bad_bias, zero, 0.01), std::invalid_argument);
    }
}
