#include <gtest/gtest.h>
#include <Eigen/Core>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "slam_core/geometry/so3.hpp"
#include "slam_core/imu/initialization.hpp"

using slam_core::geometry::is_valid_rotation;
using slam_core::imu::estimate_R_W_B_from_stationary;
using slam_core::imu::ImuMeasurement;

static const Eigen::Vector3d kGravityW(0.0, 0.0, -9.81);

static ImuMeasurement make_meas(const Eigen::Vector3d& accel) {
    ImuMeasurement m;
    m.timestamp_s = 0.0;
    m.accel_mps2  = accel;
    m.gyro_radps  = Eigen::Vector3d::Zero();
    return m;
}

static std::vector<ImuMeasurement> make_stationary_meas(const Eigen::Vector3d& accel, int n = 10) {
    std::vector<ImuMeasurement> meas(n, make_meas(accel));
    return meas;
}

// --- 1. Aligned/identity case ---

TEST(ImuInitialization, AlignedIdentity) {
    // Body aligned with world: accel_body = [0, 0, +9.81] = −gravity_W.
    // Expected: R_W_B ≈ Identity.
    const Eigen::Vector3d accel_body(0.0, 0.0, 9.81);
    auto                  meas = make_stationary_meas(accel_body);

    const Eigen::Matrix3d R = estimate_R_W_B_from_stationary(meas, kGravityW);

    EXPECT_TRUE(is_valid_rotation(R));
    EXPECT_TRUE(R.isApprox(Eigen::Matrix3d::Identity(), 1e-9));
}

// --- 2. Known roll-only case ---

TEST(ImuInitialization, KnownRoll90) {
    // R_W_B = Rx(π/2): 90° roll about world x-axis.
    // accel_body = −R_W_B^T * gravity_W
    //            = −Rx(−π/2) * [0, 0, −9.81]
    //            = [0, 9.81, 0]
    // We do not check R == Rx(π/2) because yaw is unconstrained;
    // we only verify gravity alignment: R_hat * accel_body ≈ −gravity_W.
    const double          g = 9.81;
    const Eigen::Vector3d accel_body(0.0, g, 0.0);
    auto                  meas = make_stationary_meas(accel_body);

    const Eigen::Matrix3d R = estimate_R_W_B_from_stationary(meas, kGravityW);

    EXPECT_TRUE(is_valid_rotation(R));
    const Eigen::Vector3d aligned = R * accel_body;
    EXPECT_TRUE(aligned.isApprox(-kGravityW, 1e-9));
}

// --- 3. Arbitrary roll+pitch case ---

TEST(ImuInitialization, ArbitraryRollPitch) {
    // Use a known R_W_B built from 30° roll + 20° pitch (no yaw).
    // Synthesize accel_body, then check only gravity alignment of R_hat.
    const double cr = std::cos(0.3), sr = std::sin(0.3);  // roll 0.3 rad
    const double cp = std::cos(0.2), sp = std::sin(0.2);  // pitch 0.2 rad

    // Rx(roll) * Ry(pitch)
    Eigen::Matrix3d R_W_B;
    R_W_B << cp, 0.0, sp, sr * sp, cr, -sr * cp, -cr * sp, sr, cr * cp;

    const Eigen::Vector3d accel_body = -(R_W_B.transpose() * kGravityW);
    auto                  meas       = make_stationary_meas(accel_body);

    const Eigen::Matrix3d R_hat = estimate_R_W_B_from_stationary(meas, kGravityW);

    EXPECT_TRUE(is_valid_rotation(R_hat));
    const Eigen::Vector3d aligned = R_hat * accel_body;
    EXPECT_TRUE(aligned.isApprox(-kGravityW, 1e-9));
}

// --- 4. Antiparallel case (body upside-down) ---

TEST(ImuInitialization, AntiparallelCase) {
    // Body upside-down: accel_body = [0, 0, −9.81], antiparallel to −gravity_W.
    // Requires a 180° rotation; result must still satisfy gravity alignment.
    const Eigen::Vector3d accel_body(0.0, 0.0, -9.81);
    auto                  meas = make_stationary_meas(accel_body);

    const Eigen::Matrix3d R = estimate_R_W_B_from_stationary(meas, kGravityW);

    EXPECT_TRUE(is_valid_rotation(R));
    const Eigen::Vector3d aligned = R * accel_body;
    EXPECT_TRUE(aligned.isApprox(-kGravityW, 1e-9));
}

// --- 5. Noisy stationary measurements ---

TEST(ImuInitialization, NoisyStationary) {
    // Small perturbations around [0, 0, 9.81]; gravity alignment should still hold
    // within a loose tolerance consistent with ~1% noise.
    const double                       g            = 9.81;
    const std::vector<Eigen::Vector3d> noisy_accels = {
        {0.02, -0.01, g + 0.03}, {-0.01, 0.02, g - 0.02}, {0.00, 0.01, g + 0.01},
        {0.03, -0.02, g - 0.01}, {-0.02, 0.00, g + 0.02},
    };
    std::vector<ImuMeasurement> meas;
    for (const auto& a : noisy_accels) {
        meas.push_back(make_meas(a));
    }

    const Eigen::Matrix3d R = estimate_R_W_B_from_stationary(meas, kGravityW);

    EXPECT_TRUE(is_valid_rotation(R));

    // Compute mean accel and check gravity alignment within noise-level tolerance.
    Eigen::Vector3d accel_sum = Eigen::Vector3d::Zero();
    for (const auto& m : meas) {
        accel_sum += m.accel_mps2;
    }
    const Eigen::Vector3d accel_mean = accel_sum / static_cast<double>(meas.size());

    const Eigen::Vector3d aligned = R * accel_mean;
    EXPECT_TRUE(aligned.isApprox(-kGravityW, 0.1));
}

// --- 6. Invalid: empty sequence ---

TEST(ImuInitialization, InvalidEmpty) {
    EXPECT_THROW(estimate_R_W_B_from_stationary({}, kGravityW), std::invalid_argument);
}

// --- 7. Invalid: single measurement ---

TEST(ImuInitialization, InvalidSingleMeasurement) {
    const std::vector<ImuMeasurement> meas = {make_meas({0.0, 0.0, 9.81})};
    EXPECT_THROW(estimate_R_W_B_from_stationary(meas, kGravityW), std::invalid_argument);
}

// --- 8. Invalid: non-finite accelerometer ---

TEST(ImuInitialization, InvalidNonFiniteAccel) {
    const double                nan  = std::numeric_limits<double>::quiet_NaN();
    std::vector<ImuMeasurement> meas = make_stationary_meas({0.0, 0.0, 9.81});
    meas[0].accel_mps2.x()           = nan;
    EXPECT_THROW(estimate_R_W_B_from_stationary(meas, kGravityW), std::invalid_argument);
}

// --- 9. Invalid: non-finite gravity ---

TEST(ImuInitialization, InvalidNonFiniteGravity) {
    const double nan  = std::numeric_limits<double>::quiet_NaN();
    auto         meas = make_stationary_meas({0.0, 0.0, 9.81});
    EXPECT_THROW(estimate_R_W_B_from_stationary(meas, Eigen::Vector3d(0.0, 0.0, nan)),
                 std::invalid_argument);
}

// --- 10. Invalid: near-zero gravity ---

TEST(ImuInitialization, InvalidNearZeroGravity) {
    auto meas = make_stationary_meas({0.0, 0.0, 9.81});
    EXPECT_THROW(estimate_R_W_B_from_stationary(meas, Eigen::Vector3d(0.0, 0.0, 1e-10)),
                 std::invalid_argument);
}

// --- 11. Invalid: near-zero mean accelerometer ---

TEST(ImuInitialization, InvalidNearZeroMeanAccel) {
    auto meas = make_stationary_meas({0.0, 0.0, 0.0});
    EXPECT_THROW(estimate_R_W_B_from_stationary(meas, kGravityW), std::invalid_argument);
}
