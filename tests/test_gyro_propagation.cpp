#include <gtest/gtest.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "slam_core/imu/gyro_propagation.hpp"
#include "slam_core/geometry/so3.hpp"

using slam_core::imu::propagate_gyro;
using slam_core::imu::propagate_gyro_bias_corrected;
using slam_core::geometry::exp_so3;
using slam_core::geometry::is_valid_rotation;

// --- zero gyro ---

TEST(GyroPropagation, ZeroGyroLeavesOrientationUnchanged) {
    const Eigen::Matrix3d R = exp_so3({0.3, -0.5, 0.8});
    const Eigen::Matrix3d result = propagate_gyro(R, Eigen::Vector3d::Zero(), 0.01);
    EXPECT_TRUE(result.isApprox(R, 1e-12));
}

// --- known yaw ---

TEST(GyroPropagation, ConstantZGyro90DegIn1s) {
    // omega_z = pi/2 rad/s for dt = 1 s -> 90 deg yaw about body z
    const Eigen::Vector3d gyro{0.0, 0.0, M_PI / 2.0};
    const Eigen::Matrix3d result =
        propagate_gyro(Eigen::Matrix3d::Identity(), gyro, 1.0);
    const Eigen::Matrix3d expected = exp_so3(Eigen::Vector3d{0.0, 0.0, M_PI / 2.0});
    EXPECT_TRUE(result.isApprox(expected, 1e-9));
}

// --- composition ---

TEST(GyroPropagation, ManySmallStepsMatchOneLargeStep) {
    // For a constant gyro, N steps of (gyro, T/N) must equal 1 step of (gyro, T).
    // Same rotation axis at every step -> exact on SO(3), up to floating-point.
    const Eigen::Vector3d gyro{0.1, -0.2, 0.3};
    const double total_dt = 1.0;
    const int N = 1000;
    const double small_dt = total_dt / N;

    Eigen::Matrix3d R_many = Eigen::Matrix3d::Identity();
    for (int i = 0; i < N; ++i) {
        R_many = propagate_gyro(R_many, gyro, small_dt);
    }
    const Eigen::Matrix3d R_one =
        propagate_gyro(Eigen::Matrix3d::Identity(), gyro, total_dt);

    EXPECT_TRUE(R_many.isApprox(R_one, 1e-9));
}

// --- invalid dt ---

TEST(GyroPropagation, InvalidDtThrows) {
    const Eigen::Matrix3d R  = Eigen::Matrix3d::Identity();
    const Eigen::Vector3d gyro{0.0, 0.0, 1.0};
    constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
    constexpr double kInf = std::numeric_limits<double>::infinity();

    EXPECT_THROW(propagate_gyro(R, gyro, 0.0),   std::invalid_argument);
    EXPECT_THROW(propagate_gyro(R, gyro, -1.0),  std::invalid_argument);
    EXPECT_THROW(propagate_gyro(R, gyro, kNaN),  std::invalid_argument);
    EXPECT_THROW(propagate_gyro(R, gyro,  kInf), std::invalid_argument);
    EXPECT_THROW(propagate_gyro(R, gyro, -kInf), std::invalid_argument);
}

// --- bias-corrected: zero bias matches propagate_gyro ---

TEST(GyroBiasCorrectedPropagation, ZeroBiasMatchesPropagateGyro) {
    const Eigen::Matrix3d R = exp_so3({0.3, -0.5, 0.8});
    const Eigen::Vector3d gyro{0.1, -0.2, 0.3};
    const Eigen::Matrix3d result =
        propagate_gyro_bias_corrected(R, gyro, Eigen::Vector3d::Zero(), 0.01);
    const Eigen::Matrix3d expected = propagate_gyro(R, gyro, 0.01);
    EXPECT_TRUE(result.isApprox(expected, 1e-12));
}

// --- bias-corrected: full bias leaves orientation unchanged ---

TEST(GyroBiasCorrectedPropagation, FullBiasLeavesOrientationUnchanged) {
    // gyro == bias -> corrected rate is zero -> R_W_B unchanged
    const Eigen::Matrix3d R = exp_so3({0.3, -0.5, 0.8});
    const Eigen::Vector3d gyro{0.4, -0.1, 0.7};
    const Eigen::Matrix3d result =
        propagate_gyro_bias_corrected(R, gyro, gyro, 0.05);
    EXPECT_TRUE(result.isApprox(R, 1e-12));
}

// --- bias-corrected: partial bias produces expected rotation ---

TEST(GyroBiasCorrectedPropagation, PartialBiasProducesExpectedRotation) {
    // gyro = [0, 0, 1.0] rad/s, bias = [0, 0, 0.25] rad/s, dt = 2.0 s
    // corrected rate = [0, 0, 0.75] rad/s -> angle = 0.75 * 2.0 = 1.5 rad about +z
    const Eigen::Vector3d gyro{0.0, 0.0, 1.0};
    const Eigen::Vector3d bias{0.0, 0.0, 0.25};
    const Eigen::Matrix3d result =
        propagate_gyro_bias_corrected(Eigen::Matrix3d::Identity(), gyro, bias, 2.0);
    const Eigen::Matrix3d expected =
        Eigen::AngleAxisd(1.5, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    EXPECT_TRUE(result.isApprox(expected, 1e-9));
}

// --- bias-corrected: invalid dt throws ---

TEST(GyroBiasCorrectedPropagation, InvalidDtThrows) {
    const Eigen::Matrix3d R  = Eigen::Matrix3d::Identity();
    const Eigen::Vector3d gyro{0.0, 0.0, 1.0};
    const Eigen::Vector3d bias = Eigen::Vector3d::Zero();
    constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
    constexpr double kInf = std::numeric_limits<double>::infinity();

    EXPECT_THROW(propagate_gyro_bias_corrected(R, gyro, bias, 0.0),   std::invalid_argument);
    EXPECT_THROW(propagate_gyro_bias_corrected(R, gyro, bias, -1.0),  std::invalid_argument);
    EXPECT_THROW(propagate_gyro_bias_corrected(R, gyro, bias, kNaN),  std::invalid_argument);
    EXPECT_THROW(propagate_gyro_bias_corrected(R, gyro, bias,  kInf), std::invalid_argument);
    EXPECT_THROW(propagate_gyro_bias_corrected(R, gyro, bias, -kInf), std::invalid_argument);
}

// --- bias-corrected: rotation validity ---

TEST(GyroBiasCorrectedPropagation, OutputIsValidRotationAfterManySteps) {
    const Eigen::Vector3d gyro{0.3, -0.7, 1.1};
    const Eigen::Vector3d bias{0.05, -0.1, 0.2};
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    for (int i = 0; i < 5000; ++i) {
        R = propagate_gyro_bias_corrected(R, gyro, bias, 0.005);
    }
    EXPECT_TRUE(is_valid_rotation(R, 1e-6));
}

// --- rotation validity ---

TEST(GyroPropagation, OutputIsValidRotationAfterManySteps) {
    const Eigen::Vector3d gyro{0.3, -0.7, 1.1};
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    for (int i = 0; i < 5000; ++i) {
        R = propagate_gyro(R, gyro, 0.005);
    }
    EXPECT_TRUE(is_valid_rotation(R, 1e-6));
}
