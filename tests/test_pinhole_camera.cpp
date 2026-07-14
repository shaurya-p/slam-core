#include <gtest/gtest.h>
#include <Eigen/Core>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "slam_core/camera/pinhole_camera.hpp"

using slam_core::camera::PinholeCamera;

static const double kInf = std::numeric_limits<double>::infinity();
static const double kNaN = std::numeric_limits<double>::quiet_NaN();

// --- 1. Centered projection ---

TEST(PinholeCamera, CenteredProjection) {
    PinholeCamera         cam(400.0, 400.0, 320.0, 240.0);
    const Eigen::Vector2d px = cam.project({0.0, 0.0, 5.0});
    EXPECT_DOUBLE_EQ(px.x(), 320.0);
    EXPECT_DOUBLE_EQ(px.y(), 240.0);
}

// --- 2. Off-center projection ---

TEST(PinholeCamera, OffCenterProjection) {
    // u = 500 * 1.0 / 2.0 + 320 = 570,  v = 500 * (-1.0) / 2.0 + 240 = -10
    PinholeCamera         cam(500.0, 500.0, 320.0, 240.0);
    const Eigen::Vector2d px = cam.project({1.0, -1.0, 2.0});
    EXPECT_DOUBLE_EQ(px.x(), 570.0);
    EXPECT_DOUBLE_EQ(px.y(), -10.0);
}

// --- 3. Unit-depth projection ---

TEST(PinholeCamera, UnitDepthProjection) {
    // Z = 1: u = fx * X + cx,  v = fy * Y + cy
    PinholeCamera         cam(600.0, 400.0, 320.0, 240.0);
    const Eigen::Vector2d px = cam.project({0.5, -0.25, 1.0});
    EXPECT_DOUBLE_EQ(px.x(), 600.0 * 0.5 + 320.0);
    EXPECT_DOUBLE_EQ(px.y(), 400.0 * -0.25 + 240.0);
}

// --- 4. Reprojection error: zero when observed equals predicted ---

TEST(PinholeCamera, ReprojError_ZeroWhenExact) {
    PinholeCamera         cam(400.0, 400.0, 320.0, 240.0);
    const Eigen::Vector3d p_C{1.0, 2.0, 5.0};
    const Eigen::Vector2d predicted = cam.project(p_C);
    const Eigen::Vector2d err       = cam.reprojection_error(p_C, predicted);
    EXPECT_NEAR(err.x(), 0.0, 1e-12);
    EXPECT_NEAR(err.y(), 0.0, 1e-12);
}

// --- 5. Reprojection error: signed offset ---

TEST(PinholeCamera, ReprojError_SignedOffset) {
    PinholeCamera         cam(400.0, 400.0, 320.0, 240.0);
    const Eigen::Vector3d p_C{0.0, 0.0, 1.0};
    // predicted = [320, 240]; observed is 5 right and 3 up
    const Eigen::Vector2d observed{325.0, 237.0};
    const Eigen::Vector2d err = cam.reprojection_error(p_C, observed);
    EXPECT_DOUBLE_EQ(err.x(), 5.0);   // observed to the right of predicted
    EXPECT_DOUBLE_EQ(err.y(), -3.0);  // observed above predicted (y down)
}

// --- 6-9. Invalid intrinsics ---

TEST(PinholeCamera, InvalidFx_Nonfinite) {
    EXPECT_THROW(PinholeCamera(kNaN, 400.0, 320.0, 240.0), std::invalid_argument);
}

TEST(PinholeCamera, InvalidFy_Nonfinite) {
    EXPECT_THROW(PinholeCamera(400.0, kInf, 320.0, 240.0), std::invalid_argument);
}

TEST(PinholeCamera, InvalidCx_Nonfinite) {
    EXPECT_THROW(PinholeCamera(400.0, 400.0, kNaN, 240.0), std::invalid_argument);
}

TEST(PinholeCamera, InvalidCy_Nonfinite) {
    EXPECT_THROW(PinholeCamera(400.0, 400.0, 320.0, kNaN), std::invalid_argument);
}

// --- 10-11. Non-positive fx/fy ---

TEST(PinholeCamera, InvalidFx_Zero) {
    EXPECT_THROW(PinholeCamera(0.0, 400.0, 320.0, 240.0), std::invalid_argument);
}

TEST(PinholeCamera, InvalidFy_Negative) {
    EXPECT_THROW(PinholeCamera(400.0, -1.0, 320.0, 240.0), std::invalid_argument);
}

// --- 12-14. Invalid points ---

TEST(PinholeCamera, InvalidPoint_NaN) {
    PinholeCamera cam(400.0, 400.0, 320.0, 240.0);
    EXPECT_THROW(cam.project({kNaN, 0.0, 1.0}), std::invalid_argument);
}

TEST(PinholeCamera, InvalidPoint_ZeroZ) {
    PinholeCamera cam(400.0, 400.0, 320.0, 240.0);
    EXPECT_THROW(cam.project({0.0, 0.0, 0.0}), std::invalid_argument);
}

TEST(PinholeCamera, InvalidPoint_NegativeZ) {
    PinholeCamera cam(400.0, 400.0, 320.0, 240.0);
    EXPECT_THROW(cam.project({0.0, 0.0, -1.0}), std::invalid_argument);
}

// --- 15. Unproject principal point -> [0, 0, 1] ---

TEST(PinholeCamera, Unproject_PrincipalPoint) {
    PinholeCamera         cam(400.0, 400.0, 320.0, 240.0);
    const Eigen::Vector3d b = cam.unproject_to_bearing({320.0, 240.0});
    EXPECT_NEAR(b.x(), 0.0, 1e-12);
    EXPECT_NEAR(b.y(), 0.0, 1e-12);
    EXPECT_NEAR(b.z(), 1.0, 1e-12);
}

// --- 16. Unproject one focal length to the right -> normalize([1, 0, 1]) ---

TEST(PinholeCamera, Unproject_OneFocalLengthRight) {
    PinholeCamera         cam(400.0, 400.0, 320.0, 240.0);
    const Eigen::Vector3d b        = cam.unproject_to_bearing({720.0, 240.0});
    const Eigen::Vector3d expected = Eigen::Vector3d(1.0, 0.0, 1.0).normalized();
    EXPECT_NEAR(b.x(), expected.x(), 1e-12);
    EXPECT_NEAR(b.y(), expected.y(), 1e-12);
    EXPECT_NEAR(b.z(), expected.z(), 1e-12);
}

// --- 17. Unproject one focal length down -> normalize([0, 1, 1]) ---

TEST(PinholeCamera, Unproject_OneFocalLengthDown) {
    PinholeCamera         cam(400.0, 400.0, 320.0, 240.0);
    const Eigen::Vector3d b        = cam.unproject_to_bearing({320.0, 640.0});
    const Eigen::Vector3d expected = Eigen::Vector3d(0.0, 1.0, 1.0).normalized();
    EXPECT_NEAR(b.x(), expected.x(), 1e-12);
    EXPECT_NEAR(b.y(), expected.y(), 1e-12);
    EXPECT_NEAR(b.z(), expected.z(), 1e-12);
}

// --- 18. General pixel produces expected normalized vector ---

TEST(PinholeCamera, Unproject_GeneralPixel) {
    // fx=500, fy=400, cx=320, cy=240; pixel=[570, 440]
    // x = (570-320)/500 = 0.5,  y = (440-240)/400 = 0.5
    // ray = [0.5, 0.5, 1],  norm = sqrt(1.5)
    PinholeCamera         cam(500.0, 400.0, 320.0, 240.0);
    const Eigen::Vector3d b        = cam.unproject_to_bearing({570.0, 440.0});
    const Eigen::Vector3d expected = Eigen::Vector3d(0.5, 0.5, 1.0).normalized();
    EXPECT_NEAR(b.x(), expected.x(), 1e-12);
    EXPECT_NEAR(b.y(), expected.y(), 1e-12);
    EXPECT_NEAR(b.z(), expected.z(), 1e-12);
}

// --- 19. Output norm is 1 ---

TEST(PinholeCamera, Unproject_NormIsOne) {
    PinholeCamera         cam(600.0, 400.0, 320.0, 240.0);
    const Eigen::Vector3d b = cam.unproject_to_bearing({450.0, 180.0});
    EXPECT_NEAR(b.norm(), 1.0, 1e-12);
}

// --- 20. Non-finite pixel throws ---

TEST(PinholeCamera, Unproject_NonFiniteU) {
    PinholeCamera cam(400.0, 400.0, 320.0, 240.0);
    EXPECT_THROW(cam.unproject_to_bearing({kNaN, 240.0}), std::invalid_argument);
}

TEST(PinholeCamera, Unproject_NonFiniteV) {
    PinholeCamera cam(400.0, 400.0, 320.0, 240.0);
    EXPECT_THROW(cam.unproject_to_bearing({320.0, kInf}), std::invalid_argument);
}
