#include <gtest/gtest.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>

#include "slam_core/geometry/se3.hpp"
#include "slam_core/geometry/so3.hpp"

using namespace slam_core::geometry;

// --- skew ---

TEST(Skew, CrossProductIdentity) {
    const Eigen::Vector3d w{1.0, 2.0, 3.0};
    const Eigen::Vector3d v{4.0, 5.0, 6.0};
    EXPECT_TRUE((skew(w) * v).isApprox(w.cross(v)));
}

// --- SO(3) exp ---

TEST(SO3Exp, NearZero) {
    const Eigen::Vector3d w = 1e-10 * Eigen::Vector3d{1.0, 0.5, 0.2};
    EXPECT_TRUE(exp_so3(w).isApprox(Eigen::Matrix3d::Identity(), 1e-6));
}

TEST(SO3Exp, Known90DegZ) {
    const Eigen::Vector3d w{0.0, 0.0, M_PI / 2.0};
    Eigen::Matrix3d       expected;
    expected << 0, -1, 0, 1, 0, 0, 0, 0, 1;
    EXPECT_TRUE(exp_so3(w).isApprox(expected, 1e-9));
}

// --- SO(3) exp/log round trip ---

TEST(SO3ExpLog, RoundTrip) {
    for (double angle : {0.1, 1.0, 2.5}) {
        const Eigen::Vector3d w = angle * Eigen::Vector3d{1.0, 0.0, 0.0};
        EXPECT_TRUE(log_so3(exp_so3(w)).isApprox(w, 1e-9));
    }
    const Eigen::Vector3d w_diag{0.3, -0.7, 1.1};
    EXPECT_TRUE(log_so3(exp_so3(w_diag)).isApprox(w_diag, 1e-9));
}

// --- validity checks ---

TEST(SO3Validity, AcceptsValid) {
    EXPECT_TRUE(is_valid_rotation(Eigen::Matrix3d::Identity()));
    EXPECT_TRUE(is_valid_rotation(exp_so3({0.4, -0.3, 0.9})));
}

TEST(SO3Validity, RejectsInvalid) {
    EXPECT_FALSE(is_valid_rotation(2.0 * Eigen::Matrix3d::Identity()));
    Eigen::Matrix3d row_swap;
    row_swap << 0, 1, 0, 1, 0, 0,  // det = -1
        0, 0, 1;
    EXPECT_FALSE(is_valid_rotation(row_swap));
}

// --- SE3 ---

TEST(SE3, InverseIdentity) {
    const SE3 T{exp_so3({0.3, -0.1, 0.5}), {1.0, -2.0, 0.5}};
    const SE3 result = T.compose(T.inverse());
    EXPECT_TRUE(result.R.isApprox(Eigen::Matrix3d::Identity(), 1e-9));
    EXPECT_LT(result.t.norm(), 1e-9);
}

TEST(SE3, ComposeApply) {
    // T_A_B: 90-deg rotation about Z, translate (1,0,0) in frame A
    const SE3 T_A_B{exp_so3({0.0, 0.0, M_PI / 2.0}), {1.0, 0.0, 0.0}};
    // T_B_C: no rotation, translate (0,1,0) in frame B
    const SE3 T_B_C{Eigen::Matrix3d::Identity(), {0.0, 1.0, 0.0}};

    // p_C = (1,0,0)
    // p_B = T_B_C.apply(p_C) = (1,0,0) + (0,1,0) = (1,1,0)
    // p_A = T_A_B.apply(p_B) = R_90Z*(1,1,0) + (1,0,0)
    //      = (-1,1,0) + (1,0,0) = (0,1,0)
    const Eigen::Vector3d p_C{1.0, 0.0, 0.0};
    const Eigen::Vector3d expected{0.0, 1.0, 0.0};

    const SE3 T_A_C = T_A_B.compose(T_B_C);
    EXPECT_TRUE(T_A_C.apply(p_C).isApprox(expected, 1e-9));
    EXPECT_TRUE(T_A_B.apply(T_B_C.apply(p_C)).isApprox(expected, 1e-9));
}

// --- SO(3) log near pi ---

TEST(SO3Log, NearPiX) {
    const double          theta = M_PI - 1e-6;
    const Eigen::Vector3d w     = theta * Eigen::Vector3d{1.0, 0.0, 0.0};
    EXPECT_TRUE(log_so3(exp_so3(w)).isApprox(w, 1e-9));
}

TEST(SO3Log, NearPiGeneral) {
    const double          theta = M_PI - 1e-6;
    const Eigen::Vector3d axis  = Eigen::Vector3d{1.0, 1.0, 1.0}.normalized();
    const Eigen::Vector3d w     = theta * axis;
    EXPECT_TRUE(log_so3(exp_so3(w)).isApprox(w, 1e-9));
}

TEST(SO3Log, ExactPiRoundTrip) {
    // At exactly pi, log is not unique (w and -w are both valid), so test R -> log -> exp -> R.
    const Eigen::Matrix3d R = exp_so3(M_PI * Eigen::Vector3d{1.0, 0.0, 0.0});
    EXPECT_TRUE(exp_so3(log_so3(R)).isApprox(R, 1e-9));
}
