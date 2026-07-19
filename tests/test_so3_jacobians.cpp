#include <cmath>

#include <gtest/gtest.h>
#include <Eigen/Core>

#include "slam_core/geometry/so3.hpp"

using slam_core::geometry::exp_so3;
using slam_core::geometry::log_so3;
using slam_core::geometry::right_jacobian_so3;
using slam_core::geometry::right_jacobian_so3_inverse;

namespace {

// Numeric column i of J_r(w): d/dh log_so3(exp_so3(w)ᵀ · exp_so3(w + h·e_i)),
// central differences. Follows from exp(w + dw) ≈ exp(w)·exp(J_r(w)·dw).
Eigen::Matrix3d numeric_right_jacobian(const Eigen::Vector3d& w, double h = 1e-6) {
    const Eigen::Matrix3d R = exp_so3(w);
    Eigen::Matrix3d       J;
    for (int i = 0; i < 3; ++i) {
        Eigen::Vector3d e           = Eigen::Vector3d::Zero();
        e(i)                        = h;
        const Eigen::Vector3d plus  = log_so3(R.transpose() * exp_so3(w + e));
        const Eigen::Vector3d minus = log_so3(R.transpose() * exp_so3(w - e));
        J.col(i)                    = (plus - minus) / (2.0 * h);
    }
    return J;
}

const Eigen::Vector3d kTestVectors[] = {
    {0.0, 0.0, 0.0},   {1e-7, -2e-7, 3e-7}, {0.1, 0.0, 0.0}, {0.3, -0.5, 0.8},
    {-1.2, 0.7, -0.4}, {2.0, 1.0, -1.5},    {0.0, 0.0, 3.0},
};

}  // namespace

TEST(RightJacobianSo3, MatchesNumericDerivative) {
    for (const Eigen::Vector3d& w : kTestVectors) {
        const Eigen::Matrix3d J_analytic = right_jacobian_so3(w);
        const Eigen::Matrix3d J_numeric  = numeric_right_jacobian(w);
        EXPECT_LT((J_analytic - J_numeric).cwiseAbs().maxCoeff(), 1e-8) << "w = " << w.transpose();
    }
}

TEST(RightJacobianSo3, InverseIsExactInverse) {
    for (const Eigen::Vector3d& w : kTestVectors) {
        const Eigen::Matrix3d product = right_jacobian_so3_inverse(w) * right_jacobian_so3(w);
        EXPECT_TRUE(product.isApprox(Eigen::Matrix3d::Identity(), 1e-9)) << "w = " << w.transpose();
    }
}

TEST(RightJacobianSo3, IdentityAtZero) {
    EXPECT_TRUE(
        right_jacobian_so3(Eigen::Vector3d::Zero()).isApprox(Eigen::Matrix3d::Identity(), 1e-15));
    EXPECT_TRUE(right_jacobian_so3_inverse(Eigen::Vector3d::Zero())
                    .isApprox(Eigen::Matrix3d::Identity(), 1e-15));
}

TEST(RightJacobianSo3, TaylorBranchIsContinuous) {
    // Just above the 1e-5 branch threshold the closed-form expressions must
    // agree with the Taylor expansions evaluated at the same w to O(θ³).
    const Eigen::Vector3d w = Eigen::Vector3d(1.0, 2.0, -2.0).normalized() * 1.1e-5;
    const Eigen::Matrix3d W = slam_core::geometry::skew(w);

    const Eigen::Matrix3d Jr_taylor = Eigen::Matrix3d::Identity() - 0.5 * W + (1.0 / 6.0) * W * W;
    const Eigen::Matrix3d Jr_inv_taylor =
        Eigen::Matrix3d::Identity() + 0.5 * W + (1.0 / 12.0) * W * W;

    // Tolerance dominated by (1 - cos θ)/θ² cancellation (~1e-16/θ² · θ), not
    // by the O(θ³) truncation of the Taylor branch.
    EXPECT_LT((right_jacobian_so3(w) - Jr_taylor).cwiseAbs().maxCoeff(), 1e-10);
    EXPECT_LT((right_jacobian_so3_inverse(w) - Jr_inv_taylor).cwiseAbs().maxCoeff(), 1e-10);
}

TEST(RightJacobianSo3, FiniteNearPi) {
    const Eigen::Vector3d w      = Eigen::Vector3d(0.0, 0.0, 1.0) * (M_PI - 1e-4);
    const Eigen::Matrix3d Jr     = right_jacobian_so3(w);
    const Eigen::Matrix3d Jr_inv = right_jacobian_so3_inverse(w);
    EXPECT_TRUE(Jr.allFinite());
    EXPECT_TRUE(Jr_inv.allFinite());
    EXPECT_TRUE((Jr_inv * Jr).isApprox(Eigen::Matrix3d::Identity(), 1e-6));
}

TEST(RightJacobianSo3, LocalLogApproximation) {
    // log_so3(exp(w)·exp(dw)) ≈ w + J_r⁻¹(w)·dw, error O(‖dw‖²).
    const Eigen::Vector3d w(0.4, -0.2, 0.7);
    const Eigen::Vector3d dw(1e-4, -2e-4, 1.5e-4);
    const Eigen::Vector3d lhs = log_so3(exp_so3(w) * exp_so3(dw));
    const Eigen::Vector3d rhs = w + right_jacobian_so3_inverse(w) * dw;
    EXPECT_LT((lhs - rhs).norm(), 1e-7);
}
