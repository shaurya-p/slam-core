#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>
#include <Eigen/Core>

#include "slam_core/camera/pinhole_camera.hpp"
#include "slam_core/factors/prior_factors.hpp"
#include "slam_core/factors/reprojection_factor.hpp"
#include "slam_core/geometry/so3.hpp"
#include "slam_core/optim/levenberg_marquardt.hpp"
#include "slam_core/optim/problem.hpp"

#include "numeric_jacobian.hpp"

using slam_core::camera::PinholeCamera;
using slam_core::factors::ReprojectionFactor;
using slam_core::factors::So3PriorFactor;
using slam_core::factors::VectorPriorFactor;
using slam_core::geometry::exp_so3;
using slam_core::geometry::log_so3;
using slam_core::optim::Problem;
using slam_core::optim::So3Variable;
using slam_core::optim::VectorVariable;

namespace {

// EuRoC cam0-like intrinsics and a small camera-to-body offset.
const PinholeCamera   kCam(458.654, 457.296, 367.215, 248.375);
const Eigen::Matrix3d kRBC = exp_so3({0.01, -0.02, 0.015});
const Eigen::Vector3d kTBC(-0.02, -0.06, 0.01);

// Deterministic uniform in [-1, 1] from raw mt19937 draws.
class Uniform {
public:
    explicit Uniform(std::uint32_t seed) : rng_(seed) {}
    double operator()() { return 2.0 * (static_cast<double>(rng_()) / 4294967295.0) - 1.0; }
    Eigen::Vector3d vec3() { return {(*this)(), (*this)(), (*this)()}; }

private:
    std::mt19937 rng_;
};

// Projects a world landmark through a pose; asserts positive depth.
Eigen::Vector2d project_world(const Eigen::Matrix3d& R_W_B,
                              const Eigen::Vector3d& p_W_B,
                              const Eigen::Vector3d& p_W_L) {
    const Eigen::Vector3d p_B = R_W_B.transpose() * (p_W_L - p_W_B);
    const Eigen::Vector3d p_C = kRBC.transpose() * (p_B - kTBC);
    return kCam.project(p_C);
}

}  // namespace

TEST(ReprojectionFactor, ResidualZeroAtConsistentGeometry) {
    const Eigen::Matrix3d R_gt = exp_so3({0.1, -0.2, 0.3});
    const Eigen::Vector3d p_gt(0.5, -0.3, 0.2);
    const Eigen::Vector3d landmark(1.0, 0.5, 4.0);

    So3Variable    R(R_gt);
    VectorVariable p(p_gt), L(landmark);

    const ReprojectionFactor factor(&R, &p, &L, kCam, kRBC, kTBC,
                                    project_world(R_gt, p_gt, landmark), 1.0);
    EXPECT_LT(factor.residual().norm(), 1e-10);
}

TEST(ReprojectionFactor, JacobiansMatchNumeric) {
    const Eigen::Matrix3d R_gt = exp_so3({0.1, -0.2, 0.3});
    const Eigen::Vector3d p_gt(0.5, -0.3, 0.2);
    const Eigen::Vector3d landmark(1.0, 0.5, 4.0);
    const Eigen::Vector2d obs = project_world(R_gt, p_gt, landmark);

    // Evaluate away from the optimum.
    So3Variable    R(R_gt * exp_so3({0.02, -0.01, 0.03}));
    VectorVariable p(p_gt + Eigen::Vector3d(0.1, -0.05, 0.08));
    VectorVariable L(landmark + Eigen::Vector3d(-0.1, 0.2, 0.15));

    const ReprojectionFactor factor(&R, &p, &L, kCam, kRBC, kTBC, obs, 0.5);
    // Pixel-scale Jacobians (~1e2-1e3); tolerance scaled accordingly.
    EXPECT_LT(slam_core::testing::max_jacobian_error(factor), 1e-3);
}

TEST(ReprojectionFactor, BehindCameraGivesLargeConstantResidual) {
    So3Variable    R;  // identity
    VectorVariable p(Eigen::Vector3d::Zero());
    VectorVariable L(Eigen::Vector3d(0.0, 0.0, -3.0));  // behind camera

    const ReprojectionFactor factor(&R, &p, &L, kCam, kRBC, kTBC, {320.0, 240.0}, 1.0);
    EXPECT_GT(factor.residual().norm(), 1e5);
    for (const Eigen::MatrixXd& J : factor.jacobians()) {
        EXPECT_LT(J.cwiseAbs().maxCoeff(), 1e-300);
    }
}

TEST(ReprojectionFactor, MiniBundleAdjustmentRecoversPoseAndLandmarks) {
    // 3 poses looking at 12 landmarks; poses 0 and 1 pinned by priors
    // (gauge + scale); pose 2 and all landmarks perturbed, then recovered
    // from noise-free pixel observations.
    const int n_landmarks = 12;

    std::vector<Eigen::Matrix3d> R_gt;
    std::vector<Eigen::Vector3d> p_gt;
    for (int k = 0; k < 3; ++k) {
        R_gt.push_back(exp_so3(Eigen::Vector3d(0.05, -0.03, 0.08) * k));
        p_gt.push_back(Eigen::Vector3d(0.4 * k, 0.1 * k, -0.05 * k));
    }

    Uniform                      uniform(99);
    std::vector<Eigen::Vector3d> L_gt;
    for (int i = 0; i < n_landmarks; ++i) {
        L_gt.push_back(Eigen::Vector3d(1.5 * uniform(), 1.0 * uniform(), 5.0 + 2.0 * uniform()));
    }

    std::vector<So3Variable>    R;
    std::vector<VectorVariable> p;
    std::vector<VectorVariable> L;
    for (int k = 0; k < 3; ++k) {
        // Poses 0/1 start at GT; pose 2 perturbed.
        const Eigen::Vector3d dR =
            (k == 2) ? Eigen::Vector3d(0.05, -0.04, 0.06) : Eigen::Vector3d::Zero();
        const Eigen::Vector3d dp =
            (k == 2) ? Eigen::Vector3d(0.2, -0.15, 0.1) : Eigen::Vector3d::Zero();
        R.emplace_back(R_gt[k] * exp_so3(dR));
        p.emplace_back(p_gt[k] + dp);
    }
    for (int i = 0; i < n_landmarks; ++i) {
        L.emplace_back(L_gt[i] + 0.3 * uniform.vec3());
    }

    std::vector<ReprojectionFactor> factors;
    factors.reserve(3 * n_landmarks);
    for (int k = 0; k < 3; ++k) {
        for (int i = 0; i < n_landmarks; ++i) {
            factors.emplace_back(&R[k], &p[k], &L[i], kCam, kRBC, kTBC,
                                 project_world(R_gt[k], p_gt[k], L_gt[i]), 1.0);
        }
    }

    So3PriorFactor    prior_R0(&R[0], R_gt[0], 1e-6);
    VectorPriorFactor prior_p0(&p[0], p_gt[0], 1e-6);
    So3PriorFactor    prior_R1(&R[1], R_gt[1], 1e-6);
    VectorPriorFactor prior_p1(&p[1], p_gt[1], 1e-6);

    Problem problem;
    for (int k = 0; k < 3; ++k) {
        problem.add_variable(&R[k]);
        problem.add_variable(&p[k]);
    }
    for (auto& l : L) problem.add_variable(&l);
    for (auto& f : factors) problem.add_factor(&f);
    problem.add_factor(&prior_R0);
    problem.add_factor(&prior_p0);
    problem.add_factor(&prior_R1);
    problem.add_factor(&prior_p1);

    slam_core::optim::LmOptions options;
    options.max_iterations                  = 100;
    const slam_core::optim::LmResult result = slam_core::optim::optimize(problem, options);

    EXPECT_TRUE(result.converged) << result.message;
    EXPECT_LT(result.final_cost, 1e-10);
    EXPECT_LT(log_so3(R_gt[2].transpose() * R[2].R()).norm(), 1e-6);
    EXPECT_LT((p[2].vec() - p_gt[2]).norm(), 1e-5);
    for (int i = 0; i < n_landmarks; ++i) {
        EXPECT_LT((L[i].vec() - L_gt[i]).norm(), 1e-4) << "landmark " << i;
    }
}
