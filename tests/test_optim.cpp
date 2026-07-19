#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>
#include <Eigen/Core>

#include "slam_core/factors/point_alignment_factor.hpp"
#include "slam_core/factors/prior_factors.hpp"
#include "slam_core/factors/relative_pose_factor.hpp"
#include "slam_core/geometry/so3.hpp"
#include "slam_core/optim/levenberg_marquardt.hpp"
#include "slam_core/optim/problem.hpp"
#include "slam_core/optim/variable.hpp"

#include "numeric_jacobian.hpp"

using slam_core::factors::PointAlignmentFactor;
using slam_core::factors::RelativePoseFactor;
using slam_core::factors::So3PriorFactor;
using slam_core::factors::VectorPriorFactor;
using slam_core::geometry::exp_so3;
using slam_core::geometry::is_valid_rotation;
using slam_core::geometry::log_so3;
using slam_core::optim::LmOptions;
using slam_core::optim::LmResult;
using slam_core::optim::Problem;
using slam_core::optim::So3Variable;
using slam_core::optim::VectorVariable;
using slam_core::testing::max_jacobian_error;

namespace {

// Deterministic uniform values in [-1, 1]; raw mt19937 draws so results do
// not depend on the standard library's distribution implementation.
class Uniform {
public:
    explicit Uniform(std::uint32_t seed) : rng_(seed) {}
    double operator()() { return 2.0 * (static_cast<double>(rng_()) / 4294967295.0) - 1.0; }
    Eigen::Vector3d vec3() { return {(*this)(), (*this)(), (*this)()}; }

private:
    std::mt19937 rng_;
};

}  // namespace

// --- variables ---

TEST(So3Variable, RetractRightMultipliesExp) {
    const Eigen::Matrix3d R0 = exp_so3({0.3, -0.5, 0.8});
    So3Variable           var(R0);
    const Eigen::Vector3d delta(0.01, -0.02, 0.03);
    var.retract(delta);
    EXPECT_TRUE(var.R().isApprox(R0 * exp_so3(delta), 1e-12));
}

TEST(So3Variable, ValueRoundTrips) {
    So3Variable var(exp_so3({1.0, -0.7, 0.2}));
    So3Variable other;
    other.set_value(var.value());
    EXPECT_TRUE(other.R().isApprox(var.R(), 1e-15));
}

TEST(So3Variable, RejectsInvalidInput) {
    const Eigen::Matrix3d not_a_rotation = Eigen::Matrix3d::Zero();
    EXPECT_THROW(So3Variable{not_a_rotation}, std::invalid_argument);
    So3Variable var;
    EXPECT_THROW(var.retract(Eigen::VectorXd::Zero(2)), std::invalid_argument);
}

TEST(VectorVariable, RetractIsAdditive) {
    VectorVariable var(Eigen::Vector3d(1.0, 2.0, 3.0));
    var.retract(Eigen::Vector3d(0.1, -0.1, 0.5));
    EXPECT_TRUE(var.vec().isApprox(Eigen::Vector3d(1.1, 1.9, 3.5), 1e-15));
}

// --- factor Jacobians vs numeric differentiation ---

TEST(FactorJacobians, So3PriorMatchesNumeric) {
    So3Variable    var(exp_so3({0.4, -0.1, 0.6}));
    So3PriorFactor factor(&var, exp_so3({0.1, 0.2, -0.3}));
    EXPECT_LT(max_jacobian_error(factor), 1e-8);
}

TEST(FactorJacobians, VectorPriorMatchesNumeric) {
    VectorVariable    var(Eigen::Vector3d(1.0, -2.0, 0.5));
    VectorPriorFactor factor(&var, Eigen::Vector3d(0.2, 0.1, -0.4));
    EXPECT_LT(max_jacobian_error(factor), 1e-9);
}

TEST(FactorJacobians, PointAlignmentMatchesNumeric) {
    So3Variable          rot(exp_so3({-0.2, 0.5, 0.3}));
    VectorVariable       trans(Eigen::Vector3d(1.0, -0.5, 0.8));
    PointAlignmentFactor factor(&rot, &trans, {0.7, -1.2, 0.4}, {0.5, 0.5, -0.9});
    EXPECT_LT(max_jacobian_error(factor), 1e-8);
}

TEST(FactorJacobians, RelativePoseMatchesNumeric) {
    So3Variable    R_i(exp_so3({0.2, -0.4, 0.1}));
    VectorVariable t_i(Eigen::Vector3d(0.5, 1.0, -0.3));
    So3Variable    R_j(exp_so3({-0.3, 0.2, 0.5}));
    VectorVariable t_j(Eigen::Vector3d(-0.7, 0.4, 1.2));

    RelativePoseFactor factor(&R_i, &t_i, &R_j, &t_j, exp_so3({0.05, -0.1, 0.2}),
                              Eigen::Vector3d(0.3, -0.2, 0.6));
    EXPECT_LT(max_jacobian_error(factor), 1e-8);
}

// --- problem assembly ---

TEST(Problem, AssignsOffsetsInRegistrationOrder) {
    So3Variable    R;
    VectorVariable t(Eigen::Vector3d::Zero());
    Problem        problem;
    problem.add_variable(&R);
    problem.add_variable(&t);
    EXPECT_EQ(problem.tangent_offset(&R), 0);
    EXPECT_EQ(problem.tangent_offset(&t), 3);
    EXPECT_EQ(problem.total_tangent_dim(), 6);
}

TEST(Problem, RejectsDuplicateAndUnregistered) {
    So3Variable R;
    Problem     problem;
    problem.add_variable(&R);
    EXPECT_THROW(problem.add_variable(&R), std::invalid_argument);

    So3Variable    other(exp_so3({0.1, 0.0, 0.0}));
    So3PriorFactor factor(&other, Eigen::Matrix3d::Identity());
    EXPECT_THROW(problem.add_factor(&factor), std::invalid_argument);
}

TEST(Problem, NormalEquationsAreSymmetric) {
    So3Variable          rot(exp_so3({0.3, 0.1, -0.2}));
    VectorVariable       trans(Eigen::Vector3d(0.5, -0.5, 1.0));
    PointAlignmentFactor factor(&rot, &trans, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0});

    Problem problem;
    problem.add_variable(&rot);
    problem.add_variable(&trans);
    problem.add_factor(&factor);

    Eigen::MatrixXd H;
    Eigen::VectorXd b;
    problem.build_normal_equations(H, b);
    EXPECT_EQ(H.rows(), 6);
    EXPECT_EQ(H.cols(), 6);
    EXPECT_EQ(b.size(), 6);
    EXPECT_TRUE(H.isApprox(H.transpose(), 1e-12));
    EXPECT_TRUE(H.allFinite());
}

// --- Levenberg-Marquardt on toy problems ---

TEST(LevenbergMarquardt, RecoversRigidAlignment) {
    // GT transform T_A_B; noiseless correspondences q = R_gt p + t_gt.
    const Eigen::Matrix3d R_gt = exp_so3({0.3, -0.2, 0.5});
    const Eigen::Vector3d t_gt(1.0, -0.5, 0.8);

    Uniform                           uniform(42);
    So3Variable                       rot(R_gt * exp_so3({0.3, -0.25, 0.2}));  // perturbed init
    VectorVariable                    trans(t_gt + Eigen::Vector3d(0.5, 0.4, -0.6));
    std::vector<PointAlignmentFactor> factors;
    factors.reserve(20);
    for (int i = 0; i < 20; ++i) {
        const Eigen::Vector3d p = uniform.vec3();
        factors.emplace_back(&rot, &trans, p, R_gt * p + t_gt);
    }

    Problem problem;
    problem.add_variable(&rot);
    problem.add_variable(&trans);
    for (PointAlignmentFactor& f : factors) problem.add_factor(&f);

    const LmResult result = slam_core::optim::optimize(problem);

    EXPECT_TRUE(result.converged) << result.message;
    EXPECT_LT(result.final_cost, 1e-14);
    EXPECT_TRUE(is_valid_rotation(rot.R()));
    EXPECT_LT(log_so3(R_gt.transpose() * rot.R()).norm(), 1e-6);
    EXPECT_LT((trans.vec() - t_gt).norm(), 1e-6);
}

TEST(LevenbergMarquardt, RecoversPoseChain) {
    // 5 poses on a curve; exact relative measurements; prior fixes pose 0.
    const int                    n = 5;
    std::vector<Eigen::Matrix3d> R_gt;
    std::vector<Eigen::Vector3d> t_gt;
    for (int i = 0; i < n; ++i) {
        R_gt.push_back(exp_so3(Eigen::Vector3d(0.1, -0.05, 0.2) * i));
        t_gt.push_back(Eigen::Vector3d(1.0 * i, 0.5 * i * i, -0.3 * i));
    }

    Uniform                     uniform(7);
    std::vector<So3Variable>    rots;
    std::vector<VectorVariable> transes;
    rots.reserve(n);
    transes.reserve(n);
    for (int i = 0; i < n; ++i) {
        // Noisy init (pose 0 exact; the prior pins the gauge there).
        const Eigen::Vector3d dR =
            (i == 0) ? Eigen::Vector3d::Zero().eval() : (0.2 * uniform.vec3()).eval();
        const Eigen::Vector3d dt =
            (i == 0) ? Eigen::Vector3d::Zero().eval() : (0.5 * uniform.vec3()).eval();
        rots.emplace_back(R_gt[i] * exp_so3(dR));
        transes.emplace_back(t_gt[i] + dt);
    }

    So3PriorFactor    rot_prior(&rots[0], R_gt[0]);
    VectorPriorFactor trans_prior(&transes[0], t_gt[0]);

    std::vector<RelativePoseFactor> odom;
    odom.reserve(n - 1);
    for (int i = 0; i + 1 < n; ++i) {
        const Eigen::Matrix3d R_meas = R_gt[i].transpose() * R_gt[i + 1];
        const Eigen::Vector3d t_meas = R_gt[i].transpose() * (t_gt[i + 1] - t_gt[i]);
        odom.emplace_back(&rots[i], &transes[i], &rots[i + 1], &transes[i + 1], R_meas, t_meas);
    }

    Problem problem;
    for (int i = 0; i < n; ++i) {
        problem.add_variable(&rots[i]);
        problem.add_variable(&transes[i]);
    }
    problem.add_factor(&rot_prior);
    problem.add_factor(&trans_prior);
    for (RelativePoseFactor& f : odom) problem.add_factor(&f);

    const LmResult result = slam_core::optim::optimize(problem);

    EXPECT_TRUE(result.converged) << result.message;
    EXPECT_LT(result.final_cost, 1e-14);
    for (int i = 0; i < n; ++i) {
        EXPECT_TRUE(is_valid_rotation(rots[i].R())) << "pose " << i;
        EXPECT_LT(log_so3(R_gt[i].transpose() * rots[i].R()).norm(), 1e-6) << "pose " << i;
        EXPECT_LT((transes[i].vec() - t_gt[i]).norm(), 1e-6) << "pose " << i;
    }
}

TEST(LevenbergMarquardt, CostIsMonotonicOverAcceptedSteps) {
    const Eigen::Matrix3d R_gt = exp_so3({0.5, 0.2, -0.4});
    So3Variable           rot(R_gt * exp_so3({0.4, -0.3, 0.3}));
    So3PriorFactor        prior(&rot, R_gt);

    Problem problem;
    problem.add_variable(&rot);
    problem.add_factor(&prior);

    std::vector<double> costs;
    LmOptions           options;
    options.iteration_callback = [&](const slam_core::optim::LmIterationSummary& s) {
        if (s.step_accepted) costs.push_back(s.cost);
    };

    const LmResult result = slam_core::optim::optimize(problem, options);
    EXPECT_TRUE(result.converged);
    ASSERT_GE(costs.size(), 1u);
    for (std::size_t i = 1; i < costs.size(); ++i) {
        EXPECT_LE(costs[i], costs[i - 1]);
    }
    EXPECT_LE(result.final_cost, result.initial_cost);
}

TEST(LevenbergMarquardt, AlreadyOptimalConvergesImmediately) {
    const Eigen::Matrix3d R_gt = exp_so3({0.1, 0.2, 0.3});
    So3Variable           rot(R_gt);
    So3PriorFactor        prior(&rot, R_gt);

    Problem problem;
    problem.add_variable(&rot);
    problem.add_factor(&prior);

    const LmResult result = slam_core::optim::optimize(problem);
    EXPECT_TRUE(result.converged) << result.message;
    EXPECT_LT(result.final_cost, 1e-20);
    EXPECT_TRUE(rot.R().isApprox(R_gt, 1e-12));
}

TEST(LevenbergMarquardt, EmptyProblemIsTriviallyConverged) {
    Problem        problem;
    const LmResult result = slam_core::optim::optimize(problem);
    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.iterations, 0);
}
