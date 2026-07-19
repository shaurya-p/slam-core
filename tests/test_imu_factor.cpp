#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>
#include <Eigen/Core>

#include "slam_core/factors/imu_factor.hpp"
#include "slam_core/factors/prior_factors.hpp"
#include "slam_core/geometry/so3.hpp"
#include "slam_core/imu/imu_state.hpp"
#include "slam_core/imu/preintegration.hpp"
#include "slam_core/optim/levenberg_marquardt.hpp"
#include "slam_core/optim/problem.hpp"

#include "numeric_jacobian.hpp"

using slam_core::factors::BiasRandomWalkFactor;
using slam_core::factors::ImuFactor;
using slam_core::factors::So3PriorFactor;
using slam_core::factors::VectorPriorFactor;
using slam_core::geometry::log_so3;
using slam_core::imu::ImuMeasurement;
using slam_core::imu::ImuNoiseParams;
using slam_core::imu::ImuState;
using slam_core::imu::PreintegratedImu;
using slam_core::optim::Problem;
using slam_core::optim::So3Variable;
using slam_core::optim::VectorVariable;

namespace {

const Eigen::Vector3d kGravityW(0.0, 0.0, -9.81);

// Time-varying measurement stream. The trajectory it implies is defined by
// ZOH propagation itself, so propagated states are exact ground truth for
// the preintegration model (no discretization mismatch).
std::vector<ImuMeasurement> make_measurements(int                    n,
                                              double                 dt,
                                              const Eigen::Vector3d& bias_g,
                                              const Eigen::Vector3d& bias_a) {
    std::vector<ImuMeasurement> seq;
    seq.reserve(n);
    for (int i = 0; i < n; ++i) {
        const double   t = i * dt;
        ImuMeasurement m;
        m.timestamp_s = t;
        // "True" body rates / specific force, plus the sensor bias.
        m.gyro_radps =
            Eigen::Vector3d{0.4 * std::sin(3.0 * t), -0.3, 0.6 * std::cos(2.0 * t)} + bias_g;
        m.accel_mps2 =
            Eigen::Vector3d{1.5 * std::cos(t), -1.0 * std::sin(2.0 * t), 9.81 + 0.5 * t} + bias_a;
        seq.push_back(m);
    }
    return seq;
}

// Propagates ImuState over measurements [begin, end) with the true bias.
ImuState propagate_states(const ImuState&                    start,
                          const std::vector<ImuMeasurement>& seq,
                          std::size_t                        begin,
                          std::size_t                        end,
                          double                             dt) {
    ImuState s = start;
    for (std::size_t i = begin; i < end; ++i) {
        s = propagate_imu_state(s, seq[i], kGravityW, dt);
    }
    return s;
}

ImuState make_initial_state(const Eigen::Vector3d& bias_g, const Eigen::Vector3d& bias_a) {
    ImuState s;
    s.timestamp_s     = 0.0;
    s.R_W_B           = Eigen::Matrix3d::Identity();
    s.p_W_B           = Eigen::Vector3d::Zero();
    s.v_W_B           = Eigen::Vector3d::Zero();
    s.gyro_bias_radps = bias_g;
    s.accel_bias_mps2 = bias_a;
    return s;
}

PreintegratedImu preintegrate(const std::vector<ImuMeasurement>& seq,
                              std::size_t                        begin,
                              std::size_t                        end,
                              double                             dt,
                              const Eigen::Vector3d&             bg_lin,
                              const Eigen::Vector3d&             ba_lin,
                              const ImuNoiseParams&              noise) {
    PreintegratedImu p;
    for (std::size_t i = begin; i < end; ++i) {
        integrate(p, seq[i], bg_lin, ba_lin, dt, noise);
    }
    return p;
}

}  // namespace

TEST(ImuFactor, ResidualZeroAtConsistentStates) {
    const double          dt = 0.005;
    const Eigen::Vector3d bg(0.02, -0.01, 0.03);
    const Eigen::Vector3d ba(0.1, -0.05, 0.15);
    const auto            seq = make_measurements(100, dt, bg, ba);
    const ImuNoiseParams  noise{1e-3, 1e-2};

    const ImuState s0 = make_initial_state(bg, ba);
    const ImuState s1 = propagate_states(s0, seq, 0, seq.size(), dt);

    // Preintegrated at the true bias: residual must vanish at the true states.
    const PreintegratedImu preint = preintegrate(seq, 0, seq.size(), dt, bg, ba, noise);

    So3Variable    R_i(s0.R_W_B), R_j(s1.R_W_B);
    VectorVariable v_i(s0.v_W_B), v_j(s1.v_W_B);
    VectorVariable p_i(s0.p_W_B), p_j(s1.p_W_B);
    VectorVariable bg_i(bg), ba_i(ba);

    const ImuFactor factor(&R_i, &v_i, &p_i, &R_j, &v_j, &p_j, &bg_i, &ba_i, preint, kGravityW, bg,
                           ba);
    EXPECT_LT(factor.residual().norm(), 1e-6);
}

TEST(ImuFactor, BiasCorrectionKeepsResidualSmall) {
    // Preintegrated at zero bias; the factor's first-order correction must
    // absorb the true bias to O(|bias|^2) once bias variables hold it.
    const double          dt = 0.005;
    const Eigen::Vector3d bg(0.02, -0.01, 0.03);
    const Eigen::Vector3d ba(0.1, -0.05, 0.15);
    const auto            seq = make_measurements(100, dt, bg, ba);
    const ImuNoiseParams  noise{1e-3, 1e-2};

    const ImuState s0 = make_initial_state(bg, ba);
    const ImuState s1 = propagate_states(s0, seq, 0, seq.size(), dt);

    const PreintegratedImu preint = preintegrate(seq, 0, seq.size(), dt, Eigen::Vector3d::Zero(),
                                                 Eigen::Vector3d::Zero(), noise);

    So3Variable    R_i(s0.R_W_B), R_j(s1.R_W_B);
    VectorVariable v_i(s0.v_W_B), v_j(s1.v_W_B);
    VectorVariable p_i(s0.p_W_B), p_j(s1.p_W_B);
    VectorVariable bg_i(bg), ba_i(ba);

    const ImuFactor corrected(&R_i, &v_i, &p_i, &R_j, &v_j, &p_j, &bg_i, &ba_i, preint, kGravityW,
                              Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());

    VectorVariable  bg_zero(Eigen::Vector3d::Zero()), ba_zero(Eigen::Vector3d::Zero());
    const ImuFactor uncorrected(&R_i, &v_i, &p_i, &R_j, &v_j, &p_j, &bg_zero, &ba_zero, preint,
                                kGravityW, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());

    // Whitened comparison: correction must shrink the residual dramatically.
    EXPECT_GT(uncorrected.residual().norm(), 50.0 * corrected.residual().norm());
}

TEST(ImuFactor, JacobiansMatchNumeric) {
    const double          dt = 0.005;
    const Eigen::Vector3d bg(0.02, -0.01, 0.03);
    const Eigen::Vector3d ba(0.1, -0.05, 0.15);
    const auto            seq = make_measurements(60, dt, bg, ba);
    // Moderate noise keeps sqrt-information well conditioned for the
    // finite-difference comparison.
    const ImuNoiseParams noise{1e-2, 5e-2};

    const ImuState s0 = make_initial_state(bg, ba);
    const ImuState s1 = propagate_states(s0, seq, 0, seq.size(), dt);

    const PreintegratedImu preint = preintegrate(seq, 0, seq.size(), dt, Eigen::Vector3d::Zero(),
                                                 Eigen::Vector3d::Zero(), noise);

    // Evaluate away from the optimum: perturbed states, nonzero delta-bias.
    So3Variable    R_i(s0.R_W_B * slam_core::geometry::exp_so3({0.02, -0.01, 0.03}));
    So3Variable    R_j(s1.R_W_B * slam_core::geometry::exp_so3({-0.01, 0.02, 0.01}));
    VectorVariable v_i(s0.v_W_B + Eigen::Vector3d(0.1, -0.2, 0.05));
    VectorVariable v_j(s1.v_W_B + Eigen::Vector3d(-0.05, 0.1, 0.2));
    VectorVariable p_i(s0.p_W_B + Eigen::Vector3d(0.2, 0.1, -0.1));
    VectorVariable p_j(s1.p_W_B + Eigen::Vector3d(-0.1, 0.3, 0.2));
    VectorVariable bg_i(Eigen::Vector3d(0.01, -0.005, 0.02));
    VectorVariable ba_i(Eigen::Vector3d(0.05, 0.02, -0.03));

    const ImuFactor factor(&R_i, &v_i, &p_i, &R_j, &v_j, &p_j, &bg_i, &ba_i, preint, kGravityW,
                           Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());

    double max_abs = 0.0;
    for (const Eigen::MatrixXd& J : factor.jacobians()) {
        max_abs = std::max(max_abs, J.cwiseAbs().maxCoeff());
    }
    EXPECT_LT(slam_core::testing::max_jacobian_error(factor), 1e-5 * std::max(1.0, max_abs));
}

TEST(BiasRandomWalkFactor, ResidualAndJacobiansMatchNumeric) {
    VectorVariable bg_i(Eigen::Vector3d(0.01, -0.02, 0.03));
    VectorVariable ba_i(Eigen::Vector3d(0.1, 0.05, -0.08));
    VectorVariable bg_j(Eigen::Vector3d(0.012, -0.018, 0.031));
    VectorVariable ba_j(Eigen::Vector3d(0.11, 0.04, -0.07));

    const BiasRandomWalkFactor factor(&bg_i, &ba_i, &bg_j, &ba_j, 1.9393e-5, 3e-3, 0.25);

    EXPECT_EQ(factor.residual().size(), 6);
    EXPECT_LT(slam_core::testing::max_jacobian_error(factor), 1e-3);

    // Equal biases -> zero residual.
    bg_j.set_value(bg_i.value());
    ba_j.set_value(ba_i.value());
    EXPECT_LT(factor.residual().norm(), 1e-12);
}

TEST(ImuFactor, ChainOptimizationRecoversStatesAndBias) {
    // 5 keyframes, 0.25 s apart; true nonzero bias; preintegration at zero
    // bias; priors pin both end poses/velocities. The optimizer must pull
    // interior states onto GT and discover the bias.
    const double          dt      = 0.005;
    const int             n_kf    = 5;
    const int             per_seg = 50;  // 0.25 s per interval
    const Eigen::Vector3d bg_true(0.02, -0.01, 0.03);
    const Eigen::Vector3d ba_true(0.1, -0.05, 0.15);
    const ImuNoiseParams  noise{1.7e-4, 2.0e-3};  // EuRoC densities

    const auto seq = make_measurements(per_seg * (n_kf - 1) + 1, dt, bg_true, ba_true);

    // Ground-truth states at keyframes (ZOH-exact).
    std::vector<ImuState> gt;
    gt.push_back(make_initial_state(bg_true, ba_true));
    for (int k = 1; k < n_kf; ++k) {
        gt.push_back(propagate_states(gt.back(), seq, (k - 1) * per_seg, k * per_seg, dt));
    }

    // Dead-reckoned initialization with zero bias (drifts).
    std::vector<ImuState> init;
    init.push_back(make_initial_state(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()));
    for (int k = 1; k < n_kf; ++k) {
        init.push_back(propagate_states(init.back(), seq, (k - 1) * per_seg, k * per_seg, dt));
    }

    std::vector<So3Variable>    R;
    std::vector<VectorVariable> v, p, bgs, bas;
    R.reserve(n_kf);
    v.reserve(n_kf);
    p.reserve(n_kf);
    bgs.reserve(n_kf);
    bas.reserve(n_kf);
    for (int k = 0; k < n_kf; ++k) {
        R.emplace_back(init[k].R_W_B);
        v.emplace_back(init[k].v_W_B);
        p.emplace_back(init[k].p_W_B);
        bgs.emplace_back(Eigen::Vector3d::Zero());
        bas.emplace_back(Eigen::Vector3d::Zero());
    }

    std::vector<PreintegratedImu>     preints;
    std::vector<ImuFactor>            imu_factors;
    std::vector<BiasRandomWalkFactor> walk_factors;
    preints.reserve(n_kf - 1);
    imu_factors.reserve(n_kf - 1);
    walk_factors.reserve(n_kf - 1);
    for (int k = 0; k + 1 < n_kf; ++k) {
        preints.push_back(preintegrate(seq, k * per_seg, (k + 1) * per_seg, dt,
                                       Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), noise));
    }
    for (int k = 0; k + 1 < n_kf; ++k) {
        imu_factors.emplace_back(&R[k], &v[k], &p[k], &R[k + 1], &v[k + 1], &p[k + 1], &bgs[k],
                                 &bas[k], preints[k], kGravityW, Eigen::Vector3d::Zero(),
                                 Eigen::Vector3d::Zero());
        walk_factors.emplace_back(&bgs[k], &bas[k], &bgs[k + 1], &bas[k + 1], 1.9393e-5, 3e-3,
                                  per_seg * dt);
    }

    // Gauge/anchor priors: both end poses and velocities, tight.
    So3PriorFactor    prior_R0(&R[0], gt[0].R_W_B, 1e-4);
    VectorPriorFactor prior_v0(&v[0], gt[0].v_W_B, 1e-4);
    VectorPriorFactor prior_p0(&p[0], gt[0].p_W_B, 1e-4);
    So3PriorFactor    prior_Rn(&R[n_kf - 1], gt[n_kf - 1].R_W_B, 1e-4);
    VectorPriorFactor prior_vn(&v[n_kf - 1], gt[n_kf - 1].v_W_B, 1e-4);
    VectorPriorFactor prior_pn(&p[n_kf - 1], gt[n_kf - 1].p_W_B, 1e-4);

    Problem problem;
    for (int k = 0; k < n_kf; ++k) {
        problem.add_variable(&R[k]);
        problem.add_variable(&v[k]);
        problem.add_variable(&p[k]);
        problem.add_variable(&bgs[k]);
        problem.add_variable(&bas[k]);
    }
    for (auto& f : imu_factors) problem.add_factor(&f);
    for (auto& f : walk_factors) problem.add_factor(&f);
    problem.add_factor(&prior_R0);
    problem.add_factor(&prior_v0);
    problem.add_factor(&prior_p0);
    problem.add_factor(&prior_Rn);
    problem.add_factor(&prior_vn);
    problem.add_factor(&prior_pn);

    slam_core::optim::LmOptions options;
    options.max_iterations                  = 100;
    const slam_core::optim::LmResult result = slam_core::optim::optimize(problem, options);
    EXPECT_TRUE(result.converged) << result.message;

    // Interior states: optimized must beat dead reckoning decisively.
    double max_opt_pos_err = 0.0, max_dr_pos_err = 0.0;
    for (int k = 1; k + 1 < n_kf; ++k) {
        max_opt_pos_err = std::max(max_opt_pos_err, (p[k].vec() - gt[k].p_W_B).norm());
        max_dr_pos_err  = std::max(max_dr_pos_err, (init[k].p_W_B - gt[k].p_W_B).norm());
        EXPECT_LT(log_so3(gt[k].R_W_B.transpose() * R[k].R()).norm(), 5e-3) << "keyframe " << k;
        EXPECT_LT((v[k].vec() - gt[k].v_W_B).norm(), 2e-2) << "keyframe " << k;
    }
    EXPECT_LT(max_opt_pos_err, 5e-3);
    EXPECT_GT(max_dr_pos_err, 20.0 * max_opt_pos_err);

    // Bias recovery: the graph must discover the true bias from scratch.
    for (int k = 0; k < n_kf; ++k) {
        EXPECT_LT((bgs[k].vec() - bg_true).norm(), 2e-3) << "keyframe " << k;
        EXPECT_LT((bas[k].vec() - ba_true).norm(), 2e-2) << "keyframe " << k;
    }
}
