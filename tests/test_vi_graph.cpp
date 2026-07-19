// Joint visual-inertial graph: IMU + reprojection + bias-walk factors,
// gauge fixed only at the first pose. Verifies that vision anchors the far
// end (no last-keyframe prior, unlike the IMU-only chain) and that scale,
// gravity direction, velocity, and biases are jointly observable.

#include <cstdint>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include <Eigen/Core>

#include "slam_core/camera/pinhole_camera.hpp"
#include "slam_core/factors/imu_factor.hpp"
#include "slam_core/factors/prior_factors.hpp"
#include "slam_core/factors/reprojection_factor.hpp"
#include "slam_core/geometry/so3.hpp"
#include "slam_core/imu/imu_state.hpp"
#include "slam_core/imu/preintegration.hpp"
#include "slam_core/optim/levenberg_marquardt.hpp"
#include "slam_core/optim/problem.hpp"

using slam_core::camera::PinholeCamera;
using slam_core::factors::BiasRandomWalkFactor;
using slam_core::factors::ImuFactor;
using slam_core::factors::ReprojectionFactor;
using slam_core::factors::So3PriorFactor;
using slam_core::factors::VectorPriorFactor;
using slam_core::geometry::exp_so3;
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
const PinholeCamera   kCam(458.654, 457.296, 367.215, 248.375);
// Camera looks along body +x: R_B_C maps camera z-forward onto body x.
const Eigen::Matrix3d kRBC = [] {
    Eigen::Matrix3d R;
    R << 0, 0, 1, -1, 0, 0, 0, -1, 0;
    return R;
}();
const Eigen::Vector3d kTBC(0.05, -0.02, 0.01);

class Uniform {
public:
    explicit Uniform(std::uint32_t seed) : rng_(seed) {}
    double operator()() { return 2.0 * (static_cast<double>(rng_()) / 4294967295.0) - 1.0; }
    Eigen::Vector3d vec3() { return {(*this)(), (*this)(), (*this)()}; }

private:
    std::mt19937 rng_;
};

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
        m.gyro_radps =
            Eigen::Vector3d{0.3 * std::sin(2.0 * t), -0.2, 0.4 * std::cos(3.0 * t)} + bias_g;
        m.accel_mps2 =
            Eigen::Vector3d{1.0 * std::cos(t), -0.8 * std::sin(2.0 * t), 9.81 + 0.4 * t} + bias_a;
        seq.push_back(m);
    }
    return seq;
}

Eigen::Vector2d project_world(const Eigen::Matrix3d& R_W_B,
                              const Eigen::Vector3d& p_W_B,
                              const Eigen::Vector3d& p_W_L) {
    const Eigen::Vector3d p_B = R_W_B.transpose() * (p_W_L - p_W_B);
    const Eigen::Vector3d p_C = kRBC.transpose() * (p_B - kTBC);
    return kCam.project(p_C);
}

}  // namespace

TEST(ViGraph, JointOptimizationWithoutEndPriors) {
    const double          dt      = 0.005;
    const int             n_kf    = 5;
    const int             per_seg = 50;  // 0.25 s per interval
    const Eigen::Vector3d bg_true(0.02, -0.01, 0.03);
    const Eigen::Vector3d ba_true(0.1, -0.05, 0.15);
    const ImuNoiseParams  noise{1.7e-4, 2.0e-3};

    const auto seq = make_measurements(per_seg * (n_kf - 1) + 1, dt, bg_true, ba_true);

    // ZOH-exact GT keyframe states.
    std::vector<ImuState> gt(1);
    gt[0].timestamp_s     = 0.0;
    gt[0].R_W_B           = Eigen::Matrix3d::Identity();
    gt[0].p_W_B           = Eigen::Vector3d::Zero();
    gt[0].v_W_B           = Eigen::Vector3d::Zero();
    gt[0].gyro_bias_radps = bg_true;
    gt[0].accel_bias_mps2 = ba_true;
    for (int k = 1; k < n_kf; ++k) {
        ImuState s = gt.back();
        for (int i = (k - 1) * per_seg; i < k * per_seg; ++i) {
            s = propagate_imu_state(s, seq[i], kGravityW, dt);
        }
        gt.push_back(s);
    }

    // Landmarks in front of the camera (body +x) of random keyframes.
    Uniform                      uniform(2026);
    const int                    n_landmarks = 20;
    std::vector<Eigen::Vector3d> L_gt;
    for (int i = 0; i < n_landmarks; ++i) {
        const int             k     = static_cast<int>((0.5 + 0.5 * uniform()) * (n_kf - 1));
        const double          depth = 4.0 + 2.0 * uniform();
        const Eigen::Vector3d offset_B(depth, 1.5 * uniform(), 1.0 * uniform());
        L_gt.push_back(gt[k].p_W_B + gt[k].R_W_B * offset_B);
    }

    // Variables: perturbed states, zero biases, perturbed landmarks.
    std::vector<So3Variable>    R;
    std::vector<VectorVariable> v, p, bgs, bas, L;
    for (int k = 0; k < n_kf; ++k) {
        const bool first = (k == 0);
        R.emplace_back(gt[k].R_W_B * exp_so3(first ? Eigen::Vector3d::Zero().eval()
                                                   : (0.03 * uniform.vec3()).eval()));
        v.emplace_back(gt[k].v_W_B +
                       (first ? Eigen::Vector3d::Zero().eval() : (0.2 * uniform.vec3()).eval()));
        p.emplace_back(gt[k].p_W_B +
                       (first ? Eigen::Vector3d::Zero().eval() : (0.3 * uniform.vec3()).eval()));
        bgs.emplace_back(Eigen::Vector3d::Zero());
        bas.emplace_back(Eigen::Vector3d::Zero());
    }
    for (int i = 0; i < n_landmarks; ++i) {
        L.emplace_back(L_gt[i] + 0.3 * uniform.vec3());
    }

    // IMU + bias-walk factors.
    std::vector<PreintegratedImu> preints;
    for (int k = 0; k + 1 < n_kf; ++k) {
        PreintegratedImu pre;
        for (int i = k * per_seg; i < (k + 1) * per_seg; ++i) {
            integrate(pre, seq[i], Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), dt, noise);
        }
        preints.push_back(pre);
    }
    std::vector<ImuFactor>            imu_factors;
    std::vector<BiasRandomWalkFactor> walk_factors;
    for (int k = 0; k + 1 < n_kf; ++k) {
        imu_factors.emplace_back(&R[k], &v[k], &p[k], &R[k + 1], &v[k + 1], &p[k + 1], &bgs[k],
                                 &bas[k], preints[k], kGravityW, Eigen::Vector3d::Zero(),
                                 Eigen::Vector3d::Zero());
        walk_factors.emplace_back(&bgs[k], &bas[k], &bgs[k + 1], &bas[k + 1], 1.9393e-5, 3e-3,
                                  per_seg * dt);
    }

    // Reprojection factors: every landmark seen from every keyframe with
    // positive depth (noise-free pixels).
    std::vector<ReprojectionFactor> reproj_factors;
    for (int k = 0; k < n_kf; ++k) {
        for (int i = 0; i < n_landmarks; ++i) {
            const Eigen::Vector3d p_C =
                kRBC.transpose() * (gt[k].R_W_B.transpose() * (L_gt[i] - gt[k].p_W_B) - kTBC);
            if (p_C.z() < 0.5) continue;
            reproj_factors.emplace_back(&R[k], &p[k], &L[i], kCam, kRBC, kTBC,
                                        project_world(gt[k].R_W_B, gt[k].p_W_B, L_gt[i]), 1.0);
        }
    }
    ASSERT_GT(reproj_factors.size(), 40u);

    // Gauge only: first pose. No priors on the last keyframe or velocities.
    So3PriorFactor    prior_R0(&R[0], gt[0].R_W_B, 1e-4);
    VectorPriorFactor prior_p0(&p[0], gt[0].p_W_B, 1e-4);

    Problem problem;
    for (int k = 0; k < n_kf; ++k) {
        problem.add_variable(&R[k]);
        problem.add_variable(&v[k]);
        problem.add_variable(&p[k]);
        problem.add_variable(&bgs[k]);
        problem.add_variable(&bas[k]);
    }
    for (auto& l : L) problem.add_variable(&l);
    for (auto& f : imu_factors) problem.add_factor(&f);
    for (auto& f : walk_factors) problem.add_factor(&f);
    for (auto& f : reproj_factors) problem.add_factor(&f);
    problem.add_factor(&prior_R0);
    problem.add_factor(&prior_p0);

    slam_core::optim::LmOptions options;
    options.max_iterations                  = 200;
    const slam_core::optim::LmResult result = slam_core::optim::optimize(problem, options);
    EXPECT_TRUE(result.converged) << result.message;

    for (int k = 0; k < n_kf; ++k) {
        EXPECT_LT((p[k].vec() - gt[k].p_W_B).norm(), 5e-3) << "keyframe " << k;
        EXPECT_LT(log_so3(gt[k].R_W_B.transpose() * R[k].R()).norm(), 2e-3) << "keyframe " << k;
        EXPECT_LT((v[k].vec() - gt[k].v_W_B).norm(), 2e-2) << "keyframe " << k;
        EXPECT_LT((bgs[k].vec() - bg_true).norm(), 3e-3) << "keyframe " << k;
        EXPECT_LT((bas[k].vec() - ba_true).norm(), 5e-2) << "keyframe " << k;
    }
    for (int i = 0; i < n_landmarks; ++i) {
        EXPECT_LT((L[i].vec() - L_gt[i]).norm(), 2e-2) << "landmark " << i;
    }
}
