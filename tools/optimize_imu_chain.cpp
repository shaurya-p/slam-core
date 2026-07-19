// Optimize a keyframe chain on EuRoC with preintegrated IMU factors.
//
// Usage:
//   optimize_imu_chain <euroc_imu_csv> <output_csv>
//                      [--start-s S]            offset into GT coverage (default 5)
//                      [--duration-s D]         chain length in seconds (default 20)
//                      [--keyframe-spacing-s K] keyframe spacing (default 0.5)
//                      [--gyro-noise G] [--accel-noise A]        (densities)
//                      [--gyro-walk GW] [--accel-walk AW]        (walk densities)
//
// GT path is derived from the IMU path (../state_groundtruth_estimate0/data.csv).
//
// Graph: per-keyframe (R_W_B, v_W_B, p_W_B, bg, ba); ImuFactor between
// consecutive keyframes (preintegrated at zero bias, whitened by the
// propagated covariance); bias random-walk factors; tight priors on the
// first and last pose/velocity from GT. Biases start at zero — the graph
// must discover them.
//
// Initialization: dead reckoning by chaining raw preintegrated deltas from
// the GT start state (this is also the "before" baseline in the output).
//
// Output CSV per keyframe:
//   timestamp_s,
//   gt_p_x/y/z, gt_q_w/x/y/z,
//   dr_p_x/y/z,  dr_pos_err_m,  dr_rot_err_deg,
//   opt_p_x/y/z, opt_pos_err_m, opt_rot_err_deg,
//   opt_v_x/y/z, bg_x/y/z, ba_x/y/z
//
// Default noise densities are EuRoC's ADIS16448 values (sensor.yaml).

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "slam_core/factors/imu_factor.hpp"
#include "slam_core/factors/prior_factors.hpp"
#include "slam_core/geometry/so3.hpp"
#include "slam_core/imu/preintegration.hpp"
#include "slam_core/io/euroc_csv.hpp"
#include "slam_core/optim/levenberg_marquardt.hpp"
#include "slam_core/optim/problem.hpp"

namespace {

constexpr double kPi = 3.14159265358979323846;

struct Options {
    double start_s     = 5.0;
    double duration_s  = 20.0;
    double spacing_s   = 0.5;
    double gyro_noise  = 1.6968e-4;  // rad/s/sqrt(Hz)
    double accel_noise = 2.0e-3;     // m/s^2/sqrt(Hz)
    double gyro_walk   = 1.9393e-5;  // rad/s^2/sqrt(Hz)
    double accel_walk  = 3.0e-3;     // m/s^3/sqrt(Hz)
};

bool parse_double(const char* s, double& out) {
    try {
        out = std::stod(s);
    } catch (...) {
        return false;
    }
    return std::isfinite(out);
}

double rot_err_deg(const Eigen::Matrix3d& R_a, const Eigen::Matrix3d& R_b) {
    return slam_core::geometry::log_so3(R_a.transpose() * R_b).norm() * 180.0 / kPi;
}

}  // namespace

int main(int argc, char* argv[]) {
    const char* usage =
        "Usage: optimize_imu_chain <euroc_imu_csv> <output_csv>"
        " [--start-s S] [--duration-s D] [--keyframe-spacing-s K]"
        " [--gyro-noise G] [--accel-noise A] [--gyro-walk GW] [--accel-walk AW]\n";
    if (argc < 3) {
        std::cerr << usage;
        return EXIT_FAILURE;
    }

    Options opt;
    for (int i = 3; i < argc; ++i) {
        const std::string arg(argv[i]);
        double*           target = nullptr;
        if (arg == "--start-s") target = &opt.start_s;
        else if (arg == "--duration-s") target = &opt.duration_s;
        else if (arg == "--keyframe-spacing-s") target = &opt.spacing_s;
        else if (arg == "--gyro-noise") target = &opt.gyro_noise;
        else if (arg == "--accel-noise") target = &opt.accel_noise;
        else if (arg == "--gyro-walk") target = &opt.gyro_walk;
        else if (arg == "--accel-walk") target = &opt.accel_walk;
        else {
            std::cerr << "Error: unknown argument: " << arg << '\n' << usage;
            return EXIT_FAILURE;
        }
        if (i + 1 >= argc || !parse_double(argv[++i], *target)) {
            std::cerr << "Error: " << arg << " requires a finite numeric value\n";
            return EXIT_FAILURE;
        }
    }
    if (opt.spacing_s <= 0.0 || opt.duration_s <= 2.0 * opt.spacing_s) {
        std::cerr << "Error: need duration-s > 2 * keyframe-spacing-s > 0\n";
        return EXIT_FAILURE;
    }

    // --- load data ---
    std::vector<slam_core::imu::ImuMeasurement> imu;
    std::vector<slam_core::io::EurocGtSample>   gt;
    try {
        namespace fs           = std::filesystem;
        imu                    = slam_core::io::read_euroc_imu_csv(argv[1]);
        const fs::path gt_path = fs::path(argv[1]).parent_path().parent_path() /
                                 "state_groundtruth_estimate0" / "data.csv";
        gt = slam_core::io::read_euroc_gt_csv(gt_path);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }

    // --- pick keyframe indices at IMU sample timestamps ---
    const double t0 = std::max(imu.front().timestamp_s, gt.front().timestamp_s) + opt.start_s;
    const double t_end =
        std::min(t0 + opt.duration_s, std::min(imu.back().timestamp_s, gt.back().timestamp_s));

    std::vector<std::size_t> kf_idx;
    double                   next_t = t0;
    for (std::size_t i = 0; i + 1 < imu.size(); ++i) {
        if (imu[i].timestamp_s > t_end) break;
        if (imu[i].timestamp_s >= next_t) {
            kf_idx.push_back(i);
            next_t = imu[i].timestamp_s + opt.spacing_s;
        }
    }
    const int n_kf = static_cast<int>(kf_idx.size());
    if (n_kf < 3) {
        std::cerr << "Error: fewer than 3 keyframes in the requested window\n";
        return EXIT_FAILURE;
    }

    const Eigen::Vector3d                gravity_W(0.0, 0.0, -9.81);
    const Eigen::Vector3d                zero = Eigen::Vector3d::Zero();
    const slam_core::imu::ImuNoiseParams noise{opt.gyro_noise, opt.accel_noise};

    // --- preintegrate each interval at zero bias ---
    std::vector<slam_core::imu::PreintegratedImu> preints;
    preints.reserve(n_kf - 1);
    for (int k = 0; k + 1 < n_kf; ++k) {
        slam_core::imu::PreintegratedImu p;
        for (std::size_t i = kf_idx[k]; i < kf_idx[k + 1]; ++i) {
            const double dt = imu[i + 1].timestamp_s - imu[i].timestamp_s;
            slam_core::imu::integrate(p, imu[i], zero, zero, dt, noise);
        }
        preints.push_back(p);
    }

    // --- GT at keyframes; dead-reckoned initialization from GT start ---
    std::vector<Eigen::Matrix3d>    gt_R(n_kf);
    std::vector<Eigen::Vector3d>    gt_p(n_kf), gt_v(n_kf);
    std::vector<Eigen::Quaterniond> gt_q(n_kf);
    for (int k = 0; k < n_kf; ++k) {
        const auto& s = slam_core::io::nearest_gt(imu[kf_idx[k]].timestamp_s, gt);
        gt_R[k]       = s.R_W_B();
        gt_q[k]       = s.q_W_B;
        gt_p[k]       = s.p_W_B;
        gt_v[k]       = s.v_W_B;
    }

    std::vector<Eigen::Matrix3d> dr_R(n_kf);
    std::vector<Eigen::Vector3d> dr_p(n_kf), dr_v(n_kf);
    dr_R[0] = gt_R[0];
    dr_p[0] = gt_p[0];
    dr_v[0] = gt_v[0];
    for (int k = 0; k + 1 < n_kf; ++k) {
        const double dt = preints[k].delta_t_s;
        dr_p[k + 1] =
            dr_p[k] + dr_v[k] * dt + 0.5 * gravity_W * dt * dt + dr_R[k] * preints[k].delta_p;
        dr_v[k + 1] = dr_v[k] + gravity_W * dt + dr_R[k] * preints[k].delta_v;
        dr_R[k + 1] = dr_R[k] * preints[k].delta_R;
    }

    // --- build the graph ---
    std::vector<slam_core::optim::So3Variable>    R;
    std::vector<slam_core::optim::VectorVariable> v, p, bg, ba;
    R.reserve(n_kf);
    v.reserve(n_kf);
    p.reserve(n_kf);
    bg.reserve(n_kf);
    ba.reserve(n_kf);
    for (int k = 0; k < n_kf; ++k) {
        R.emplace_back(dr_R[k]);
        v.emplace_back(dr_v[k]);
        p.emplace_back(dr_p[k]);
        bg.emplace_back(zero);
        ba.emplace_back(zero);
    }

    std::vector<slam_core::factors::ImuFactor>            imu_factors;
    std::vector<slam_core::factors::BiasRandomWalkFactor> walk_factors;
    imu_factors.reserve(n_kf - 1);
    walk_factors.reserve(n_kf - 1);
    for (int k = 0; k + 1 < n_kf; ++k) {
        imu_factors.emplace_back(&R[k], &v[k], &p[k], &R[k + 1], &v[k + 1], &p[k + 1], &bg[k],
                                 &ba[k], preints[k], gravity_W, zero, zero);
        walk_factors.emplace_back(&bg[k], &ba[k], &bg[k + 1], &ba[k + 1], opt.gyro_walk,
                                  opt.accel_walk, preints[k].delta_t_s);
    }

    slam_core::factors::So3PriorFactor    prior_R0(&R[0], gt_R[0], 1e-3);
    slam_core::factors::VectorPriorFactor prior_p0(&p[0], gt_p[0], 1e-3);
    slam_core::factors::VectorPriorFactor prior_v0(&v[0], gt_v[0], 1e-2);
    slam_core::factors::So3PriorFactor    prior_Rn(&R[n_kf - 1], gt_R[n_kf - 1], 1e-3);
    slam_core::factors::VectorPriorFactor prior_pn(&p[n_kf - 1], gt_p[n_kf - 1], 1e-3);
    slam_core::factors::VectorPriorFactor prior_vn(&v[n_kf - 1], gt_v[n_kf - 1], 1e-2);

    slam_core::optim::Problem problem;
    for (int k = 0; k < n_kf; ++k) {
        problem.add_variable(&R[k]);
        problem.add_variable(&v[k]);
        problem.add_variable(&p[k]);
        problem.add_variable(&bg[k]);
        problem.add_variable(&ba[k]);
    }
    for (auto& f : imu_factors) problem.add_factor(&f);
    for (auto& f : walk_factors) problem.add_factor(&f);
    problem.add_factor(&prior_R0);
    problem.add_factor(&prior_p0);
    problem.add_factor(&prior_v0);
    problem.add_factor(&prior_Rn);
    problem.add_factor(&prior_pn);
    problem.add_factor(&prior_vn);

    slam_core::optim::LmOptions lm;
    lm.max_iterations                       = 100;
    const slam_core::optim::LmResult result = slam_core::optim::optimize(problem, lm);

    // --- write per-keyframe CSV ---
    const std::filesystem::path out_path(argv[2]);
    if (out_path.has_parent_path()) std::filesystem::create_directories(out_path.parent_path());
    std::ofstream out(out_path);
    if (!out.is_open()) {
        std::cerr << "Error: cannot open output: " << argv[2] << '\n';
        return EXIT_FAILURE;
    }
    out << std::setprecision(12) << std::fixed;
    out << "timestamp_s,gt_p_x,gt_p_y,gt_p_z,gt_q_w,gt_q_x,gt_q_y,gt_q_z"
        << ",dr_p_x,dr_p_y,dr_p_z,dr_pos_err_m,dr_rot_err_deg"
        << ",opt_p_x,opt_p_y,opt_p_z,opt_pos_err_m,opt_rot_err_deg"
        << ",opt_v_x,opt_v_y,opt_v_z,bg_x,bg_y,bg_z,ba_x,ba_y,ba_z\n";

    double dr_rmse = 0.0, opt_rmse = 0.0, dr_max = 0.0, opt_max = 0.0;
    for (int k = 0; k < n_kf; ++k) {
        const double dr_err  = (dr_p[k] - gt_p[k]).norm();
        const double opt_err = (p[k].vec() - gt_p[k]).norm();
        const double dr_rot  = rot_err_deg(gt_R[k], dr_R[k]);
        const double opt_rot = rot_err_deg(gt_R[k], R[k].R());
        dr_rmse += dr_err * dr_err;
        opt_rmse += opt_err * opt_err;
        dr_max  = std::max(dr_max, dr_err);
        opt_max = std::max(opt_max, opt_err);

        out << imu[kf_idx[k]].timestamp_s;
        out << ',' << gt_p[k].x() << ',' << gt_p[k].y() << ',' << gt_p[k].z();
        out << ',' << gt_q[k].w() << ',' << gt_q[k].x() << ',' << gt_q[k].y() << ',' << gt_q[k].z();
        out << ',' << dr_p[k].x() << ',' << dr_p[k].y() << ',' << dr_p[k].z();
        out << ',' << dr_err << ',' << dr_rot;
        out << ',' << p[k].vec().x() << ',' << p[k].vec().y() << ',' << p[k].vec().z();
        out << ',' << opt_err << ',' << opt_rot;
        out << ',' << v[k].vec().x() << ',' << v[k].vec().y() << ',' << v[k].vec().z();
        out << ',' << bg[k].vec().x() << ',' << bg[k].vec().y() << ',' << bg[k].vec().z();
        out << ',' << ba[k].vec().x() << ',' << ba[k].vec().y() << ',' << ba[k].vec().z();
        out << '\n';
    }
    dr_rmse  = std::sqrt(dr_rmse / n_kf);
    opt_rmse = std::sqrt(opt_rmse / n_kf);

    Eigen::Vector3d bg_mean = zero, ba_mean = zero;
    for (int k = 0; k < n_kf; ++k) {
        bg_mean += bg[k].vec();
        ba_mean += ba[k].vec();
    }
    bg_mean /= n_kf;
    ba_mean /= n_kf;

    std::cerr << std::setprecision(6) << std::fixed;
    std::cerr << "optimize_imu_chain:\n"
              << "  keyframes:            " << n_kf << " (spacing " << opt.spacing_s << " s, "
              << (t_end - t0) << " s window)\n"
              << "  LM:                   " << (result.converged ? "converged" : "NOT converged")
              << " in " << result.iterations << " iterations (" << result.message << ")\n"
              << "  cost:                 " << result.initial_cost << " -> " << result.final_cost
              << '\n'
              << "  dead-reckon pos err:  rmse " << dr_rmse << " m, max " << dr_max << " m\n"
              << "  optimized  pos err:   rmse " << opt_rmse << " m, max " << opt_max << " m\n"
              << "  mean gyro bias:       [" << bg_mean.x() << ", " << bg_mean.y() << ", "
              << bg_mean.z() << "] rad/s\n"
              << "  mean accel bias:      [" << ba_mean.x() << ", " << ba_mean.y() << ", "
              << ba_mean.z() << "] m/s^2\n"
              << "  output:               " << argv[2] << '\n';

    return result.converged ? EXIT_SUCCESS : EXIT_FAILURE;
}
