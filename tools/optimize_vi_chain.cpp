// Full visual-inertial optimization on EuRoC: real IMU + real camera
// calibration + synthetic landmark tracks (GT-projected pixels with noise).
//
// Usage:
//   optimize_vi_chain <euroc_imu_csv> <keyframes_csv> <landmarks_csv>
//                     [--start-s S] [--duration-s D] [--keyframe-spacing-s K]
//                     [--n-landmarks N] [--pixel-sigma P] [--seed SEED]
//                     [--pos-perturb M] [--rot-perturb RAD] [--vel-perturb MPS]
//
// GT and cam0 calibration are found relative to the IMU path
// (../state_groundtruth_estimate0/data.csv, ../cam0/sensor.yaml).
//
// Landmarks are synthesized by unprojecting random in-image pixels at
// random depths through GT poses (real T_B_C and intrinsics), observed
// from every keyframe where they fall inside the image with positive
// depth, with Gaussian pixel noise. Landmarks need >= 3 observations.
//
// Graph: per-keyframe (R, v, p, bg, ba); IMU factors (zero linearization
// bias, EuRoC noise); bias random-walk; reprojection factors; gauge
// priors on the FIRST pose only — no last-keyframe or velocity priors.
// Vision anchors the far end; biases start at zero.
//
// Initialization: GT perturbed by --pos/rot/vel-perturb (states) and
// 0.3 m (landmarks); biases zero. The exported "init" columns are this
// starting point.
//
// keyframes_csv:  timestamp_s, gt_p_*, gt_q_*, init_p_*, init_pos_err_m,
//                 opt_p_*, opt_pos_err_m, opt_rot_err_deg, opt_v_*,
//                 bg_*, ba_*
// landmarks_csv:  gt_x/y/z, init_x/y/z, opt_x/y/z, n_obs, opt_err_m

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "slam_core/camera/pinhole_camera.hpp"
#include "slam_core/factors/imu_factor.hpp"
#include "slam_core/factors/prior_factors.hpp"
#include "slam_core/factors/reprojection_factor.hpp"
#include "slam_core/geometry/so3.hpp"
#include "slam_core/imu/preintegration.hpp"
#include "slam_core/io/euroc_camera.hpp"
#include "slam_core/io/euroc_csv.hpp"
#include "slam_core/optim/levenberg_marquardt.hpp"
#include "slam_core/optim/problem.hpp"

namespace {

constexpr double kPi = 3.14159265358979323846;

struct Options {
    double        start_s     = 5.0;
    double        duration_s  = 20.0;
    double        spacing_s   = 0.5;
    int           n_landmarks = 150;
    double        pixel_sigma = 1.0;
    std::uint32_t seed        = 42;
    double        pos_perturb = 0.3;   // m
    double        rot_perturb = 0.05;  // rad
    double        vel_perturb = 0.2;   // m/s
};

class Rng {
public:
    explicit Rng(std::uint32_t seed) : rng_(seed) {}
    double uniform() { return 2.0 * (static_cast<double>(rng_()) / 4294967295.0) - 1.0; }
    double uniform01() { return static_cast<double>(rng_()) / 4294967295.0; }
    double gaussian() {
        const double u1 = (static_cast<double>(rng_()) + 1.0) / 4294967297.0;
        const double u2 = (static_cast<double>(rng_()) + 1.0) / 4294967297.0;
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * kPi * u2);
    }
    Eigen::Vector3d uniform_vec3() { return {uniform(), uniform(), uniform()}; }
    Eigen::Vector3d gaussian_vec3() { return {gaussian(), gaussian(), gaussian()}; }

private:
    std::mt19937 rng_;
};

bool parse_value(const char* s, double& out) {
    try {
        out = std::stod(s);
    } catch (...) {
        return false;
    }
    return std::isfinite(out);
}

}  // namespace

int main(int argc, char* argv[]) {
    const char* usage =
        "Usage: optimize_vi_chain <euroc_imu_csv> <keyframes_csv> <landmarks_csv>"
        " [--start-s S] [--duration-s D] [--keyframe-spacing-s K] [--n-landmarks N]"
        " [--pixel-sigma P] [--seed SEED] [--pos-perturb M] [--rot-perturb RAD]"
        " [--vel-perturb MPS]\n";
    if (argc < 4) {
        std::cerr << usage;
        return EXIT_FAILURE;
    }

    Options opt;
    for (int i = 4; i < argc; ++i) {
        const std::string arg(argv[i]);
        double            value = 0.0;
        if (i + 1 >= argc || !parse_value(argv[i + 1], value)) {
            std::cerr << "Error: " << arg << " requires a finite numeric value\n";
            return EXIT_FAILURE;
        }
        ++i;
        if (arg == "--start-s") opt.start_s = value;
        else if (arg == "--duration-s") opt.duration_s = value;
        else if (arg == "--keyframe-spacing-s") opt.spacing_s = value;
        else if (arg == "--n-landmarks") opt.n_landmarks = static_cast<int>(value);
        else if (arg == "--pixel-sigma") opt.pixel_sigma = value;
        else if (arg == "--seed") opt.seed = static_cast<std::uint32_t>(value);
        else if (arg == "--pos-perturb") opt.pos_perturb = value;
        else if (arg == "--rot-perturb") opt.rot_perturb = value;
        else if (arg == "--vel-perturb") opt.vel_perturb = value;
        else {
            std::cerr << "Error: unknown argument: " << arg << '\n' << usage;
            return EXIT_FAILURE;
        }
    }

    // --- load IMU, GT, camera calibration ---
    namespace fs = std::filesystem;
    std::vector<slam_core::imu::ImuMeasurement> imu;
    std::vector<slam_core::io::EurocGtSample>   gt;
    slam_core::io::EurocCameraCalib             calib;
    try {
        imu                     = slam_core::io::read_euroc_imu_csv(argv[1]);
        const fs::path mav_root = fs::path(argv[1]).parent_path().parent_path();
        gt =
            slam_core::io::read_euroc_gt_csv(mav_root / "state_groundtruth_estimate0" / "data.csv");
        calib = slam_core::io::read_euroc_camera_yaml(mav_root / "cam0" / "sensor.yaml");
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
    const slam_core::camera::PinholeCamera cam(calib.fx, calib.fy, calib.cx, calib.cy);

    // --- keyframes at IMU sample timestamps ---
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

    // --- synthesize landmarks and observations through GT + real calib ---
    Rng          rng(opt.seed);
    const double margin = 20.0;  // px, keep observations away from the border

    struct Observation {
        int             keyframe;
        Eigen::Vector2d pixel;
    };
    std::vector<Eigen::Vector3d>          L_gt;
    std::vector<std::vector<Observation>> L_obs;

    int attempts = 0;
    while (static_cast<int>(L_gt.size()) < opt.n_landmarks && attempts < 50 * opt.n_landmarks) {
        ++attempts;
        // Unproject a random in-image pixel of a random keyframe at random depth.
        const int             k = static_cast<int>(rng.uniform01() * (n_kf - 1));
        const Eigen::Vector2d px(margin + rng.uniform01() * (calib.width - 2.0 * margin),
                                 margin + rng.uniform01() * (calib.height - 2.0 * margin));
        const double          depth     = 2.0 + 6.0 * rng.uniform01();
        const Eigen::Vector3d bearing_C = cam.unproject_to_bearing(px);
        const Eigen::Vector3d p_C       = bearing_C * (depth / bearing_C.z());
        const Eigen::Vector3d p_B       = calib.R_B_C * p_C + calib.t_B_C;
        const Eigen::Vector3d p_W       = gt_R[k] * p_B + gt_p[k];

        std::vector<Observation> obs;
        for (int j = 0; j < n_kf; ++j) {
            const Eigen::Vector3d q_B = gt_R[j].transpose() * (p_W - gt_p[j]);
            const Eigen::Vector3d q_C = calib.R_B_C.transpose() * (q_B - calib.t_B_C);
            Eigen::Vector2d       pixel;
            if (!cam.try_project(q_C, pixel, 0.3)) continue;
            if (pixel.x() < margin || pixel.x() > calib.width - margin || pixel.y() < margin ||
                pixel.y() > calib.height - margin)
                continue;
            obs.push_back(
                {j, pixel + opt.pixel_sigma * Eigen::Vector2d(rng.gaussian(), rng.gaussian())});
        }
        if (obs.size() < 3) continue;
        // Parallax gate: depth is unobservable without baseline (the MAV
        // hovers early in MH_01); require the bearing to the landmark to
        // swing by a minimum angle across its observations.
        double max_parallax = 0.0;
        for (std::size_t a = 0; a < obs.size(); ++a) {
            for (std::size_t b = a + 1; b < obs.size(); ++b) {
                const Eigen::Vector3d u_a   = (p_W - gt_p[obs[a].keyframe]).normalized();
                const Eigen::Vector3d u_b   = (p_W - gt_p[obs[b].keyframe]).normalized();
                const double          angle = std::acos(std::clamp(u_a.dot(u_b), -1.0, 1.0));
                max_parallax                = std::max(max_parallax, angle);
            }
        }
        if (max_parallax < 1.5 * kPi / 180.0) continue;
        L_gt.push_back(p_W);
        L_obs.push_back(std::move(obs));
    }
    if (static_cast<int>(L_gt.size()) < opt.n_landmarks) {
        std::cerr << "Warning: only " << L_gt.size() << " of " << opt.n_landmarks
                  << " landmarks have >= 3 observations\n";
    }
    const int n_lm = static_cast<int>(L_gt.size());
    if (n_lm < 10) {
        std::cerr << "Error: too few landmarks; widen the window\n";
        return EXIT_FAILURE;
    }

    // --- preintegrate intervals ---
    const Eigen::Vector3d                         gravity_W(0.0, 0.0, -9.81);
    const Eigen::Vector3d                         zero = Eigen::Vector3d::Zero();
    const slam_core::imu::ImuNoiseParams          noise{1.6968e-4, 2.0e-3};
    std::vector<slam_core::imu::PreintegratedImu> preints;
    for (int k = 0; k + 1 < n_kf; ++k) {
        slam_core::imu::PreintegratedImu p;
        for (std::size_t i = kf_idx[k]; i < kf_idx[k + 1]; ++i) {
            slam_core::imu::integrate(p, imu[i], zero, zero,
                                      imu[i + 1].timestamp_s - imu[i].timestamp_s, noise);
        }
        preints.push_back(p);
    }

    // --- variables: perturbed states (first pose exact), zero biases ---
    std::vector<slam_core::optim::So3Variable>    R;
    std::vector<slam_core::optim::VectorVariable> v, p, bg, ba, L;
    std::vector<Eigen::Vector3d>                  init_p(n_kf);
    for (int k = 0; k < n_kf; ++k) {
        const bool            first = (k == 0);
        const Eigen::Vector3d dR =
            first ? zero : Eigen::Vector3d(opt.rot_perturb * rng.uniform_vec3());
        const Eigen::Vector3d dp =
            first ? zero : Eigen::Vector3d(opt.pos_perturb * rng.uniform_vec3());
        const Eigen::Vector3d dv =
            first ? zero : Eigen::Vector3d(opt.vel_perturb * rng.uniform_vec3());
        R.emplace_back(gt_R[k] * slam_core::geometry::exp_so3(dR));
        p.emplace_back(gt_p[k] + dp);
        v.emplace_back(gt_v[k] + dv);
        bg.emplace_back(zero);
        ba.emplace_back(zero);
        init_p[k] = gt_p[k] + dp;
    }
    std::vector<Eigen::Vector3d> init_L(n_lm);
    for (int i = 0; i < n_lm; ++i) {
        init_L[i] = L_gt[i] + 0.3 * rng.uniform_vec3();
        L.emplace_back(init_L[i]);
    }

    // --- factors ---
    std::vector<slam_core::factors::ImuFactor>            imu_factors;
    std::vector<slam_core::factors::BiasRandomWalkFactor> walk_factors;
    for (int k = 0; k + 1 < n_kf; ++k) {
        imu_factors.emplace_back(&R[k], &v[k], &p[k], &R[k + 1], &v[k + 1], &p[k + 1], &bg[k],
                                 &ba[k], preints[k], gravity_W, zero, zero);
        walk_factors.emplace_back(&bg[k], &ba[k], &bg[k + 1], &ba[k + 1], 1.9393e-5, 3.0e-3,
                                  preints[k].delta_t_s);
    }
    std::vector<slam_core::factors::ReprojectionFactor> reproj_factors;
    for (int i = 0; i < n_lm; ++i) {
        for (const Observation& o : L_obs[i]) {
            reproj_factors.emplace_back(&R[o.keyframe], &p[o.keyframe], &L[i], cam, calib.R_B_C,
                                        calib.t_B_C, o.pixel, opt.pixel_sigma);
        }
    }
    slam_core::factors::So3PriorFactor    prior_R0(&R[0], gt_R[0], 1e-3);
    slam_core::factors::VectorPriorFactor prior_p0(&p[0], gt_p[0], 1e-3);

    slam_core::optim::Problem problem;
    for (int k = 0; k < n_kf; ++k) {
        problem.add_variable(&R[k]);
        problem.add_variable(&v[k]);
        problem.add_variable(&p[k]);
        problem.add_variable(&bg[k]);
        problem.add_variable(&ba[k]);
    }
    for (auto& l : L) problem.add_variable(&l);
    for (auto& f : imu_factors) problem.add_factor(&f);
    for (auto& f : walk_factors) problem.add_factor(&f);
    for (auto& f : reproj_factors) problem.add_factor(&f);
    problem.add_factor(&prior_R0);
    problem.add_factor(&prior_p0);

    slam_core::optim::LmOptions lm;
    lm.max_iterations                       = 100;
    const slam_core::optim::LmResult result = slam_core::optim::optimize(problem, lm);

    // --- outputs ---
    for (const char* path : {argv[2], argv[3]}) {
        const fs::path out(path);
        if (out.has_parent_path()) fs::create_directories(out.parent_path());
    }
    std::ofstream kf_out(argv[2]);
    std::ofstream lm_out(argv[3]);
    if (!kf_out.is_open() || !lm_out.is_open()) {
        std::cerr << "Error: cannot open output files\n";
        return EXIT_FAILURE;
    }
    kf_out << std::setprecision(12) << std::fixed;
    lm_out << std::setprecision(12) << std::fixed;

    kf_out << "timestamp_s,gt_p_x,gt_p_y,gt_p_z,gt_q_w,gt_q_x,gt_q_y,gt_q_z"
           << ",init_p_x,init_p_y,init_p_z,init_pos_err_m"
           << ",opt_p_x,opt_p_y,opt_p_z,opt_pos_err_m,opt_rot_err_deg"
           << ",opt_v_x,opt_v_y,opt_v_z,bg_x,bg_y,bg_z,ba_x,ba_y,ba_z\n";
    double init_rmse = 0.0, opt_rmse = 0.0, opt_max = 0.0;
    for (int k = 0; k < n_kf; ++k) {
        const double init_err = (init_p[k] - gt_p[k]).norm();
        const double opt_err  = (p[k].vec() - gt_p[k]).norm();
        const double rot_err =
            slam_core::geometry::log_so3(gt_R[k].transpose() * R[k].R()).norm() * 180.0 / kPi;
        init_rmse += init_err * init_err;
        opt_rmse += opt_err * opt_err;
        opt_max = std::max(opt_max, opt_err);

        kf_out << imu[kf_idx[k]].timestamp_s;
        kf_out << ',' << gt_p[k].x() << ',' << gt_p[k].y() << ',' << gt_p[k].z();
        kf_out << ',' << gt_q[k].w() << ',' << gt_q[k].x() << ',' << gt_q[k].y() << ','
               << gt_q[k].z();
        kf_out << ',' << init_p[k].x() << ',' << init_p[k].y() << ',' << init_p[k].z() << ','
               << init_err;
        kf_out << ',' << p[k].vec().x() << ',' << p[k].vec().y() << ',' << p[k].vec().z();
        kf_out << ',' << opt_err << ',' << rot_err;
        kf_out << ',' << v[k].vec().x() << ',' << v[k].vec().y() << ',' << v[k].vec().z();
        kf_out << ',' << bg[k].vec().x() << ',' << bg[k].vec().y() << ',' << bg[k].vec().z();
        kf_out << ',' << ba[k].vec().x() << ',' << ba[k].vec().y() << ',' << ba[k].vec().z();
        kf_out << '\n';
    }
    init_rmse = std::sqrt(init_rmse / n_kf);
    opt_rmse  = std::sqrt(opt_rmse / n_kf);

    lm_out << "gt_x,gt_y,gt_z,init_x,init_y,init_z,opt_x,opt_y,opt_z,n_obs,opt_err_m\n";
    double lm_rmse = 0.0;
    for (int i = 0; i < n_lm; ++i) {
        const double err = (L[i].vec() - L_gt[i]).norm();
        lm_rmse += err * err;
        lm_out << L_gt[i].x() << ',' << L_gt[i].y() << ',' << L_gt[i].z();
        lm_out << ',' << init_L[i].x() << ',' << init_L[i].y() << ',' << init_L[i].z();
        lm_out << ',' << L[i].vec().x() << ',' << L[i].vec().y() << ',' << L[i].vec().z();
        lm_out << ',' << L_obs[i].size() << ',' << err << '\n';
    }
    lm_rmse = std::sqrt(lm_rmse / n_lm);

    Eigen::Vector3d bg_mean = zero, ba_mean = zero;
    for (int k = 0; k < n_kf; ++k) {
        bg_mean += bg[k].vec();
        ba_mean += ba[k].vec();
    }
    bg_mean /= n_kf;
    ba_mean /= n_kf;

    std::cerr << std::setprecision(6) << std::fixed;
    std::cerr << "optimize_vi_chain:\n"
              << "  keyframes / landmarks: " << n_kf << " / " << n_lm << " ("
              << reproj_factors.size() << " reprojection factors, pixel sigma " << opt.pixel_sigma
              << ")\n"
              << "  LM:                   " << (result.converged ? "converged" : "NOT converged")
              << " in " << result.iterations << " iterations (" << result.message << ")\n"
              << "  cost:                 " << result.initial_cost << " -> " << result.final_cost
              << '\n'
              << "  keyframe pos err:     init rmse " << init_rmse << " m -> opt rmse " << opt_rmse
              << " m (max " << opt_max << " m)\n"
              << "  landmark rmse:        " << lm_rmse << " m\n"
              << "  mean gyro bias:       [" << bg_mean.x() << ", " << bg_mean.y() << ", "
              << bg_mean.z() << "] rad/s\n"
              << "  mean accel bias:      [" << ba_mean.x() << ", " << ba_mean.y() << ", "
              << ba_mean.z() << "] m/s^2\n"
              << "  outputs:              " << argv[2] << ", " << argv[3] << '\n';

    return result.converged ? EXIT_SUCCESS : EXIT_FAILURE;
}
