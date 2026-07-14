// Evaluate approximate accelerometer bias from EuRoC ground truth.
//
// Usage:
//   evaluate_accel_bias_from_gt <imu_csv> <gt_csv> <output_csv>
//                               [--window-duration <sec>] [--gravity-z <value>]
//
// Convention:
//   GT quaternion: q_W_B (EuRoC Hamilton, w-first).
//   R_W_B = Eigen::Quaterniond(q_w, q_x, q_y, q_z).toRotationMatrix()
//   GT velocity: v_W_B (columns 8-10 in EuRoC GT CSV, world frame, m/s).
//
//   For each non-overlapping window [t0, t1]:
//     dv_GT  = v_gt_nearest_t1 - v_gt_nearest_t0      (world frame)
//     dv_raw = sum_i( R_W_B_gt_i * accel_meas_i * dt_i )  (no bias, world frame)
//     M      = sum_i( R_W_B_gt_i * dt_i )              (3x3)
//     dv_err = (dv_raw + gravity_W * integrated_dt) - dv_GT
//     accel_bias_B = solve(M, dv_err)
//
//   GT orientation is looked up at each IMU timestamp (no gyro propagation).
//   Sign convention: accel_meas = accel_true + bias.
//
// Output CSV columns:
//   timestamp_start_s, timestamp_end_s, integrated_dt_s,
//   accel_bias_est_x_mps2, accel_bias_est_y_mps2, accel_bias_est_z_mps2,
//   accel_bias_est_norm_mps2, velocity_error_norm_mps

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/QR>

#include "slam_core/imu/imu_measurement.hpp"
#include "slam_core/io/euroc_csv.hpp"

int main(int argc, char* argv[]) {
    const char* usage =
        "Usage: evaluate_accel_bias_from_gt"
        " <imu_csv> <gt_csv> <output_csv>"
        " [--window-duration <sec>] [--gravity-z <value>]\n";

    if (argc < 4) { std::cerr << usage; return EXIT_FAILURE; }

    double window_duration = 1.0;
    double gravity_z       = -9.81;

    for (int i = 4; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--window-duration") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --window-duration requires a value\n";
                return EXIT_FAILURE;
            }
            try { window_duration = std::stod(argv[++i]); }
            catch (...) {
                std::cerr << "Error: invalid --window-duration value\n";
                return EXIT_FAILURE;
            }
            if (window_duration <= 0.0) {
                std::cerr << "Error: --window-duration must be positive\n";
                return EXIT_FAILURE;
            }
        } else if (arg == "--gravity-z") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --gravity-z requires a value\n";
                return EXIT_FAILURE;
            }
            try { gravity_z = std::stod(argv[++i]); }
            catch (...) {
                std::cerr << "Error: invalid --gravity-z value\n";
                return EXIT_FAILURE;
            }
        } else {
            std::cerr << "Error: unknown argument: " << arg << '\n' << usage;
            return EXIT_FAILURE;
        }
    }

    const Eigen::Vector3d gravity_W{0.0, 0.0, gravity_z};

    // Load IMU and GT samples into memory.
    std::vector<slam_core::imu::ImuMeasurement>   imu_data;
    std::vector<slam_core::io::EurocGtSample>     gt_data;
    int imu_skipped = 0, gt_skipped = 0;
    try {
        imu_data = slam_core::io::read_euroc_imu_csv(argv[1], &imu_skipped);
        gt_data  = slam_core::io::read_euroc_gt_csv(argv[2], &gt_skipped);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
    if (imu_skipped > 0)
        std::cerr << "Warning: " << imu_skipped << " IMU rows skipped (unparseable)\n";
    if (gt_skipped > 0)
        std::cerr << "Warning: " << gt_skipped << " GT rows skipped (unparseable)\n";
    if (imu_data.size() < 2) {
        std::cerr << "Error: too few IMU samples\n";
        return EXIT_FAILURE;
    }
    const double gt_start = gt_data.front().timestamp_s;
    const double gt_end   = gt_data.back().timestamp_s;

    // Open output.
    std::ofstream out_file(argv[3]);
    if (!out_file.is_open()) {
        std::cerr << "Error: cannot open output: " << argv[3] << '\n';
        return EXIT_FAILURE;
    }
    out_file << std::setprecision(12) << std::fixed;
    out_file << "timestamp_start_s,timestamp_end_s,integrated_dt_s"
             << ",accel_bias_est_x_mps2,accel_bias_est_y_mps2,accel_bias_est_z_mps2"
             << ",accel_bias_est_norm_mps2,velocity_error_norm_mps\n";

    // Advance to first IMU sample inside GT coverage.
    std::size_t idx_base = 0;
    while (idx_base < imu_data.size() &&
           imu_data[idx_base].timestamp_s < gt_start) ++idx_base;
    if (idx_base >= imu_data.size()) {
        std::cerr << "Error: no IMU samples within GT coverage ["
                  << gt_start << ", " << gt_end << "] s\n";
        return EXIT_FAILURE;
    }

    // Summary accumulators.
    std::vector<double> bias_norms;
    std::vector<double> vel_error_norms;
    Eigen::Vector3d bias_sum          = Eigen::Vector3d::Zero();
    double          integrated_dt_sum = 0.0;
    int n_windows = 0;

    double window_start_t = imu_data[idx_base].timestamp_s;

    while (window_start_t + window_duration <= gt_end) {
        const double window_end_t = window_start_t + window_duration;

        // Find i0: first IMU index >= window_start_t.
        while (idx_base < imu_data.size() &&
               imu_data[idx_base].timestamp_s < window_start_t) ++idx_base;
        const std::size_t i0 = idx_base;

        // Find i1: last IMU index with timestamp <= window_end_t.
        std::size_t i1 = i0;
        while (i1 + 1 < imu_data.size() &&
               imu_data[i1 + 1].timestamp_s <= window_end_t) ++i1;

        if (i1 <= i0) { window_start_t += window_duration; continue; }

        const double t0            = imu_data[i0].timestamp_s;
        const double t1            = imu_data[i1].timestamp_s;
        const double integrated_dt = t1 - t0;
        if (integrated_dt <= 0.0) { window_start_t += window_duration; continue; }

        // GT velocity change at window boundaries.
        const Eigen::Vector3d dv_gt =
            slam_core::io::nearest_gt(t1, gt_data).v_W_B -
            slam_core::io::nearest_gt(t0, gt_data).v_W_B;

        // Accumulate raw velocity delta and rotation-weighted time matrix.
        // GT orientation is looked up at each IMU timestamp — no gyro propagation.
        Eigen::Vector3d dv_raw = Eigen::Vector3d::Zero();
        Eigen::Matrix3d M      = Eigen::Matrix3d::Zero();

        for (std::size_t i = i0; i < i1; ++i) {
            const double dt_i =
                imu_data[i + 1].timestamp_s - imu_data[i].timestamp_s;
            const Eigen::Matrix3d R_W_B_i =
                slam_core::io::nearest_gt(imu_data[i].timestamp_s, gt_data).R_W_B();
            dv_raw += R_W_B_i * imu_data[i].accel_mps2 * dt_i;
            M      += R_W_B_i * dt_i;
        }

        // dv_err = (dv_raw + gravity_W * integrated_dt) - dv_GT
        // M * accel_bias_B = dv_err  (accel_meas = accel_true + bias)
        const Eigen::Vector3d dv_err =
            (dv_raw + gravity_W * integrated_dt) - dv_gt;
        const Eigen::Vector3d accel_bias_B =
            M.colPivHouseholderQr().solve(dv_err);

        out_file << t0 << ',' << t1 << ',' << integrated_dt
                 << ',' << accel_bias_B.x()
                 << ',' << accel_bias_B.y()
                 << ',' << accel_bias_B.z()
                 << ',' << accel_bias_B.norm()
                 << ',' << dv_err.norm() << '\n';

        bias_sum          += accel_bias_B;
        integrated_dt_sum += integrated_dt;
        bias_norms.push_back(accel_bias_B.norm());
        vel_error_norms.push_back(dv_err.norm());

        idx_base = i1;
        window_start_t += window_duration;
        ++n_windows;
    }

    if (n_windows == 0) {
        std::cerr << "Warning: no windows produced"
                     " (GT coverage too short for requested window duration?)\n";
        return EXIT_SUCCESS;
    }

    const Eigen::Vector3d mean_bias = bias_sum / n_windows;

    std::vector<double> sorted_norms = bias_norms;
    std::sort(sorted_norms.begin(), sorted_norms.end());
    const double median_norm =
        sorted_norms[sorted_norms.size() / 2];
    const double mean_norm =
        std::accumulate(bias_norms.begin(), bias_norms.end(), 0.0) / n_windows;
    const double mean_vel_err =
        std::accumulate(vel_error_norms.begin(), vel_error_norms.end(), 0.0) / n_windows;
    const double max_vel_err =
        *std::max_element(vel_error_norms.begin(), vel_error_norms.end());
    const double mean_integrated_dt = integrated_dt_sum / n_windows;

    std::cerr << std::setprecision(6) << std::fixed;
    std::cerr << "evaluate_accel_bias_from_gt:\n"
              << "  windows:              " << n_windows << "\n"
              << "  nominal window dur:   " << window_duration    << " s\n"
              << "  mean integrated dt:   " << mean_integrated_dt << " s\n"
              << "  gravity_W:            [0, 0, " << gravity_z   << "] m/s²\n"
              << "  mean bias est:        ["
                  << mean_bias.x() << ", "
                  << mean_bias.y() << ", "
                  << mean_bias.z() << "] m/s²\n"
              << "  mean  bias norm:      " << mean_norm   << " m/s²\n"
              << "  median bias norm:     " << median_norm << " m/s²\n"
              << "  mean  vel error:      " << mean_vel_err << " m/s\n"
              << "  max   vel error:      " << max_vel_err  << " m/s\n"
              << "  output:               " << argv[3] << '\n';

    return EXIT_SUCCESS;
}
