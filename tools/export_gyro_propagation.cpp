// Export gyro-only SO(3) orientation propagation results to CSV.
//
// Usage:
//   export_gyro_propagation <euroc_imu_csv_path> <output_csv_path>
//   export_gyro_propagation <euroc_imu_csv_path> <output_csv_path> --gyro-bias bx by bz
//
// Input:  EuRoC IMU CSV — columns: timestamp_ns, wx, wy, wz, ax, ay, az
//
// Output: CSV — timestamp_s, r00..r22 (row-major R_W_B entries)
//               R_W_B maps body frame B into world frame W.
//               Initialized to identity at t0.
//               Default: R_W_B_next = R_W_B * exp_so3(gyro_radps * dt_s)
//               With --gyro-bias: bias is subtracted before integration.
//               Zeroth-order hold: gyro at t_i is applied over [t_i, t_{i+1}].
//
// --gyro-bias bx by bz:
//   Offline validation only. User-supplied constant body-frame gyro bias [rad/s].
//   Convention: omega_meas = omega_true + bias; correction = omega_meas - bias.
//   This does not represent a runtime estimator or GT-derived capability.

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "slam_core/imu/gyro_propagation.hpp"
#include "slam_core/imu/imu_measurement.hpp"
#include "slam_core/io/euroc_csv.hpp"

namespace {

void write_row(std::ofstream& out, double timestamp_s, const Eigen::Matrix3d& R_W_B) {
    out << timestamp_s;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) out << ',' << R_W_B(r, c);
    out << '\n';
}

}  // namespace

int main(int argc, char* argv[]) {
    const char* usage =
        "Usage: export_gyro_propagation"
        " <euroc_imu_csv_path> <output_csv_path>"
        " [--gyro-bias bx by bz]\n";

    if (argc != 3 && argc != 7) {
        std::cerr << usage;
        return EXIT_FAILURE;
    }

    bool            use_bias_correction = false;
    Eigen::Vector3d gyro_bias           = Eigen::Vector3d::Zero();

    if (argc == 7) {
        if (std::string(argv[3]) != "--gyro-bias") {
            std::cerr << "Error: unknown argument: " << argv[3] << '\n' << usage;
            return EXIT_FAILURE;
        }
        try {
            gyro_bias = {std::stod(argv[4]), std::stod(argv[5]), std::stod(argv[6])};
        } catch (...) {
            std::cerr << "Error: --gyro-bias requires three numeric values\n";
            return EXIT_FAILURE;
        }
        use_bias_correction = true;
    }

    std::vector<slam_core::imu::ImuMeasurement> imu_data;
    int                                         parse_skipped = 0;
    try {
        imu_data = slam_core::io::read_euroc_imu_csv(argv[1], &parse_skipped);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
    if (parse_skipped > 0)
        std::cerr << "Warning: " << parse_skipped << " IMU rows skipped (unparseable)\n";

    std::ofstream out_file(argv[2]);
    if (!out_file.is_open()) {
        std::cerr << "Error: cannot open output: " << argv[2] << '\n';
        return EXIT_FAILURE;
    }
    out_file << std::setprecision(12) << std::fixed;
    out_file << "timestamp_s,r00,r01,r02,r10,r11,r12,r20,r21,r22\n";

    if (use_bias_correction) {
        std::cerr << "export_gyro_propagation: bias correction enabled"
                     " (user-provided constant bias, offline validation only)\n"
                  << "  gyro_bias_radps: [" << gyro_bias.x() << ", " << gyro_bias.y() << ", "
                  << gyro_bias.z() << "]\n"
                  << "  omega_corrected = omega_meas - gyro_bias\n";
    } else {
        std::cerr << "export_gyro_propagation: bias correction disabled (raw gyro)\n";
    }

    // R_W_B: rotation mapping body frame B into world frame W.
    // Initialized to identity — no gravity alignment.
    Eigen::Matrix3d R_W_B = Eigen::Matrix3d::Identity();

    // Write identity orientation at the first timestamp (before integration).
    write_row(out_file, imu_data.front().timestamp_s, R_W_B);
    int written = 1, skipped = 0;

    for (std::size_t i = 0; i + 1 < imu_data.size(); ++i) {
        const slam_core::imu::ImuMeasurement& prev = imu_data[i];
        const slam_core::imu::ImuMeasurement& curr = imu_data[i + 1];

        const double dt_s = curr.timestamp_s - prev.timestamp_s;
        if (dt_s <= 0.0 || !std::isfinite(dt_s)) {
            std::cerr << "Warning: non-positive dt_s=" << dt_s << " at t=" << curr.timestamp_s
                      << ", skipping interval\n";
            ++skipped;
            continue;
        }

        // Zeroth-order hold: apply prev.gyro over [prev.timestamp_s, curr.timestamp_s].
        R_W_B = slam_core::imu::propagate_gyro(R_W_B, prev.gyro_radps, dt_s, gyro_bias);
        write_row(out_file, curr.timestamp_s, R_W_B);
        ++written;
    }

    std::cerr << "export_gyro_propagation: wrote " << written << " rows";
    if (skipped > 0) std::cerr << " (" << skipped << " intervals skipped)";
    std::cerr << " -> " << argv[2] << '\n';
    return EXIT_SUCCESS;
}
