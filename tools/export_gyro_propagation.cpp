// Export gyro-only SO(3) orientation propagation results to CSV.
//
// Usage: export_gyro_propagation <euroc_imu_csv_path> <output_csv_path>
//
// Input:  EuRoC IMU CSV — columns: timestamp_ns, wx, wy, wz, ax, ay, az
//                         Gyro is columns 1,2,3 (wx, wy, wz) immediately after timestamp.
//
// Output: CSV — timestamp_s, r00..r22 (row-major R_W_B entries)
//               R_W_B maps body frame B into world frame W.
//               Initialized to identity at t0.
//               Propagation: R_W_B_next = R_W_B * exp_so3(gyro_radps * dt_s)
//               Zeroth-order hold: gyro at t_i is applied over [t_i, t_{i+1}].

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include <Eigen/Core>

#include "slam_core/imu/gyro_propagation.hpp"

namespace {

struct ImuRow {
    double          timestamp_s;  // seconds
    Eigen::Vector3d gyro_radps;   // body frame, rad/s (wx, wy, wz — columns 1,2,3)
};

// EuRoC IMU CSV: timestamp_ns, wx, wy, wz, ax, ay, az
// Parse only the first 4 columns; accel is not used.
bool parse_imu_row(const std::string& line, ImuRow& out) {
    std::istringstream ss(line);
    std::string tok;
    double vals[4];
    for (int i = 0; i < 4; ++i) {
        if (!std::getline(ss, tok, ',')) return false;
        try {
            vals[i] = std::stod(tok);
        } catch (...) {
            return false;
        }
    }
    // col 0: timestamp_ns -> seconds
    out.timestamp_s = vals[0] * 1e-9;
    // cols 1,2,3: wx, wy, wz
    out.gyro_radps = Eigen::Vector3d{vals[1], vals[2], vals[3]};
    return true;
}

void write_row(std::ofstream& out, double timestamp_s,
               const Eigen::Matrix3d& R_W_B) {
    out << timestamp_s;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            out << ',' << R_W_B(r, c);
    out << '\n';
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <euroc_imu_csv_path> <output_csv_path>\n";
        return EXIT_FAILURE;
    }

    std::ifstream in_file(argv[1]);
    if (!in_file.is_open()) {
        std::cerr << "Error: cannot open input: " << argv[1] << '\n';
        return EXIT_FAILURE;
    }

    std::ofstream out_file(argv[2]);
    if (!out_file.is_open()) {
        std::cerr << "Error: cannot open output: " << argv[2] << '\n';
        return EXIT_FAILURE;
    }
    out_file << std::setprecision(12) << std::fixed;
    out_file << "timestamp_s,r00,r01,r02,r10,r11,r12,r20,r21,r22\n";

    // Skip EuRoC CSV header line
    std::string line;
    if (!std::getline(in_file, line)) {
        std::cerr << "Error: input CSV is empty\n";
        return EXIT_FAILURE;
    }

    // R_W_B: rotation mapping body frame B into world frame W.
    // Initialized to identity — no gravity alignment, no bias correction.
    Eigen::Matrix3d R_W_B = Eigen::Matrix3d::Identity();

    ImuRow prev{}, curr{};
    bool have_prev = false;
    int written = 0, skipped = 0;

    while (std::getline(in_file, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (!parse_imu_row(line, curr)) {
            std::cerr << "Warning: unparseable row skipped\n";
            ++skipped;
            continue;
        }

        if (!have_prev) {
            // Write identity orientation at the first timestamp (before integration).
            write_row(out_file, curr.timestamp_s, R_W_B);
            ++written;
            prev = curr;
            have_prev = true;
            continue;
        }

        const double dt_s = curr.timestamp_s - prev.timestamp_s;
        if (dt_s <= 0.0 || !std::isfinite(dt_s)) {
            std::cerr << "Warning: non-positive dt_s=" << dt_s
                      << " at t=" << curr.timestamp_s << ", skipping interval\n";
            ++skipped;
            prev = curr;
            continue;
        }

        // Zeroth-order hold: apply prev.gyro over [prev.timestamp_s, curr.timestamp_s].
        R_W_B = slam_core::imu::propagate_gyro(R_W_B, prev.gyro_radps, dt_s);
        write_row(out_file, curr.timestamp_s, R_W_B);
        ++written;
        prev = curr;
    }

    std::cerr << "export_gyro_propagation: wrote " << written << " rows";
    if (skipped > 0) std::cerr << " (" << skipped << " intervals skipped)";
    std::cerr << " -> " << argv[2] << '\n';
    return EXIT_SUCCESS;
}
