// Export full IMU state propagation results to CSV.
//
// Usage: export_imu_state_propagation <euroc_imu_csv_path> <output_csv_path>
//
// Input:  EuRoC IMU CSV — columns: timestamp_ns, wx, wy, wz, ax, ay, az
//
// Output: CSV — timestamp_s, p_x/y/z, v_x/y/z, q_w/x/y/z,
//               r00..r22 (row-major R_W_B), gyro_bias_x/y/z, accel_bias_x/y/z
//
// Propagation:
//   state[0] is initialized at t[0]: identity R, zero p/v/biases.
//   state[i+1] is produced by propagate_imu_state(state[i], meas[i], gravity_W, dt)
//   where meas[i] is the EuRoC measurement at t[i] and dt = t[i+1] - t[i].
//   Zeroth-order hold: meas[i] is applied constant over [t[i], t[i+1]].
//   Written timestamp matches state[i+1].timestamp_s = t[i+1].
//   gravity_W = [0, 0, -9.81] m/s^2. Biases are zero throughout.

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "slam_core/imu/imu_measurement.hpp"
#include "slam_core/imu/imu_state.hpp"

namespace {

// EuRoC IMU CSV: timestamp_ns, wx, wy, wz, ax, ay, az
bool parse_imu_row(const std::string& line,
                   slam_core::imu::ImuMeasurement& out) {
    std::istringstream ss(line);
    std::string tok;
    double vals[7];
    for (int i = 0; i < 7; ++i) {
        if (!std::getline(ss, tok, ',')) return false;
        try {
            vals[i] = std::stod(tok);
        } catch (...) {
            return false;
        }
    }
    out.timestamp_s = vals[0] * 1e-9;                          // ns -> s
    out.gyro_radps  = {vals[1], vals[2], vals[3]};             // wx, wy, wz
    out.accel_mps2  = {vals[4], vals[5], vals[6]};             // ax, ay, az
    return true;
}

void write_header(std::ofstream& out) {
    out << "timestamp_s"
        << ",p_x,p_y,p_z"
        << ",v_x,v_y,v_z"
        << ",q_w,q_x,q_y,q_z"
        << ",r00,r01,r02,r10,r11,r12,r20,r21,r22"
        << ",gyro_bias_x,gyro_bias_y,gyro_bias_z"
        << ",accel_bias_x,accel_bias_y,accel_bias_z"
        << '\n';
}

void write_row(std::ofstream& out,
               const slam_core::imu::ImuState& s) {
    const Eigen::Quaterniond q(s.R_W_B);

    out << s.timestamp_s;
    out << ',' << s.p_W_B.x()           << ',' << s.p_W_B.y()           << ',' << s.p_W_B.z();
    out << ',' << s.v_W_B.x()           << ',' << s.v_W_B.y()           << ',' << s.v_W_B.z();
    out << ',' << q.w()                 << ',' << q.x()                 << ',' << q.y()   << ',' << q.z();
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            out << ',' << s.R_W_B(r, c);
    out << ',' << s.gyro_bias_radps.x() << ',' << s.gyro_bias_radps.y() << ',' << s.gyro_bias_radps.z();
    out << ',' << s.accel_bias_mps2.x() << ',' << s.accel_bias_mps2.y() << ',' << s.accel_bias_mps2.z();
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
    write_header(out_file);

    // Skip EuRoC CSV header line
    std::string line;
    if (!std::getline(in_file, line)) {
        std::cerr << "Error: input CSV is empty\n";
        return EXIT_FAILURE;
    }

    const Eigen::Vector3d gravity_W{0.0, 0.0, -9.81};

    slam_core::imu::ImuState state{};
    slam_core::imu::ImuMeasurement prev_meas{}, curr_meas{};
    bool have_prev = false;
    int written = 0, skipped = 0;

    while (std::getline(in_file, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (!parse_imu_row(line, curr_meas)) {
            std::cerr << "Warning: unparseable row skipped\n";
            ++skipped;
            continue;
        }

        if (!have_prev) {
            // state[0]: initialized at t[0] with identity R, zero p/v/biases.
            state.timestamp_s     = curr_meas.timestamp_s;
            state.R_W_B           = Eigen::Matrix3d::Identity();
            state.p_W_B           = Eigen::Vector3d::Zero();
            state.v_W_B           = Eigen::Vector3d::Zero();
            state.gyro_bias_radps = Eigen::Vector3d::Zero();
            state.accel_bias_mps2 = Eigen::Vector3d::Zero();
            write_row(out_file, state);
            ++written;
            prev_meas = curr_meas;
            have_prev = true;
            continue;
        }

        const double dt_s = curr_meas.timestamp_s - prev_meas.timestamp_s;
        if (dt_s <= 0.0 || !std::isfinite(dt_s)) {
            std::cerr << "Warning: non-positive dt_s=" << dt_s
                      << " at t=" << curr_meas.timestamp_s << ", skipping interval\n";
            ++skipped;
            prev_meas = curr_meas;
            continue;
        }

        // state[i+1] = propagate(state[i], meas[i], gravity_W, dt)
        // meas[i] is prev_meas; dt = t[i+1] - t[i].
        // Written timestamp: state[i+1].timestamp_s = t[i] + dt = t[i+1].
        state = slam_core::imu::propagate_imu_state(
            state, prev_meas, gravity_W, dt_s);
        write_row(out_file, state);
        ++written;
        prev_meas = curr_meas;
    }

    std::cerr << "export_imu_state_propagation: wrote " << written << " rows";
    if (skipped > 0) std::cerr << " (" << skipped << " intervals skipped)";
    std::cerr << " -> " << argv[2] << '\n';
    return EXIT_SUCCESS;
}
