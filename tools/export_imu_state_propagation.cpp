// Export full IMU state propagation results to CSV.
//
// Usage:
//   export_imu_state_propagation <euroc_imu_csv_path> <output_csv_path> [--init-from-gt]
//
// Input:  EuRoC IMU CSV — columns: timestamp_ns, wx, wy, wz, ax, ay, az
//
// Output: CSV — timestamp_s, p_x/y/z, v_x/y/z, q_w/x/y/z,
//               r00..r22 (row-major R_W_B), gyro_bias_x/y/z, accel_bias_x/y/z
//
// Propagation:
//   state[0] is initialized at t[0].
//   Default: identity R, zero p/v/biases.
//   --init-from-gt: R/p/v from nearest GT sample to t[0]; biases zero.
//   state[i+1] is produced by propagate_imu_state(state[i], meas[i], gravity_W, dt)
//   where meas[i] is the EuRoC measurement at t[i] and dt = t[i+1] - t[i].
//   Zeroth-order hold: meas[i] is applied constant over [t[i], t[i+1]].
//   Written timestamp matches state[i+1].timestamp_s = t[i+1].
//   gravity_W = [0, 0, -9.81] m/s^2. Biases are zero throughout.

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

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

// EuRoC GT CSV: timestamp_ns, p_x, p_y, p_z, q_w, q_x, q_y, q_z, v_x, v_y, v_z, ...
struct GtSample {
    double timestamp_s;
    Eigen::Vector3d p;           // position in world [m]
    double q_w, q_x, q_y, q_z;  // quaternion (Hamilton, w scalar first)
    Eigen::Vector3d v;           // velocity in world [m/s]
};

bool parse_gt_row(const std::string& line, GtSample& out) {
    std::istringstream ss(line);
    std::string tok;
    double vals[11];
    for (int i = 0; i < 11; ++i) {
        if (!std::getline(ss, tok, ',')) return false;
        try { vals[i] = std::stod(tok); } catch (...) { return false; }
    }
    out.timestamp_s = vals[0] * 1e-9;
    out.p  = {vals[1], vals[2], vals[3]};
    out.q_w = vals[4]; out.q_x = vals[5]; out.q_y = vals[6]; out.q_z = vals[7];
    out.v  = {vals[8], vals[9], vals[10]};
    return true;
}

std::vector<GtSample> read_gt_csv(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "Error: cannot open GT CSV: " << path << '\n';
        std::exit(EXIT_FAILURE);
    }
    std::string line;
    std::getline(f, line);  // skip header
    std::vector<GtSample> samples;
    int skipped = 0;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        GtSample s;
        if (parse_gt_row(line, s)) samples.push_back(s);
        else ++skipped;
    }
    if (skipped > 0)
        std::cerr << "Warning: " << skipped << " GT rows skipped (unparseable)\n";
    return samples;
}

// Nearest-neighbor by absolute timestamp difference (linear scan; GT ≤ ~36k rows).
// Timestamps are sorted ascending: once we pass ts the distance grows monotonically.
const GtSample& nearest_gt(double ts, const std::vector<GtSample>& samples) {
    std::size_t best = 0;
    double best_dt = std::abs(samples[0].timestamp_s - ts);
    for (std::size_t i = 1; i < samples.size(); ++i) {
        const double dt = std::abs(samples[i].timestamp_s - ts);
        if (dt < best_dt) { best_dt = dt; best = i; }
        if (samples[i].timestamp_s >= ts) break;
    }
    return samples[best];
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
    const char* usage =
        "Usage: export_imu_state_propagation"
        " <euroc_imu_csv_path> <output_csv_path> [--init-from-gt]\n";

    if (argc < 3 || argc > 4) { std::cerr << usage; return EXIT_FAILURE; }
    bool init_from_gt = false;
    if (argc == 4) {
        if (std::string(argv[3]) != "--init-from-gt") {
            std::cerr << "Error: unknown argument: " << argv[3] << '\n' << usage;
            return EXIT_FAILURE;
        }
        init_from_gt = true;
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

    // Load GT samples when requested.
    // IMU path: <seq_root>/mav0/imu0/data.csv
    // GT path:  <seq_root>/mav0/state_groundtruth_estimate0/data.csv
    std::vector<GtSample> gt_samples;
    if (init_from_gt) {
        namespace fs = std::filesystem;
        const fs::path gt_path =
            fs::path(argv[1]).parent_path().parent_path()
            / "state_groundtruth_estimate0" / "data.csv";
        gt_samples = read_gt_csv(gt_path.string());
        if (gt_samples.empty()) {
            std::cerr << "Error: GT CSV is empty: " << gt_path << '\n';
            return EXIT_FAILURE;
        }
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
            // state[0]: initialized at t[0]; timestamp consistent with default behavior.
            state.timestamp_s     = curr_meas.timestamp_s;
            state.gyro_bias_radps = Eigen::Vector3d::Zero();
            state.accel_bias_mps2 = Eigen::Vector3d::Zero();

            if (init_from_gt) {
                const GtSample& gt = nearest_gt(curr_meas.timestamp_s, gt_samples);

                const double q_norm = std::sqrt(
                    gt.q_w*gt.q_w + gt.q_x*gt.q_x + gt.q_y*gt.q_y + gt.q_z*gt.q_z);
                if (q_norm < 1e-10) {
                    std::cerr << "Error: GT quaternion norm near zero at t="
                              << gt.timestamp_s << '\n';
                    return EXIT_FAILURE;
                }
                const Eigen::Quaterniond q(
                    gt.q_w/q_norm, gt.q_x/q_norm, gt.q_y/q_norm, gt.q_z/q_norm);
                state.R_W_B = q.toRotationMatrix();
                state.p_W_B = gt.p;
                state.v_W_B = gt.v;

                std::cerr << "export_imu_state_propagation: --init-from-gt\n"
                          << "  first IMU t:  " << curr_meas.timestamp_s << " s\n"
                          << "  matched GT t: " << gt.timestamp_s << " s\n"
                          << "  initial p:    ["
                              << gt.p.x() << ", " << gt.p.y() << ", " << gt.p.z() << "] m\n"
                          << "  initial v:    ["
                              << gt.v.x() << ", " << gt.v.y() << ", " << gt.v.z() << "] m/s\n"
                          << "  initial q:    [w=" << gt.q_w/q_norm
                              << ", x=" << gt.q_x/q_norm
                              << ", y=" << gt.q_y/q_norm
                              << ", z=" << gt.q_z/q_norm << "]\n";
            } else {
                state.R_W_B = Eigen::Matrix3d::Identity();
                state.p_W_B = Eigen::Vector3d::Zero();
                state.v_W_B = Eigen::Vector3d::Zero();

                std::cerr << "export_imu_state_propagation: default init\n"
                          << "  first IMU t:  " << curr_meas.timestamp_s << " s\n"
                          << "  initial p:    [0, 0, 0] m\n"
                          << "  initial v:    [0, 0, 0] m/s\n"
                          << "  initial q:    [w=1, x=0, y=0, z=0]\n";
            }

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
