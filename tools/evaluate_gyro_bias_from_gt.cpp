// Evaluate approximate gyro bias from EuRoC ground truth.
//
// Usage:
//   evaluate_gyro_bias_from_gt <imu_csv> <gt_csv> <output_csv>
//                              [--window-duration <sec>]
//
// Convention (confirmed):
//   GT quaternion is interpreted as q_W_B (EuRoC convention, Hamilton, w-first).
//   R_W_B = Eigen::Quaterniond(q_w, q_x, q_y, q_z).toRotationMatrix()
//   GT relative rotation: R_rel_gt = R_W_B0.transpose() * R_W_B1 = R_B0_B1
//
//   propagate_gyro uses right-multiply: R_next = R * exp_so3(omega_B * dt).
//   Starting from Identity, integration produces R_B0_B1 in the same sense.
//
// For each non-overlapping window [t0, t1] over the GT coverage interval:
//   R_rel_imu     = integrate raw gyro over [t0, t1] starting from Identity
//   R_rel_gt      = R_W_B0.transpose() * R_W_B1
//   R_err         = R_rel_imu.transpose() * R_rel_gt
//   integrated_dt = t1 - t0  (actual span of integrated IMU samples, not nominal window_duration)
//   gyro_bias_est = -log_so3(R_err) / integrated_dt
//
// Sign convention: omega_meas = omega_true + bias.
// log_so3(R_err) / integrated_dt is the rotation correction toward GT.
// Negating gives the physical gyro bias.
//
// Output CSV columns:
//   timestamp_start_s, timestamp_end_s, dt_s, integrated_dt_s,
//   gyro_bias_est_x_radps, gyro_bias_est_y_radps, gyro_bias_est_z_radps,
//   gyro_bias_est_norm_radps, error_angle_deg

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "slam_core/geometry/so3.hpp"
#include "slam_core/imu/gyro_propagation.hpp"
#include "slam_core/imu/imu_measurement.hpp"

namespace {

constexpr double kPi = 3.14159265358979323846;

// EuRoC IMU CSV: timestamp_ns, wx, wy, wz, ax, ay, az
bool parse_imu_row(const std::string& line,
                   slam_core::imu::ImuMeasurement& out) {
    std::istringstream ss(line);
    std::string tok;
    double vals[7];
    for (int i = 0; i < 7; ++i) {
        if (!std::getline(ss, tok, ',')) return false;
        try { vals[i] = std::stod(tok); } catch (...) { return false; }
    }
    out.timestamp_s = vals[0] * 1e-9;
    out.gyro_radps  = {vals[1], vals[2], vals[3]};
    out.accel_mps2  = {vals[4], vals[5], vals[6]};
    return true;
}

// EuRoC GT CSV: timestamp_ns, p_x, p_y, p_z, q_w, q_x, q_y, q_z, v_x, ...
struct GtSample {
    double timestamp_s;
    double q_w, q_x, q_y, q_z;
};

bool parse_gt_row(const std::string& line, GtSample& out) {
    std::istringstream ss(line);
    std::string tok;
    double vals[8];
    for (int i = 0; i < 8; ++i) {
        if (!std::getline(ss, tok, ',')) return false;
        try { vals[i] = std::stod(tok); } catch (...) { return false; }
    }
    out.timestamp_s = vals[0] * 1e-9;
    out.q_w = vals[4]; out.q_x = vals[5]; out.q_y = vals[6]; out.q_z = vals[7];
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

Eigen::Matrix3d gt_rotation(const GtSample& s) {
    const double norm = std::sqrt(
        s.q_w*s.q_w + s.q_x*s.q_x + s.q_y*s.q_y + s.q_z*s.q_z);
    const Eigen::Quaterniond q(s.q_w/norm, s.q_x/norm, s.q_y/norm, s.q_z/norm);
    return q.toRotationMatrix();
}

// Nearest-neighbor timestamp lookup (samples sorted ascending, early exit).
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

}  // namespace

int main(int argc, char* argv[]) {
    const char* usage =
        "Usage: evaluate_gyro_bias_from_gt"
        " <imu_csv> <gt_csv> <output_csv>"
        " [--window-duration <sec>]\n";

    if (argc < 4) { std::cerr << usage; return EXIT_FAILURE; }

    double window_duration = 1.0;
    for (int i = 4; i < argc; ++i) {
        if (std::string(argv[i]) == "--window-duration") {
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
        } else {
            std::cerr << "Error: unknown argument: " << argv[i] << '\n' << usage;
            return EXIT_FAILURE;
        }
    }

    // Load IMU samples into memory.
    std::ifstream imu_file(argv[1]);
    if (!imu_file.is_open()) {
        std::cerr << "Error: cannot open IMU CSV: " << argv[1] << '\n';
        return EXIT_FAILURE;
    }
    std::string line;
    std::getline(imu_file, line);  // skip header

    std::vector<slam_core::imu::ImuMeasurement> imu_data;
    int imu_skipped = 0;
    while (std::getline(imu_file, line)) {
        if (line.empty() || line[0] == '#') continue;
        slam_core::imu::ImuMeasurement m;
        if (parse_imu_row(line, m)) imu_data.push_back(m);
        else ++imu_skipped;
    }
    if (imu_skipped > 0)
        std::cerr << "Warning: " << imu_skipped << " IMU rows skipped (unparseable)\n";
    if (imu_data.size() < 2) {
        std::cerr << "Error: too few IMU samples\n";
        return EXIT_FAILURE;
    }

    // Load GT samples.
    const std::vector<GtSample> gt_data = read_gt_csv(argv[2]);
    if (gt_data.empty()) {
        std::cerr << "Error: GT CSV is empty\n";
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
    out_file << "timestamp_start_s,timestamp_end_s,dt_s,integrated_dt_s"
             << ",gyro_bias_est_x_radps,gyro_bias_est_y_radps,gyro_bias_est_z_radps"
             << ",gyro_bias_est_norm_radps,error_angle_deg\n";

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
    std::vector<double> error_angles_deg;
    Eigen::Vector3d bias_sum            = Eigen::Vector3d::Zero();
    double          integrated_dt_sum   = 0.0;
    double          dt_abs_diff_sum     = 0.0;
    double first_imu_t0 = 0.0, last_imu_t1 = 0.0;
    double first_gt_t0  = 0.0, last_gt_t1  = 0.0;
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

        const double t0 = imu_data[i0].timestamp_s;
        const double t1 = imu_data[i1].timestamp_s;
        const double dt = t1 - t0;
        if (dt <= 0.0) { window_start_t += window_duration; continue; }

        // GT relative rotation: R_B0_B1_gt = R_W_B0.T * R_W_B1.
        const Eigen::Matrix3d R_W_B0  = gt_rotation(nearest_gt(t0, gt_data));
        const Eigen::Matrix3d R_W_B1  = gt_rotation(nearest_gt(t1, gt_data));
        const Eigen::Matrix3d R_rel_gt = R_W_B0.transpose() * R_W_B1;

        // IMU relative rotation: integrate raw gyro starting from Identity.
        Eigen::Matrix3d R_rel_imu = Eigen::Matrix3d::Identity();
        for (std::size_t i = i0; i < i1; ++i) {
            const double dt_i =
                imu_data[i + 1].timestamp_s - imu_data[i].timestamp_s;
            R_rel_imu = slam_core::imu::propagate_gyro(
                R_rel_imu, imu_data[i].gyro_radps, dt_i);
        }

        // R_err = R_rel_imu.T * R_rel_gt
        // log_so3(R_err) / integrated_dt = rotation correction toward GT
        // gyro_bias_est = -log_so3(R_err) / integrated_dt  (omega_meas = omega_true + bias)
        // integrated_dt = t1 - t0 = actual IMU span; bias denominator uses this, not window_duration.
        const double          integrated_dt = dt;  // dt = t1 - t0 (actual), confirmed equal
        const Eigen::Matrix3d R_err         = R_rel_imu.transpose() * R_rel_gt;
        const Eigen::Vector3d log_err       = slam_core::geometry::log_so3(R_err);
        const Eigen::Vector3d bias_est      = -log_err / integrated_dt;
        const double          error_deg     = log_err.norm() * (180.0 / kPi);

        // Nearest GT timestamps actually used for this window.
        const double gt_t0 = nearest_gt(t0, gt_data).timestamp_s;
        const double gt_t1 = nearest_gt(t1, gt_data).timestamp_s;

        out_file << t0 << ',' << t1 << ',' << dt << ',' << integrated_dt
                 << ',' << bias_est.x()
                 << ',' << bias_est.y()
                 << ',' << bias_est.z()
                 << ',' << bias_est.norm()
                 << ',' << error_deg << '\n';

        bias_sum          += bias_est;
        integrated_dt_sum += integrated_dt;
        dt_abs_diff_sum   += std::abs(integrated_dt - window_duration);
        bias_norms.push_back(bias_est.norm());
        error_angles_deg.push_back(error_deg);

        if (n_windows == 0) {
            first_imu_t0 = t0;
            first_gt_t0  = gt_t0;
        }
        last_imu_t1 = t1;
        last_gt_t1  = gt_t1;
        ++n_windows;

        // Advance base index to avoid re-scanning from the start next iteration.
        idx_base = i1;
        window_start_t += window_duration;
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
    const double mean_err =
        std::accumulate(error_angles_deg.begin(), error_angles_deg.end(), 0.0) / n_windows;
    const double max_err =
        *std::max_element(error_angles_deg.begin(), error_angles_deg.end());
    const double mean_integrated_dt  = integrated_dt_sum / n_windows;
    const double mean_dt_abs_diff    = dt_abs_diff_sum   / n_windows;

    std::cerr << std::setprecision(6) << std::fixed;
    std::cerr << "evaluate_gyro_bias_from_gt:\n"
              << "  windows:              " << n_windows << "\n"
              << "  nominal window dur:   " << window_duration    << " s\n"
              << "  mean integrated dt:   " << mean_integrated_dt << " s\n"
              << "  mean |dt - nominal|:  " << mean_dt_abs_diff   << " s\n"
              << "  first IMU t0:         " << first_imu_t0 << " s\n"
              << "  last  IMU t1:         " << last_imu_t1  << " s\n"
              << "  first GT  t0:         " << first_gt_t0  << " s\n"
              << "  last  GT  t1:         " << last_gt_t1   << " s\n"
              << "  mean bias est:        ["
                  << mean_bias.x() << ", "
                  << mean_bias.y() << ", "
                  << mean_bias.z() << "] rad/s\n"
              << "  mean  bias norm:      " << mean_norm   << " rad/s\n"
              << "  median bias norm:     " << median_norm << " rad/s\n"
              << "  mean  error:          " << mean_err    << " deg\n"
              << "  max   error:          " << max_err     << " deg\n"
              << "  output:               " << argv[3] << '\n';

    return EXIT_SUCCESS;
}
