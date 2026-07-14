#include "slam_core/io/euroc_csv.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace slam_core::io {

namespace {

// Parses the first n comma-separated numeric columns of line into vals.
bool parse_columns(const std::string& line, double* vals, int n) {
    std::istringstream ss(line);
    std::string tok;
    for (int i = 0; i < n; ++i) {
        if (!std::getline(ss, tok, ',')) return false;
        try {
            vals[i] = std::stod(tok);
        } catch (...) {
            return false;
        }
    }
    return true;
}

std::ifstream open_or_throw(const std::filesystem::path& path,
                            const char* what) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error(std::string("cannot open ") + what + ": " +
                                 path.string());
    }
    return f;
}

}  // namespace

std::vector<imu::ImuMeasurement> read_euroc_imu_csv(
    const std::filesystem::path& path, int* skipped_rows) {
    std::ifstream f = open_or_throw(path, "EuRoC IMU CSV");

    std::string line;
    if (!std::getline(f, line)) {
        throw std::invalid_argument("EuRoC IMU CSV is empty: " + path.string());
    }

    std::vector<imu::ImuMeasurement> samples;
    int skipped = 0;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        double vals[7];
        if (!parse_columns(line, vals, 7)) {
            ++skipped;
            continue;
        }
        imu::ImuMeasurement m;
        m.timestamp_s = vals[0] * 1e-9;                // ns -> s
        m.gyro_radps  = {vals[1], vals[2], vals[3]};   // wx, wy, wz
        m.accel_mps2  = {vals[4], vals[5], vals[6]};   // ax, ay, az
        samples.push_back(m);
    }

    if (skipped_rows != nullptr) *skipped_rows = skipped;
    if (samples.empty()) {
        throw std::invalid_argument("EuRoC IMU CSV has no parseable rows: " +
                                    path.string());
    }
    return samples;
}

std::vector<EurocGtSample> read_euroc_gt_csv(
    const std::filesystem::path& path, int* skipped_rows) {
    std::ifstream f = open_or_throw(path, "EuRoC GT CSV");

    std::string line;
    if (!std::getline(f, line)) {
        throw std::invalid_argument("EuRoC GT CSV is empty: " + path.string());
    }

    std::vector<EurocGtSample> samples;
    int skipped = 0;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        double vals[11];
        if (!parse_columns(line, vals, 11)) {
            ++skipped;
            continue;
        }
        // Same normalization arithmetic as the pre-consolidation tools:
        // norm over (w, x, y, z), each component divided individually.
        const double q_w = vals[4], q_x = vals[5], q_y = vals[6], q_z = vals[7];
        const double norm =
            std::sqrt(q_w * q_w + q_x * q_x + q_y * q_y + q_z * q_z);
        if (norm < 1e-10) {
            ++skipped;
            continue;
        }
        EurocGtSample s;
        s.timestamp_s = vals[0] * 1e-9;  // ns -> s
        s.p_W_B       = {vals[1], vals[2], vals[3]};
        s.q_W_B = Eigen::Quaterniond(q_w / norm, q_x / norm, q_y / norm,
                                     q_z / norm);
        s.v_W_B = {vals[8], vals[9], vals[10]};
        samples.push_back(s);
    }

    if (skipped_rows != nullptr) *skipped_rows = skipped;
    if (samples.empty()) {
        throw std::invalid_argument("EuRoC GT CSV has no parseable rows: " +
                                    path.string());
    }
    return samples;
}

const EurocGtSample& nearest_gt(double timestamp_s,
                                const std::vector<EurocGtSample>& samples) {
    if (samples.empty()) {
        throw std::invalid_argument("nearest_gt: samples is empty");
    }
    auto it = std::lower_bound(
        samples.begin(), samples.end(), timestamp_s,
        [](const EurocGtSample& s, double t) { return s.timestamp_s < t; });
    if (it == samples.end()) return samples.back();
    if (it == samples.begin()) return samples.front();
    const auto prev = std::prev(it);
    return (timestamp_s - prev->timestamp_s <= it->timestamp_s - timestamp_s)
               ? *prev
               : *it;
}

}  // namespace slam_core::io
