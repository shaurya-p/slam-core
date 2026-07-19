#include "slam_core/io/euroc_camera.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "slam_core/geometry/so3.hpp"

namespace slam_core::io {

namespace {

// Extracts all numbers from bracketed lists that follow "key:" in the
// fixed EuRoC layout, where the list may span multiple lines (T_BS data).
std::vector<double> numbers_after_key(const std::string& text,
                                      const std::string& key,
                                      std::size_t        expected) {
    const std::size_t key_pos = text.find(key);
    if (key_pos == std::string::npos) {
        throw std::invalid_argument("EuRoC sensor.yaml: missing key: " + key);
    }
    const std::size_t open  = text.find('[', key_pos);
    const std::size_t close = text.find(']', open);
    if (open == std::string::npos || close == std::string::npos) {
        throw std::invalid_argument("EuRoC sensor.yaml: malformed list for key: " + key);
    }
    std::string list = text.substr(open + 1, close - open - 1);
    for (char& ch : list) {
        if (ch == ',' || ch == '\n') ch = ' ';
    }
    std::istringstream  ss(list);
    std::vector<double> values;
    double              v;
    while (ss >> v) values.push_back(v);
    if (values.size() != expected) {
        throw std::invalid_argument("EuRoC sensor.yaml: expected " + std::to_string(expected) +
                                    " values for key: " + key);
    }
    return values;
}

double scalar_after_key(const std::string& text, const std::string& key) {
    const std::size_t key_pos = text.find(key);
    if (key_pos == std::string::npos) {
        throw std::invalid_argument("EuRoC sensor.yaml: missing key: " + key);
    }
    std::istringstream ss(text.substr(key_pos + key.size()));
    double             v;
    if (!(ss >> v)) {
        throw std::invalid_argument("EuRoC sensor.yaml: malformed value for key: " + key);
    }
    return v;
}

}  // namespace

EurocCameraCalib read_euroc_camera_yaml(const std::filesystem::path& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("cannot open EuRoC sensor.yaml: " + path.string());
    }
    std::stringstream buffer;
    buffer << f.rdbuf();
    const std::string text = buffer.str();

    EurocCameraCalib calib;

    const std::vector<double> T = numbers_after_key(text, "T_BS", 16);
    for (int r = 0; r < 3; ++r) {
        for (int col = 0; col < 3; ++col) calib.R_B_C(r, col) = T[4 * r + col];
        calib.t_B_C(r) = T[4 * r + 3];
    }
    if (!geometry::is_valid_rotation(calib.R_B_C, 1e-4)) {
        throw std::invalid_argument("EuRoC sensor.yaml: T_BS rotation block is not a rotation");
    }

    const std::vector<double> res = numbers_after_key(text, "resolution", 2);
    calib.width                   = static_cast<int>(res[0]);
    calib.height                  = static_cast<int>(res[1]);

    const std::vector<double> intr = numbers_after_key(text, "intrinsics", 4);
    calib.fx                       = intr[0];
    calib.fy                       = intr[1];
    calib.cx                       = intr[2];
    calib.cy                       = intr[3];
    if (!(calib.fx > 0.0) || !(calib.fy > 0.0)) {
        throw std::invalid_argument("EuRoC sensor.yaml: fx and fy must be positive");
    }

    const std::vector<double> dist = numbers_after_key(text, "distortion_coefficients", 4);
    calib.distortion_radtan        = Eigen::Vector4d(dist[0], dist[1], dist[2], dist[3]);

    calib.rate_hz = scalar_after_key(text, "rate_hz:");
    return calib;
}

}  // namespace slam_core::io
