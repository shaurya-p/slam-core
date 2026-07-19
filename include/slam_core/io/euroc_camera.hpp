#pragma once

#include <filesystem>

#include <Eigen/Core>

namespace slam_core::io {

// EuRoC camera calibration parsed from a cam sensor.yaml.
//
// T_B_C: extrinsics mapping camera frame C into body frame B
// (EuRoC "T_BS", sensor-to-body): p_B = R_B_C * p_C + t_B_C.
// Intrinsics are pinhole [fx, fy, cx, cy] in pixels; distortion is
// radial-tangential [k1, k2, p1, p2] (not yet applied anywhere —
// EuRoC images are distorted, factors currently assume undistorted
// pixel measurements).
struct EurocCameraCalib {
    Eigen::Matrix3d R_B_C;
    Eigen::Vector3d t_B_C;
    double          fx = 0.0, fy = 0.0, cx = 0.0, cy = 0.0;  // pixels
    int             width = 0, height = 0;                   // pixels
    double          rate_hz           = 0.0;
    Eigen::Vector4d distortion_radtan = Eigen::Vector4d::Zero();  // k1,k2,p1,p2
};

// Parses the fixed EuRoC sensor.yaml layout (line-based; not a general
// YAML parser). Validates that R_B_C is a rotation and fx/fy positive.
//
// Throws std::runtime_error if the file cannot be opened;
// std::invalid_argument if required fields are missing or invalid.
EurocCameraCalib read_euroc_camera_yaml(const std::filesystem::path& path);

}  // namespace slam_core::io
