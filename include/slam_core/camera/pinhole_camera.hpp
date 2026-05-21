#pragma once

#include <Eigen/Core>

namespace slam_core::camera {

// Pinhole camera model with no distortion.
//
// Camera frame convention: x right, y down, z forward.
// Projection: u = fx * X / Z + cx,  v = fy * Y / Z + cy
// Reprojection error: observed_px − project(p_C)
struct PinholeCamera {
    double fx;  // pixels
    double fy;  // pixels
    double cx;  // pixels, principal point x
    double cy;  // pixels, principal point y

    // Throws std::invalid_argument if fx or fy are non-positive or any
    // intrinsic is non-finite.
    PinholeCamera(double fx, double fy, double cx, double cy);

    // Returns true if all intrinsics are valid (finite, fx/fy > 0).
    bool is_valid() const;

    // Projects p_C (3D point in camera frame) to image pixel [u, v].
    // Throws std::invalid_argument if p_C has non-finite coordinates or Z <= 0.
    Eigen::Vector2d project(const Eigen::Vector3d& p_C) const;

    // Reprojection error: observed_px − project(p_C).
    // Positive residual in u means observed is to the right of predicted.
    // Throws if project(p_C) would throw.
    Eigen::Vector2d reprojection_error(const Eigen::Vector3d& p_C,
                                       const Eigen::Vector2d& observed_px) const;

    // Unprojects pixel [u, v] to a unit-length bearing vector in camera frame C.
    // x = (u - cx) / fx,  y = (v - cy) / fy,  bearing = normalize([x, y, 1])
    // Throws std::invalid_argument if pixel has non-finite coordinates.
    Eigen::Vector3d unproject_to_bearing(const Eigen::Vector2d& pixel) const;
};

}  // namespace slam_core::camera
