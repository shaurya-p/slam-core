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

    // Non-throwing projection for optimizer use: returns false (leaving
    // pixel untouched) if p_C is non-finite or Z <= z_min. An optimizer
    // line search may legitimately probe behind-camera states; callers
    // decide the failure policy instead of unwinding.
    bool try_project(const Eigen::Vector3d& p_C, Eigen::Vector2d& pixel, double z_min = 1e-6) const;

    // Jacobian of project() w.r.t. p_C, evaluated at p_C (2x3):
    //   d[u,v]/dp_C = [ fx/Z    0    -fx*X/Z² ]
    //                 [   0   fy/Z  -fy*Y/Z²  ]
    // Throws like project() (non-finite p_C or Z <= 0).
    Eigen::Matrix<double, 2, 3> project_jacobian(const Eigen::Vector3d& p_C) const;
};

}  // namespace slam_core::camera
