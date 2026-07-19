#include "slam_core/camera/pinhole_camera.hpp"

#include <cmath>
#include <stdexcept>

namespace slam_core::camera {

PinholeCamera::PinholeCamera(double fx_, double fy_, double cx_, double cy_)
    : fx(fx_), fy(fy_), cx(cx_), cy(cy_) {
    if (!std::isfinite(fx_) || !std::isfinite(fy_) || !std::isfinite(cx_) || !std::isfinite(cy_)) {
        throw std::invalid_argument("PinholeCamera: intrinsics must be finite");
    }
    if (fx_ <= 0.0 || fy_ <= 0.0) {
        throw std::invalid_argument("PinholeCamera: fx and fy must be positive");
    }
}

bool PinholeCamera::is_valid() const {
    return std::isfinite(fx) && std::isfinite(fy) && std::isfinite(cx) && std::isfinite(cy) &&
           fx > 0.0 && fy > 0.0;
}

Eigen::Vector2d PinholeCamera::project(const Eigen::Vector3d& p_C) const {
    if (!p_C.allFinite()) {
        throw std::invalid_argument("PinholeCamera::project: p_C has non-finite coordinates");
    }
    if (p_C.z() <= 0.0) {
        throw std::invalid_argument(
            "PinholeCamera::project: Z must be positive (point must be in front of camera)");
    }
    return {fx * p_C.x() / p_C.z() + cx, fy * p_C.y() / p_C.z() + cy};
}

Eigen::Vector2d PinholeCamera::reprojection_error(const Eigen::Vector3d& p_C,
                                                  const Eigen::Vector2d& observed_px) const {
    return observed_px - project(p_C);
}

Eigen::Vector3d PinholeCamera::unproject_to_bearing(const Eigen::Vector2d& pixel) const {
    if (!pixel.allFinite()) {
        throw std::invalid_argument(
            "PinholeCamera::unproject_to_bearing: pixel has non-finite coordinates");
    }
    const Eigen::Vector3d ray((pixel.x() - cx) / fx, (pixel.y() - cy) / fy, 1.0);
    return ray.normalized();
}

bool PinholeCamera::try_project(const Eigen::Vector3d& p_C,
                                Eigen::Vector2d&       pixel,
                                double                 z_min) const {
    if (!p_C.allFinite() || p_C.z() <= z_min) {
        return false;
    }
    pixel = {fx * p_C.x() / p_C.z() + cx, fy * p_C.y() / p_C.z() + cy};
    return true;
}

Eigen::Matrix<double, 2, 3> PinholeCamera::project_jacobian(const Eigen::Vector3d& p_C) const {
    if (!p_C.allFinite()) {
        throw std::invalid_argument(
            "PinholeCamera::project_jacobian: p_C has non-finite coordinates");
    }
    if (p_C.z() <= 0.0) {
        throw std::invalid_argument("PinholeCamera::project_jacobian: Z must be positive");
    }
    const double                z_inv  = 1.0 / p_C.z();
    const double                z_inv2 = z_inv * z_inv;
    Eigen::Matrix<double, 2, 3> J;
    J << fx * z_inv, 0.0, -fx * p_C.x() * z_inv2, 0.0, fy * z_inv, -fy * p_C.y() * z_inv2;
    return J;
}

}  // namespace slam_core::camera
