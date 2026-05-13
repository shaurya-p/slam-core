#include "slam_core/geometry/so3.hpp"

#include <algorithm>
#include <cmath>
#include <Eigen/LU>

namespace slam_core::geometry {

Eigen::Matrix3d skew(const Eigen::Vector3d& w) {
    Eigen::Matrix3d S;
    S <<     0, -w(2),  w(1),
          w(2),     0, -w(0),
         -w(1),  w(0),     0;
    return S;
}

Eigen::Matrix3d exp_so3(const Eigen::Vector3d& w) {
    const double theta = w.norm();
    if (theta < 1e-8) {
        return Eigen::Matrix3d::Identity() + skew(w);
    }
    const Eigen::Matrix3d K = skew(w / theta);
    return Eigen::Matrix3d::Identity()
        + std::sin(theta) * K
        + (1.0 - std::cos(theta)) * K * K;
}

Eigen::Vector3d log_so3(const Eigen::Matrix3d& R) {
    if ((R - Eigen::Matrix3d::Identity()).norm() < 1e-8) {
        return Eigen::Vector3d::Zero();
    }
    const double cos_theta = std::clamp(0.5 * (R.trace() - 1.0), -1.0, 1.0);
    const double theta = std::acos(cos_theta);
    const double sin_theta = std::sin(theta);

    // Near pi: sin_theta -> 0, so R - R^T extraction becomes unstable.
    // Use diagonal of R to recover axis: n_i^2 = (R_ii - cos_theta) / (1 - cos_theta)
    if (std::abs(sin_theta) < 1e-4) {
        const double one_minus_cos = 1.0 - cos_theta;
        const Eigen::Vector3d n2{
            (R(0, 0) - cos_theta) / one_minus_cos,
            (R(1, 1) - cos_theta) / one_minus_cos,
            (R(2, 2) - cos_theta) / one_minus_cos
        };
        int i = 0;
        if (n2(1) > n2(i)) i = 1;
        if (n2(2) > n2(i)) i = 2;

        Eigen::Vector3d axis = Eigen::Vector3d::Zero();
        axis(i) = std::sqrt(std::max(0.0, n2(i)));
        const double denom = 2.0 * one_minus_cos * axis(i);
        for (int j = 0; j < 3; ++j) {
            if (j != i) {
                axis(j) = (R(i, j) + R(j, i)) / denom;
            }
        }
        return theta * axis;
    }

    // General case: R - R^T = 2 * sin(theta) * skew(axis)
    const Eigen::Vector3d axis{
        R(2, 1) - R(1, 2),
        R(0, 2) - R(2, 0),
        R(1, 0) - R(0, 1)
    };
    return (theta / (2.0 * sin_theta)) * axis;
}

bool is_valid_rotation(const Eigen::Matrix3d& R, double tol) {
    const bool orthonormal =
        (R * R.transpose() - Eigen::Matrix3d::Identity()).norm() < tol;
    const bool det_ok = std::abs(R.determinant() - 1.0) < tol;
    return orthonormal && det_ok;
}

}  // namespace slam_core::geometry
