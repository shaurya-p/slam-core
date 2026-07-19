#include "slam_core/factors/prior_factors.hpp"

#include <cmath>
#include <stdexcept>

#include "slam_core/geometry/so3.hpp"

namespace slam_core::factors {

So3PriorFactor::So3PriorFactor(optim::So3Variable*    variable,
                               const Eigen::Matrix3d& R_prior,
                               double                 sigma)
    : Factor({variable}), var_(variable), R_prior_(R_prior), sigma_(sigma) {
    if (!geometry::is_valid_rotation(R_prior_)) {
        throw std::invalid_argument("So3PriorFactor: R_prior is not a valid rotation");
    }
    if (!std::isfinite(sigma_) || sigma_ <= 0.0) {
        throw std::invalid_argument("So3PriorFactor: sigma must be positive and finite");
    }
}

Eigen::VectorXd So3PriorFactor::residual() const {
    return geometry::log_so3(R_prior_.transpose() * var_->R()) / sigma_;
}

std::vector<Eigen::MatrixXd> So3PriorFactor::jacobians() const {
    const Eigen::Vector3d r = geometry::log_so3(R_prior_.transpose() * var_->R());
    return {geometry::right_jacobian_so3_inverse(r) / sigma_};
}

VectorPriorFactor::VectorPriorFactor(optim::VectorVariable* variable,
                                     const Eigen::VectorXd& prior,
                                     double                 sigma)
    : Factor({variable}), var_(variable), prior_(prior), sigma_(sigma) {
    if (prior_.size() != variable->tangent_dim()) {
        throw std::invalid_argument("VectorPriorFactor: prior size mismatch");
    }
    if (!std::isfinite(sigma_) || sigma_ <= 0.0) {
        throw std::invalid_argument("VectorPriorFactor: sigma must be positive and finite");
    }
}

int VectorPriorFactor::residual_dim() const {
    return static_cast<int>(prior_.size());
}

Eigen::VectorXd VectorPriorFactor::residual() const {
    return (var_->vec() - prior_) / sigma_;
}

std::vector<Eigen::MatrixXd> VectorPriorFactor::jacobians() const {
    return {Eigen::MatrixXd::Identity(prior_.size(), prior_.size()) / sigma_};
}

}  // namespace slam_core::factors
