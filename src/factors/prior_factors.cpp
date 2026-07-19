#include "slam_core/factors/prior_factors.hpp"

#include <stdexcept>

#include "slam_core/geometry/so3.hpp"

namespace slam_core::factors {

So3PriorFactor::So3PriorFactor(optim::So3Variable* variable, const Eigen::Matrix3d& R_prior)
    : Factor({variable}), var_(variable), R_prior_(R_prior) {
    if (!geometry::is_valid_rotation(R_prior_)) {
        throw std::invalid_argument("So3PriorFactor: R_prior is not a valid rotation");
    }
}

Eigen::VectorXd So3PriorFactor::residual() const {
    return geometry::log_so3(R_prior_.transpose() * var_->R());
}

std::vector<Eigen::MatrixXd> So3PriorFactor::jacobians() const {
    const Eigen::Vector3d r = geometry::log_so3(R_prior_.transpose() * var_->R());
    return {geometry::right_jacobian_so3_inverse(r)};
}

VectorPriorFactor::VectorPriorFactor(optim::VectorVariable* variable, const Eigen::VectorXd& prior)
    : Factor({variable}), var_(variable), prior_(prior) {
    if (prior_.size() != variable->tangent_dim()) {
        throw std::invalid_argument("VectorPriorFactor: prior size mismatch");
    }
}

int VectorPriorFactor::residual_dim() const {
    return static_cast<int>(prior_.size());
}

Eigen::VectorXd VectorPriorFactor::residual() const {
    return var_->vec() - prior_;
}

std::vector<Eigen::MatrixXd> VectorPriorFactor::jacobians() const {
    return {Eigen::MatrixXd::Identity(prior_.size(), prior_.size())};
}

}  // namespace slam_core::factors
