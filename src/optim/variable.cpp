#include "slam_core/optim/variable.hpp"

#include <stdexcept>
#include <utility>

#include "slam_core/geometry/so3.hpp"

namespace slam_core::optim {

namespace {

void check_delta(const Eigen::VectorXd& delta, int expected_dim, const char* who) {
    if (delta.size() != expected_dim) {
        throw std::invalid_argument(std::string(who) + ": delta has wrong size");
    }
    if (!delta.allFinite()) {
        throw std::invalid_argument(std::string(who) + ": delta is not finite");
    }
}

}  // namespace

So3Variable::So3Variable(const Eigen::Matrix3d& R) : R_(R) {
    if (!geometry::is_valid_rotation(R_)) {
        throw std::invalid_argument("So3Variable: R is not a valid rotation");
    }
}

void So3Variable::retract(const Eigen::VectorXd& delta) {
    check_delta(delta, 3, "So3Variable::retract");
    R_ = R_ * geometry::exp_so3(delta);
}

Eigen::VectorXd So3Variable::value() const {
    Eigen::VectorXd v(9);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) v(3 * r + c) = R_(r, c);
    return v;
}

void So3Variable::set_value(const Eigen::VectorXd& value) {
    if (value.size() != 9) {
        throw std::invalid_argument("So3Variable::set_value: expected 9 entries");
    }
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) R_(r, c) = value(3 * r + c);
}

VectorVariable::VectorVariable(Eigen::VectorXd v) : v_(std::move(v)) {
    if (v_.size() == 0) {
        throw std::invalid_argument("VectorVariable: empty state");
    }
    if (!v_.allFinite()) {
        throw std::invalid_argument("VectorVariable: state is not finite");
    }
}

void VectorVariable::retract(const Eigen::VectorXd& delta) {
    check_delta(delta, tangent_dim(), "VectorVariable::retract");
    v_ += delta;
}

void VectorVariable::set_value(const Eigen::VectorXd& value) {
    if (value.size() != v_.size()) {
        throw std::invalid_argument("VectorVariable::set_value: wrong size");
    }
    v_ = value;
}

}  // namespace slam_core::optim
