#pragma once

#include <unordered_map>
#include <vector>

#include <Eigen/Core>

#include "slam_core/optim/factor.hpp"
#include "slam_core/optim/variable.hpp"

namespace slam_core::optim {

// A nonlinear least-squares problem: cost(x) = ½ Σ_f ‖r_f(x)‖².
//
// Holds non-owning pointers; variables and factors must outlive the
// Problem. Each variable gets a contiguous tangent-space slice in
// registration order; total_tangent_dim() is the dense state size N.
class Problem {
public:
    // Throws std::invalid_argument on null or duplicate registration.
    void add_variable(Variable* variable);

    // All of the factor's variables must already be registered.
    // Throws std::invalid_argument otherwise.
    void add_factor(Factor* factor);

    double cost() const;

    // Dense Gauss-Newton normal equations at the current state:
    //   H = Σ JᵀJ  (N×N, symmetric),  b = Σ Jᵀr  (N).
    // The minimizing step solves H delta = -b.
    void build_normal_equations(Eigen::MatrixXd& H, Eigen::VectorXd& b) const;

    // Retracts each variable by its slice of delta (size N).
    // Throws std::invalid_argument if delta has the wrong size.
    void apply_step(const Eigen::VectorXd& delta);

    int total_tangent_dim() const { return total_dim_; }

    // Tangent-space offset of a registered variable.
    // Throws std::invalid_argument for unregistered variables.
    int tangent_offset(const Variable* variable) const;

    const std::vector<Variable*>& variables() const { return variables_; }
    const std::vector<Factor*>&   factors() const { return factors_; }

private:
    std::vector<Variable*>                   variables_;
    std::vector<Factor*>                     factors_;
    std::unordered_map<const Variable*, int> offsets_;
    int                                      total_dim_ = 0;
};

}  // namespace slam_core::optim
