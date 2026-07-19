#include "slam_core/optim/problem.hpp"

#include <stdexcept>

namespace slam_core::optim {

void Problem::add_variable(Variable* variable) {
    if (variable == nullptr) {
        throw std::invalid_argument("Problem::add_variable: null variable");
    }
    if (offsets_.count(variable) > 0) {
        throw std::invalid_argument("Problem::add_variable: variable already registered");
    }
    offsets_[variable] = total_dim_;
    total_dim_ += variable->tangent_dim();
    variables_.push_back(variable);
}

void Problem::add_factor(Factor* factor) {
    if (factor == nullptr) {
        throw std::invalid_argument("Problem::add_factor: null factor");
    }
    for (const Variable* v : factor->variables()) {
        if (offsets_.count(v) == 0) {
            throw std::invalid_argument(
                "Problem::add_factor: factor references unregistered variable");
        }
    }
    factors_.push_back(factor);
}

double Problem::cost() const {
    double c = 0.0;
    for (const Factor* f : factors_) {
        c += 0.5 * f->residual().squaredNorm();
    }
    return c;
}

void Problem::build_normal_equations(Eigen::MatrixXd& H, Eigen::VectorXd& b) const {
    H.setZero(total_dim_, total_dim_);
    b.setZero(total_dim_);

    for (const Factor* f : factors_) {
        const Eigen::VectorXd              r    = f->residual();
        const std::vector<Eigen::MatrixXd> Js   = f->jacobians();
        const std::vector<Variable*>&      vars = f->variables();

        for (std::size_t i = 0; i < vars.size(); ++i) {
            const int oi = offsets_.at(vars[i]);
            const int ni = vars[i]->tangent_dim();
            b.segment(oi, ni) += Js[i].transpose() * r;
            for (std::size_t j = i; j < vars.size(); ++j) {
                const int             oj    = offsets_.at(vars[j]);
                const int             nj    = vars[j]->tangent_dim();
                const Eigen::MatrixXd block = Js[i].transpose() * Js[j];
                H.block(oi, oj, ni, nj) += block;
                if (i != j) {
                    H.block(oj, oi, nj, ni) += block.transpose();
                }
            }
        }
    }
}

void Problem::apply_step(const Eigen::VectorXd& delta) {
    if (delta.size() != total_dim_) {
        throw std::invalid_argument("Problem::apply_step: delta has wrong size");
    }
    for (Variable* v : variables_) {
        v->retract(delta.segment(offsets_.at(v), v->tangent_dim()));
    }
}

int Problem::tangent_offset(const Variable* variable) const {
    const auto it = offsets_.find(variable);
    if (it == offsets_.end()) {
        throw std::invalid_argument("Problem::tangent_offset: unregistered variable");
    }
    return it->second;
}

}  // namespace slam_core::optim
