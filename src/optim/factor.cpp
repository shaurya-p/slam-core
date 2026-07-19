#include "slam_core/optim/factor.hpp"

#include <stdexcept>
#include <utility>

namespace slam_core::optim {

Factor::Factor(std::vector<Variable*> variables) : variables_(std::move(variables)) {
    if (variables_.empty()) {
        throw std::invalid_argument("Factor: no variables");
    }
    for (const Variable* v : variables_) {
        if (v == nullptr) {
            throw std::invalid_argument("Factor: null variable");
        }
    }
}

}  // namespace slam_core::optim
