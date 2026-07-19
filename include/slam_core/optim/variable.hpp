#pragma once

#include <Eigen/Core>

namespace slam_core::optim {

// One variable block of the optimization state.
//
// The state x lives on a manifold; the optimizer works with tangent-space
// increments delta ∈ R^tangent_dim() applied via retract: x ← x ⊞ delta.
// value()/set_value() serialize the ambient state (for save/restore during
// step rejection and for numeric differentiation); they are exact inverses.
class Variable {
public:
    virtual ~Variable() = default;

    virtual int tangent_dim() const = 0;

    // Applies a tangent-space increment to the state.
    // Throws std::invalid_argument if delta has the wrong size or is
    // non-finite.
    virtual void retract(const Eigen::VectorXd& delta) = 0;

    // Ambient-state serialization. set_value(value()) is a no-op.
    virtual Eigen::VectorXd value() const                           = 0;
    virtual void            set_value(const Eigen::VectorXd& value) = 0;
};

// SO(3) block storing a rotation matrix, e.g. R_W_B.
// Right (local) perturbation: R ← R * exp_so3(delta), delta ∈ R³ (rad).
class So3Variable final : public Variable {
public:
    explicit So3Variable(const Eigen::Matrix3d& R = Eigen::Matrix3d::Identity());

    int  tangent_dim() const override { return 3; }
    void retract(const Eigen::VectorXd& delta) override;

    Eigen::VectorXd value() const override;  // 9 entries, row-major
    void            set_value(const Eigen::VectorXd& value) override;

    const Eigen::Matrix3d& R() const { return R_; }

private:
    Eigen::Matrix3d R_;
};

// Euclidean block with additive retract: v ← v + delta.
class VectorVariable final : public Variable {
public:
    explicit VectorVariable(Eigen::VectorXd v);

    int  tangent_dim() const override { return static_cast<int>(v_.size()); }
    void retract(const Eigen::VectorXd& delta) override;

    Eigen::VectorXd value() const override { return v_; }
    void            set_value(const Eigen::VectorXd& value) override;

    const Eigen::VectorXd& vec() const { return v_; }

private:
    Eigen::VectorXd v_;
};

}  // namespace slam_core::optim
