// Demo: Levenberg-Marquardt rigid alignment, exported per iteration for
// visualization.
//
// Usage:
//   demo_lm_point_alignment <iterations_csv> <points_csv>
//
// Generates a deterministic synthetic problem: 40 points in [-1,1]^3,
// observed as q_A = R_gt * p_B + t_gt + noise, optimized from a heavily
// perturbed initial pose.
//
// iterations_csv: iteration, cost, lambda, step_norm, accepted,
//                 r00..r22 (row-major R_A_B), t_x, t_y, t_z
//                 Row 0 is the initial state before any step.
// points_csv:     p_x, p_y, p_z, q_x, q_y, q_z   (model point, observation)

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <Eigen/Core>

#include "slam_core/factors/point_alignment_factor.hpp"
#include "slam_core/geometry/so3.hpp"
#include "slam_core/optim/levenberg_marquardt.hpp"
#include "slam_core/optim/problem.hpp"

namespace {

// Raw mt19937 draws scaled to [-1, 1]; avoids distribution implementations
// that differ across standard libraries.
class Uniform {
public:
    explicit Uniform(std::uint32_t seed) : rng_(seed) {}
    double operator()() { return 2.0 * (static_cast<double>(rng_()) / 4294967295.0) - 1.0; }
    Eigen::Vector3d vec3() { return {(*this)(), (*this)(), (*this)()}; }

private:
    std::mt19937 rng_;
};

void write_pose_row(std::ofstream&         out,
                    int                    iteration,
                    double                 cost,
                    double                 lambda,
                    double                 step_norm,
                    bool                   accepted,
                    const Eigen::Matrix3d& R,
                    const Eigen::Vector3d& t) {
    out << iteration << ',' << cost << ',' << lambda << ',' << step_norm << ','
        << (accepted ? 1 : 0);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) out << ',' << R(r, c);
    out << ',' << t.x() << ',' << t.y() << ',' << t.z() << '\n';
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: demo_lm_point_alignment <iterations_csv> <points_csv>\n";
        return EXIT_FAILURE;
    }

    // Ground-truth transform T_A_B and synthetic correspondences.
    const Eigen::Matrix3d R_gt = slam_core::geometry::exp_so3({0.3, -0.2, 0.5});
    const Eigen::Vector3d t_gt(1.0, -0.5, 0.8);

    Uniform                      uniform(42);
    std::vector<Eigen::Vector3d> points_B, points_A;
    for (int i = 0; i < 40; ++i) {
        const Eigen::Vector3d p     = uniform.vec3();
        const Eigen::Vector3d noise = 0.01 * uniform.vec3();  // ~1 cm observation noise
        points_B.push_back(p);
        points_A.push_back(R_gt * p + t_gt + noise);
    }

    // Heavily perturbed initial pose (~63 deg rotation error, ~1.5 m offset).
    slam_core::optim::So3Variable    rot(R_gt * slam_core::geometry::exp_so3({0.6, -0.5, 0.7}));
    slam_core::optim::VectorVariable trans(t_gt + Eigen::Vector3d(1.0, 0.8, -0.9));

    std::vector<slam_core::factors::PointAlignmentFactor> factors;
    factors.reserve(points_B.size());
    for (std::size_t i = 0; i < points_B.size(); ++i) {
        factors.emplace_back(&rot, &trans, points_B[i], points_A[i]);
    }

    slam_core::optim::Problem problem;
    problem.add_variable(&rot);
    problem.add_variable(&trans);
    for (auto& f : factors) problem.add_factor(&f);

    for (const char* path : {argv[1], argv[2]}) {
        const std::filesystem::path p(path);
        if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
    }

    std::ofstream iter_out(argv[1]);
    std::ofstream points_out(argv[2]);
    if (!iter_out.is_open() || !points_out.is_open()) {
        std::cerr << "Error: cannot open output files\n";
        return EXIT_FAILURE;
    }
    iter_out << std::setprecision(12) << std::fixed;
    points_out << std::setprecision(12) << std::fixed;

    points_out << "p_x,p_y,p_z,q_x,q_y,q_z\n";
    for (std::size_t i = 0; i < points_B.size(); ++i) {
        points_out << points_B[i].x() << ',' << points_B[i].y() << ',' << points_B[i].z() << ','
                   << points_A[i].x() << ',' << points_A[i].y() << ',' << points_A[i].z() << '\n';
    }

    iter_out << "iteration,cost,lambda,step_norm,accepted,"
                "r00,r01,r02,r10,r11,r12,r20,r21,r22,t_x,t_y,t_z\n";
    write_pose_row(iter_out, -1, problem.cost(), 0.0, 0.0, true, rot.R(), trans.vec());

    slam_core::optim::LmOptions options;
    options.iteration_callback = [&](const slam_core::optim::LmIterationSummary& s) {
        write_pose_row(iter_out, s.iteration, s.cost, s.lambda, s.step_norm, s.step_accepted,
                       rot.R(), trans.vec());
    };

    const slam_core::optim::LmResult result = slam_core::optim::optimize(problem, options);

    const double rot_err_deg =
        slam_core::geometry::log_so3(R_gt.transpose() * rot.R()).norm() * 180.0 / M_PI;
    std::cerr << "demo_lm_point_alignment:\n"
              << "  converged:      " << (result.converged ? "yes" : "no") << " (" << result.message
              << ")\n"
              << "  iterations:     " << result.iterations << '\n'
              << "  cost:           " << result.initial_cost << " -> " << result.final_cost << '\n'
              << "  rotation error: " << rot_err_deg << " deg\n"
              << "  translation error: " << (trans.vec() - t_gt).norm() << " m\n"
              << "  wrote " << argv[1] << ", " << argv[2] << '\n';

    return result.converged ? EXIT_SUCCESS : EXIT_FAILURE;
}
