#pragma once

#include <filesystem>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "slam_core/imu/imu_measurement.hpp"

namespace slam_core::io {

// EuRoC IMU CSV: timestamp_ns, wx, wy, wz, ax, ay, az
// Timestamps converted ns -> s. Skips the header line and blank / '#' lines.
// Rows with fewer than 7 parseable numeric columns are dropped and counted
// in *skipped_rows (if non-null).
//
// Throws std::runtime_error if the file cannot be opened.
// Throws std::invalid_argument if the file is empty or no row parses.
std::vector<imu::ImuMeasurement> read_euroc_imu_csv(
    const std::filesystem::path& path,
    int*                         skipped_rows = nullptr);

// One EuRoC ground-truth sample.
// EuRoC GT CSV: timestamp_ns, p_x, p_y, p_z, q_w, q_x, q_y, q_z, v_x, v_y, v_z, ...
// Frame R (world) == W under the project convention; q is q_W_B, Hamilton, w-first.
struct EurocGtSample {
    double             timestamp_s;  // seconds (ns * 1e-9)
    Eigen::Vector3d    p_W_B;        // body position in world frame (m)
    Eigen::Quaterniond q_W_B;        // normalized on parse
    Eigen::Vector3d    v_W_B;        // body velocity in world frame (m/s)

    // Rotation mapping body frame B into world frame W.
    Eigen::Matrix3d R_W_B() const { return q_W_B.toRotationMatrix(); }
};

// Reads EuRoC ground truth. Skips the header line and blank / '#' lines.
// Rows with fewer than 11 parseable numeric columns, or a quaternion norm
// below 1e-10, are dropped and counted in *skipped_rows (if non-null).
//
// Throws std::runtime_error if the file cannot be opened.
// Throws std::invalid_argument if the file is empty or no row parses.
std::vector<EurocGtSample> read_euroc_gt_csv(
    const std::filesystem::path& path,
    int*                         skipped_rows = nullptr);

// Sample with the nearest timestamp by absolute difference; ties resolve to
// the earlier sample. Requires samples sorted ascending by timestamp.
//
// Throws std::invalid_argument if samples is empty.
const EurocGtSample& nearest_gt(double                            timestamp_s,
                                const std::vector<EurocGtSample>& samples);

}  // namespace slam_core::io
