#include "slam_core/io/euroc_csv.hpp"

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "slam_core/geometry/so3.hpp"

namespace {

using slam_core::io::EurocGtSample;
using slam_core::io::nearest_gt;
using slam_core::io::read_euroc_gt_csv;
using slam_core::io::read_euroc_imu_csv;

// Writes content to a unique file under the gtest temp dir and returns its path.
std::filesystem::path write_temp_csv(const std::string& name, const std::string& content) {
    const std::filesystem::path path = std::filesystem::path(testing::TempDir()) / name;
    std::ofstream               f(path);
    f << content;
    return path;
}

const char* kImuHeader =
    "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z "
    "[rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n";

const char* kGtHeader =
    "#timestamp,p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w [],q_RS_x "
    "[],q_RS_y [],q_RS_z [],v_RS_R_x [m s^-1],v_RS_R_y [m s^-1],v_RS_R_z "
    "[m s^-1]\n";

TEST(ReadEurocImuCsv, ParsesTimestampGyroAccel) {
    const auto path =
        write_temp_csv("imu_basic.csv", std::string(kImuHeader) +
                                            "1403636579758555392,-0.1,0.2,0.3,8.0,-0.5,-2.0\n"
                                            "1403636579763555584,0.4,-0.5,0.6,8.1,-0.6,-2.1\n");

    const auto samples = read_euroc_imu_csv(path);
    ASSERT_EQ(samples.size(), 2u);
    EXPECT_DOUBLE_EQ(samples[0].timestamp_s, 1403636579758555392.0 * 1e-9);
    EXPECT_DOUBLE_EQ(samples[0].gyro_radps.x(), -0.1);
    EXPECT_DOUBLE_EQ(samples[0].gyro_radps.y(), 0.2);
    EXPECT_DOUBLE_EQ(samples[0].gyro_radps.z(), 0.3);
    EXPECT_DOUBLE_EQ(samples[0].accel_mps2.x(), 8.0);
    EXPECT_DOUBLE_EQ(samples[0].accel_mps2.y(), -0.5);
    EXPECT_DOUBLE_EQ(samples[0].accel_mps2.z(), -2.0);
    EXPECT_LT(samples[0].timestamp_s, samples[1].timestamp_s);
}

TEST(ReadEurocImuCsv, SkipsMalformedBlankAndCommentRows) {
    const auto path =
        write_temp_csv("imu_malformed.csv", std::string(kImuHeader) +
                                                "1000000000,0.1,0.2,0.3,1.0,2.0,3.0\n"
                                                "\n"
                                                "# a comment row\n"
                                                "2000000000,0.1,not_a_number,0.3,1.0,2.0,3.0\n"
                                                "3000000000,0.1,0.2\n"
                                                "4000000000,0.1,0.2,0.3,1.0,2.0,3.0\n");

    int        skipped = -1;
    const auto samples = read_euroc_imu_csv(path, &skipped);
    ASSERT_EQ(samples.size(), 2u);
    EXPECT_EQ(skipped, 2);  // malformed value + short row; blank/comment not counted
    EXPECT_DOUBLE_EQ(samples[0].timestamp_s, 1.0);
    EXPECT_DOUBLE_EQ(samples[1].timestamp_s, 4.0);
}

TEST(ReadEurocImuCsv, MissingFileThrowsRuntimeError) {
    EXPECT_THROW(read_euroc_imu_csv("/nonexistent/imu.csv"), std::runtime_error);
}

TEST(ReadEurocImuCsv, EmptyFileThrowsInvalidArgument) {
    const auto path = write_temp_csv("imu_empty.csv", "");
    EXPECT_THROW(read_euroc_imu_csv(path), std::invalid_argument);
}

TEST(ReadEurocImuCsv, HeaderOnlyThrowsInvalidArgument) {
    const auto path = write_temp_csv("imu_header_only.csv", kImuHeader);
    EXPECT_THROW(read_euroc_imu_csv(path), std::invalid_argument);
}

TEST(ReadEurocGtCsv, ParsesPositionQuaternionVelocity) {
    // Quaternion deliberately unnormalized (norm 2): [1, 1, 1, 1].
    const auto path = write_temp_csv(
        "gt_basic.csv", std::string(kGtHeader) +
                            "1403636580838555648,4.6,-1.8,0.8,1.0,1.0,1.0,1.0,0.01,-0.02,0.03\n");

    const auto samples = read_euroc_gt_csv(path);
    ASSERT_EQ(samples.size(), 1u);
    const EurocGtSample& s = samples[0];
    EXPECT_DOUBLE_EQ(s.timestamp_s, 1403636580838555648.0 * 1e-9);
    EXPECT_DOUBLE_EQ(s.p_W_B.x(), 4.6);
    EXPECT_DOUBLE_EQ(s.p_W_B.y(), -1.8);
    EXPECT_DOUBLE_EQ(s.p_W_B.z(), 0.8);
    EXPECT_DOUBLE_EQ(s.v_W_B.x(), 0.01);
    EXPECT_DOUBLE_EQ(s.v_W_B.y(), -0.02);
    EXPECT_DOUBLE_EQ(s.v_W_B.z(), 0.03);
    // Normalized on parse: each component 1/2.
    EXPECT_DOUBLE_EQ(s.q_W_B.w(), 0.5);
    EXPECT_DOUBLE_EQ(s.q_W_B.x(), 0.5);
    EXPECT_DOUBLE_EQ(s.q_W_B.y(), 0.5);
    EXPECT_DOUBLE_EQ(s.q_W_B.z(), 0.5);
    EXPECT_TRUE(slam_core::geometry::is_valid_rotation(s.R_W_B()));
}

TEST(ReadEurocGtCsv, SkipsShortAndZeroQuaternionRows) {
    const auto path = write_temp_csv("gt_malformed.csv",
                                     std::string(kGtHeader) +
                                         "1000000000,0,0,0,1,0,0,0,0,0,0\n"
                                         "2000000000,0,0,0,1,0,0\n"          // short row
                                         "3000000000,0,0,0,0,0,0,0,0,0,0\n"  // zero quaternion
                                         "4000000000,1,2,3,0,0,0,1,0,0,0\n");

    int        skipped = -1;
    const auto samples = read_euroc_gt_csv(path, &skipped);
    ASSERT_EQ(samples.size(), 2u);
    EXPECT_EQ(skipped, 2);
    EXPECT_DOUBLE_EQ(samples[0].timestamp_s, 1.0);
    EXPECT_DOUBLE_EQ(samples[1].timestamp_s, 4.0);
}

TEST(ReadEurocGtCsv, MissingFileThrowsRuntimeError) {
    EXPECT_THROW(read_euroc_gt_csv("/nonexistent/gt.csv"), std::runtime_error);
}

TEST(ReadEurocGtCsv, EmptyFileThrowsInvalidArgument) {
    const auto path = write_temp_csv("gt_empty.csv", "");
    EXPECT_THROW(read_euroc_gt_csv(path), std::invalid_argument);
}

std::vector<EurocGtSample> make_gt_samples(const std::vector<double>& times) {
    std::vector<EurocGtSample> samples;
    for (double t : times) {
        EurocGtSample s;
        s.timestamp_s = t;
        s.p_W_B       = Eigen::Vector3d::Zero();
        s.q_W_B       = Eigen::Quaterniond::Identity();
        s.v_W_B       = Eigen::Vector3d::Zero();
        samples.push_back(s);
    }
    return samples;
}

TEST(NearestGt, ClampsBeforeFirstAndAfterLast) {
    const auto samples = make_gt_samples({1.0, 2.0, 3.0});
    EXPECT_DOUBLE_EQ(nearest_gt(0.0, samples).timestamp_s, 1.0);
    EXPECT_DOUBLE_EQ(nearest_gt(10.0, samples).timestamp_s, 3.0);
}

TEST(NearestGt, PicksCloserByAbsoluteDifference) {
    const auto samples = make_gt_samples({1.0, 2.0, 3.0});
    EXPECT_DOUBLE_EQ(nearest_gt(1.4, samples).timestamp_s, 1.0);
    EXPECT_DOUBLE_EQ(nearest_gt(1.6, samples).timestamp_s, 2.0);
    EXPECT_DOUBLE_EQ(nearest_gt(2.0, samples).timestamp_s, 2.0);
}

TEST(NearestGt, TieResolvesToEarlierSample) {
    const auto samples = make_gt_samples({1.0, 3.0});
    EXPECT_DOUBLE_EQ(nearest_gt(2.0, samples).timestamp_s, 1.0);
}

TEST(NearestGt, EmptyThrowsInvalidArgument) {
    const std::vector<EurocGtSample> empty;
    EXPECT_THROW(nearest_gt(1.0, empty), std::invalid_argument);
}

}  // namespace

// --- FG-5: camera sensor.yaml calibration ---

#include "slam_core/io/euroc_camera.hpp"

namespace {

const char* kCamYaml = R"(# General sensor definitions.
sensor_type: camera

T_BS:
  cols: 4
  rows: 4
  data: [0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
         0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
        -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
         0.0, 0.0, 0.0, 1.0]

rate_hz: 20
resolution: [752, 480]
camera_model: pinhole
intrinsics: [458.654, 457.296, 367.215, 248.375] #fu, fv, cu, cv
distortion_model: radial-tangential
distortion_coefficients: [-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05]
)";

}  // namespace

TEST(ReadEurocCameraYaml, ParsesEurocCam0Format) {
    const auto path  = write_temp_csv("cam_sensor.yaml", kCamYaml);
    const auto calib = slam_core::io::read_euroc_camera_yaml(path);

    EXPECT_TRUE(slam_core::geometry::is_valid_rotation(calib.R_B_C, 1e-4));
    EXPECT_NEAR(calib.R_B_C(0, 0), 0.0148655429818, 1e-12);
    EXPECT_NEAR(calib.R_B_C(2, 2), 0.999660727178, 1e-12);
    EXPECT_NEAR(calib.t_B_C.x(), -0.0216401454975, 1e-12);
    EXPECT_NEAR(calib.t_B_C.z(), 0.00981073058949, 1e-12);
    EXPECT_DOUBLE_EQ(calib.fx, 458.654);
    EXPECT_DOUBLE_EQ(calib.cy, 248.375);
    EXPECT_EQ(calib.width, 752);
    EXPECT_EQ(calib.height, 480);
    EXPECT_DOUBLE_EQ(calib.rate_hz, 20.0);
    EXPECT_NEAR(calib.distortion_radtan(0), -0.28340811, 1e-12);
    EXPECT_NEAR(calib.distortion_radtan(3), 1.76187114e-05, 1e-15);
}

TEST(ReadEurocCameraYaml, MissingFileAndMissingKeyThrow) {
    EXPECT_THROW(slam_core::io::read_euroc_camera_yaml("/nonexistent/sensor.yaml"),
                 std::runtime_error);
    const auto path = write_temp_csv("cam_bad.yaml", "sensor_type: camera\nrate_hz: 20\n");
    EXPECT_THROW(slam_core::io::read_euroc_camera_yaml(path), std::invalid_argument);
}
