#include <gtest/gtest.h>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "slam_core/geometry/so3.hpp"
#include "slam_core/imu/preintegration.hpp"

using slam_core::geometry::exp_so3;
using slam_core::imu::ImuMeasurement;
using slam_core::imu::integrate;
using slam_core::imu::integrate_sequence;
using slam_core::imu::integrate_window;
using slam_core::imu::PreintegratedImu;

// --- helpers ---

static ImuMeasurement make_meas(double gx = 0.0,
                                double gy = 0.0,
                                double gz = 0.0,
                                double ax = 0.0,
                                double ay = 0.0,
                                double az = 0.0) {
    ImuMeasurement m;
    m.timestamp_s = 0.0;
    m.gyro_radps  = {gx, gy, gz};
    m.accel_mps2  = {ax, ay, az};
    return m;
}

// --- 1. default construction and reset ---

TEST(Preintegration, DefaultConstructed) {
    PreintegratedImu preint;
    EXPECT_TRUE(preint.delta_R.isApprox(Eigen::Matrix3d::Identity(), 1e-12));
    EXPECT_TRUE(preint.delta_v.isApprox(Eigen::Vector3d::Zero(), 1e-12));
    EXPECT_TRUE(preint.delta_p.isApprox(Eigen::Vector3d::Zero(), 1e-12));
    EXPECT_DOUBLE_EQ(preint.delta_t_s, 0.0);
}

TEST(Preintegration, ResetRestoresInitialState) {
    PreintegratedImu preint;
    integrate(preint, make_meas(0.1, 0.2, 0.3, 1.0, 2.0, 3.0), Eigen::Vector3d::Zero(),
              Eigen::Vector3d::Zero(), 0.05);

    preint.reset();

    EXPECT_TRUE(preint.delta_R.isApprox(Eigen::Matrix3d::Identity(), 1e-12));
    EXPECT_TRUE(preint.delta_v.isApprox(Eigen::Vector3d::Zero(), 1e-12));
    EXPECT_TRUE(preint.delta_p.isApprox(Eigen::Vector3d::Zero(), 1e-12));
    EXPECT_DOUBLE_EQ(preint.delta_t_s, 0.0);
}

// --- 2. zero net motion: delta_t_s accumulates, R/v/p stay at identity/zero ---

TEST(Preintegration, ZeroNetMotionAccumulatesTime) {
    PreintegratedImu      preint;
    const Eigen::Vector3d zero  = Eigen::Vector3d::Zero();
    const double          dt    = 0.01;
    const int             steps = 100;

    for (int i = 0; i < steps; ++i) {
        integrate(preint, make_meas(), zero, zero, dt);
    }

    EXPECT_NEAR(preint.delta_t_s, steps * dt, 1e-9);
    EXPECT_TRUE(preint.delta_R.isApprox(Eigen::Matrix3d::Identity(), 1e-9));
    EXPECT_TRUE(preint.delta_v.isApprox(Eigen::Vector3d::Zero(), 1e-9));
    EXPECT_TRUE(preint.delta_p.isApprox(Eigen::Vector3d::Zero(), 1e-9));
}

// --- 3. constant acceleration, no rotation ---
//
// gyro = 0, a_B = [2, 0, 0], dt = 0.5 s
// Expected: delta_v = a*dt = [1, 0, 0]
//           delta_p = 0.5*a*dt^2 = [0.25, 0, 0]

TEST(Preintegration, ConstantAcceleration) {
    PreintegratedImu      preint;
    const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
    const double          ax   = 2.0;
    const double          dt   = 0.5;

    integrate(preint, make_meas(0, 0, 0, ax, 0, 0), zero, zero, dt);

    EXPECT_TRUE(preint.delta_v.isApprox(Eigen::Vector3d(ax * dt, 0, 0), 1e-9));
    EXPECT_TRUE(preint.delta_p.isApprox(Eigen::Vector3d(0.5 * ax * dt * dt, 0, 0), 1e-9));
    EXPECT_TRUE(preint.delta_R.isApprox(Eigen::Matrix3d::Identity(), 1e-9));
}

// --- 4. constant yaw rate, no acceleration ---
//
// omega_z = pi/2 rad/s, dt = 1.0 s
// Expected: delta_R = exp_so3([0, 0, pi/2])
//           delta_v = 0, delta_p = 0

TEST(Preintegration, ConstantYawRate) {
    PreintegratedImu      preint;
    const Eigen::Vector3d zero    = Eigen::Vector3d::Zero();
    const double          omega_z = M_PI / 2.0;
    const double          dt      = 1.0;

    integrate(preint, make_meas(0, 0, omega_z), zero, zero, dt);

    const Eigen::Matrix3d expected_R = exp_so3(Eigen::Vector3d(0, 0, omega_z * dt));
    EXPECT_TRUE(preint.delta_R.isApprox(expected_R, 1e-9));
    EXPECT_TRUE(preint.delta_v.isApprox(Eigen::Vector3d::Zero(), 1e-9));
    EXPECT_TRUE(preint.delta_p.isApprox(Eigen::Vector3d::Zero(), 1e-9));
}

// --- 5. bias correction: measurement == bias -> zero net motion ---

TEST(Preintegration, BiasEqualsRawMeasurementProducesZeroMotion) {
    PreintegratedImu      preint;
    const Eigen::Vector3d gyro_bias{0.1, 0.2, 0.3};
    const Eigen::Vector3d accel_bias{1.0, 2.0, 3.0};
    const double          dt = 0.05;

    // measurement equals bias -> omega = 0, a = 0
    ImuMeasurement meas = make_meas(0.1, 0.2, 0.3, 1.0, 2.0, 3.0);
    for (int i = 0; i < 20; ++i) {
        integrate(preint, meas, gyro_bias, accel_bias, dt);
    }

    EXPECT_TRUE(preint.delta_R.isApprox(Eigen::Matrix3d::Identity(), 1e-9));
    EXPECT_TRUE(preint.delta_v.isApprox(Eigen::Vector3d::Zero(), 1e-9));
    EXPECT_TRUE(preint.delta_p.isApprox(Eigen::Vector3d::Zero(), 1e-9));
}

// --- 7. rotated-frame acceleration ---
//
// Validates that body-frame acceleration is rotated by the current delta_R
// before accumulating into delta_v and delta_p.

TEST(Preintegration, BodyAccelRotatedByDeltaR_YawPlusZ) {
    // Step 1: 90-deg yaw about +Z, zero accel
    //   delta_R = R_z(pi/2), delta_v = 0, delta_p = 0
    // Step 2: zero gyro, accel = [2, 0, 0] (body +X), dt = 0.5 s
    //   delta_R * [2,0,0] = R_z(pi/2)*[2,0,0] = [0, 2, 0]
    //   delta_v = [0, 1, 0]   (= [0,2,0] * 0.5)
    //   delta_p = [0, 0.25, 0] (= 0.5 * [0,2,0] * 0.25)
    PreintegratedImu      preint;
    const Eigen::Vector3d zero = Eigen::Vector3d::Zero();

    integrate(preint, make_meas(0, 0, M_PI / 2.0), zero, zero, 1.0);
    integrate(preint, make_meas(0, 0, 0, 2.0, 0, 0), zero, zero, 0.5);

    EXPECT_TRUE(preint.delta_v.isApprox(Eigen::Vector3d(0.0, 1.0, 0.0), 1e-9));
    EXPECT_TRUE(preint.delta_p.isApprox(Eigen::Vector3d(0.0, 0.25, 0.0), 1e-9));
}

TEST(Preintegration, BodyAccelRotatedByDeltaR_PitchPlusY) {
    // Step 1: 90-deg pitch about +Y, zero accel
    //   delta_R = R_y(pi/2), delta_v = 0, delta_p = 0
    // Step 2: zero gyro, accel = [0, 0, 3] (body +Z), dt = 0.4 s
    //   delta_R * [0,0,3] = R_y(pi/2)*[0,0,3] = [3, 0, 0]
    //   delta_v = [1.2, 0, 0]  (= [3,0,0] * 0.4)
    //   delta_p = [0.24, 0, 0] (= 0.5 * [3,0,0] * 0.16)
    PreintegratedImu      preint;
    const Eigen::Vector3d zero = Eigen::Vector3d::Zero();

    integrate(preint, make_meas(0, M_PI / 2.0, 0), zero, zero, 1.0);
    integrate(preint, make_meas(0, 0, 0, 0, 0, 3.0), zero, zero, 0.4);

    EXPECT_TRUE(preint.delta_v.isApprox(Eigen::Vector3d(1.2, 0.0, 0.0), 1e-9));
    EXPECT_TRUE(preint.delta_p.isApprox(Eigen::Vector3d(0.24, 0.0, 0.0), 1e-9));
}

// --- 6. invalid inputs throw ---

TEST(Preintegration, InvalidDtThrows) {
    PreintegratedImu      preint;
    const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
    constexpr double      kNaN = std::numeric_limits<double>::quiet_NaN();
    constexpr double      kInf = std::numeric_limits<double>::infinity();

    EXPECT_THROW(integrate(preint, make_meas(), zero, zero, 0.0), std::invalid_argument);
    EXPECT_THROW(integrate(preint, make_meas(), zero, zero, -0.01), std::invalid_argument);
    EXPECT_THROW(integrate(preint, make_meas(), zero, zero, kNaN), std::invalid_argument);
    EXPECT_THROW(integrate(preint, make_meas(), zero, zero, kInf), std::invalid_argument);
}

TEST(Preintegration, NonFiniteMeasurementThrows) {
    PreintegratedImu      preint;
    const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
    constexpr double      kNaN = std::numeric_limits<double>::quiet_NaN();

    {
        ImuMeasurement m = make_meas();
        m.gyro_radps.z() = kNaN;
        EXPECT_THROW(integrate(preint, m, zero, zero, 0.01), std::invalid_argument);
    }
    {
        ImuMeasurement m = make_meas();
        m.accel_mps2.x() = kNaN;
        EXPECT_THROW(integrate(preint, m, zero, zero, 0.01), std::invalid_argument);
    }
    {
        const Eigen::Vector3d bad_bias{0.0, kNaN, 0.0};
        EXPECT_THROW(integrate(preint, make_meas(), bad_bias, zero, 0.01), std::invalid_argument);
    }
}

// --- integrate_sequence: invalid input ---

static ImuMeasurement make_timed_meas(double t,
                                      double gx = 0.0,
                                      double gy = 0.0,
                                      double gz = 0.0,
                                      double ax = 0.0,
                                      double ay = 0.0,
                                      double az = 0.0) {
    ImuMeasurement m;
    m.timestamp_s = t;
    m.gyro_radps  = {gx, gy, gz};
    m.accel_mps2  = {ax, ay, az};
    return m;
}

TEST(IntegrateSequence, TooFewMeasurementsThrows) {
    const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
    EXPECT_THROW(integrate_sequence({}, zero, zero), std::invalid_argument);
    EXPECT_THROW(integrate_sequence({make_timed_meas(0.0)}, zero, zero), std::invalid_argument);
}

TEST(IntegrateSequence, NonFiniteMeasurementThrows) {
    const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
    constexpr double      kNaN = std::numeric_limits<double>::quiet_NaN();

    ImuMeasurement bad = make_timed_meas(0.1);
    bad.gyro_radps.z() = kNaN;
    EXPECT_THROW(integrate_sequence({make_timed_meas(0.0), bad}, zero, zero),
                 std::invalid_argument);
}

TEST(IntegrateSequence, NonFiniteBiasThrows) {
    const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
    constexpr double      kNaN = std::numeric_limits<double>::quiet_NaN();

    const std::vector<ImuMeasurement> seq = {make_timed_meas(0.0), make_timed_meas(0.1)};
    EXPECT_THROW(integrate_sequence(seq, Eigen::Vector3d{0.0, kNaN, 0.0}, zero),
                 std::invalid_argument);
    EXPECT_THROW(integrate_sequence(seq, zero, Eigen::Vector3d{kNaN, 0.0, 0.0}),
                 std::invalid_argument);
}

TEST(IntegrateSequence, NonIncreasingTimestampThrows) {
    const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
    // Equal timestamps -> dt = 0
    EXPECT_THROW(integrate_sequence({make_timed_meas(0.1), make_timed_meas(0.1)}, zero, zero),
                 std::invalid_argument);
    // Decreasing timestamps -> dt < 0
    EXPECT_THROW(integrate_sequence({make_timed_meas(0.2), make_timed_meas(0.1)}, zero, zero),
                 std::invalid_argument);
}

// --- integrate_sequence: correct accumulation ---

// 3 samples, uniform dt=0.5 s, constant x-accel=2 m/s², zero gyro.
// 2 steps -> delta_v = [2, 0, 0]*0.5 + [2, 0, 0]*0.5 = [2, 0, 0]
//            delta_p = (0 + 0.5*2*0.25) + (1*0.5 + 0.5*2*0.25) = 0.25 + 0.75 = 1.0
TEST(IntegrateSequence, ConstantAccelMatchesManual) {
    const Eigen::Vector3d             zero = Eigen::Vector3d::Zero();
    const double                      ax   = 2.0;
    const double                      dt   = 0.5;
    const std::vector<ImuMeasurement> seq  = {
        make_timed_meas(0.0, 0, 0, 0, ax, 0, 0),
        make_timed_meas(dt, 0, 0, 0, ax, 0, 0),
        make_timed_meas(2 * dt, 0, 0, 0, ax, 0, 0),
    };

    PreintegratedImu manual;
    integrate(manual, seq[0], zero, zero, dt);
    integrate(manual, seq[1], zero, zero, dt);

    const PreintegratedImu result = integrate_sequence(seq, zero, zero);

    EXPECT_TRUE(result.delta_v.isApprox(manual.delta_v, 1e-9));
    EXPECT_TRUE(result.delta_p.isApprox(manual.delta_p, 1e-9));
    EXPECT_TRUE(result.delta_R.isApprox(manual.delta_R, 1e-9));
    EXPECT_NEAR(result.delta_t_s, manual.delta_t_s, 1e-9);
}

// 3 samples, uniform dt=0.5 s, constant yaw rate = pi/2 rad/s, zero accel.
TEST(IntegrateSequence, ConstantGyroMatchesManual) {
    const Eigen::Vector3d             zero    = Eigen::Vector3d::Zero();
    const double                      omega_z = M_PI / 2.0;
    const double                      dt      = 0.5;
    const std::vector<ImuMeasurement> seq     = {
        make_timed_meas(0.0, 0, 0, omega_z),
        make_timed_meas(dt, 0, 0, omega_z),
        make_timed_meas(2 * dt, 0, 0, omega_z),
    };

    PreintegratedImu manual;
    integrate(manual, seq[0], zero, zero, dt);
    integrate(manual, seq[1], zero, zero, dt);

    const PreintegratedImu result = integrate_sequence(seq, zero, zero);

    EXPECT_TRUE(result.delta_R.isApprox(manual.delta_R, 1e-9));
    EXPECT_TRUE(result.delta_v.isApprox(Eigen::Vector3d::Zero(), 1e-9));
    EXPECT_TRUE(result.delta_p.isApprox(Eigen::Vector3d::Zero(), 1e-9));
}

// measurement == bias over 5 samples -> zero net motion.
TEST(IntegrateSequence, BiasCancellationProducesZeroMotion) {
    const Eigen::Vector3d gyro_bias{0.1, 0.2, 0.3};
    const Eigen::Vector3d accel_bias{1.0, 2.0, 3.0};

    std::vector<ImuMeasurement> seq;
    for (int i = 0; i < 5; ++i) {
        seq.push_back(make_timed_meas(i * 0.05, gyro_bias.x(), gyro_bias.y(), gyro_bias.z(),
                                      accel_bias.x(), accel_bias.y(), accel_bias.z()));
    }

    const PreintegratedImu result = integrate_sequence(seq, gyro_bias, accel_bias);

    EXPECT_TRUE(result.delta_R.isApprox(Eigen::Matrix3d::Identity(), 1e-9));
    EXPECT_TRUE(result.delta_v.isApprox(Eigen::Vector3d::Zero(), 1e-9));
    EXPECT_TRUE(result.delta_p.isApprox(Eigen::Vector3d::Zero(), 1e-9));
}

// Non-uniform dt, mixed gyro+accel: sequence result must equal manual stepwise.
TEST(IntegrateSequence, MatchesManualStepwiseNonUniformDt) {
    const Eigen::Vector3d gyro_bias{0.01, -0.02, 0.03};
    const Eigen::Vector3d accel_bias{0.1, 0.2, -0.1};

    const std::vector<ImuMeasurement> seq = {
        make_timed_meas(0.00, 0.1, 0.0, -0.05, 0.5, -0.3, 0.2),
        make_timed_meas(0.01, 0.0, 0.2, 0.1, -0.1, 0.4, -0.3),
        make_timed_meas(0.03, -0.05, 0.1, 0.0, 0.2, 0.1, 0.5),
        make_timed_meas(0.06, 0.2, -0.1, 0.05, -0.2, -0.1, 0.3),
    };

    PreintegratedImu manual;
    for (std::size_t i = 0; i + 1 < seq.size(); ++i) {
        const double dt = seq[i + 1].timestamp_s - seq[i].timestamp_s;
        integrate(manual, seq[i], gyro_bias, accel_bias, dt);
    }

    const PreintegratedImu result = integrate_sequence(seq, gyro_bias, accel_bias);

    EXPECT_TRUE(result.delta_R.isApprox(manual.delta_R, 1e-12));
    EXPECT_TRUE(result.delta_v.isApprox(manual.delta_v, 1e-12));
    EXPECT_TRUE(result.delta_p.isApprox(manual.delta_p, 1e-12));
    EXPECT_NEAR(result.delta_t_s, manual.delta_t_s, 1e-12);
}

// =============================================================================
// integrate_window tests
// =============================================================================

// Builds a uniform 5-measurement stream at t = 0.0, 0.1, 0.2, 0.3, 0.4.
static std::vector<ImuMeasurement> make_stream_5() {
    std::vector<ImuMeasurement> s;
    for (int i = 0; i < 5; ++i) {
        s.push_back(make_timed_meas(i * 0.1, 0.1, 0.0, -0.05, 0.5, -0.3, 0.2));
    }
    return s;
}

// --- invalid window arguments ---

TEST(IntegrateWindow, InvalidWindowTimestampsThrow) {
    const Eigen::Vector3d zero   = Eigen::Vector3d::Zero();
    const auto            stream = make_stream_5();
    constexpr double      kNaN   = std::numeric_limits<double>::quiet_NaN();
    constexpr double      kInf   = std::numeric_limits<double>::infinity();

    EXPECT_THROW(integrate_window(stream, kNaN, 0.3, zero, zero), std::invalid_argument);
    EXPECT_THROW(integrate_window(stream, 0.0, kNaN, zero, zero), std::invalid_argument);
    EXPECT_THROW(integrate_window(stream, kInf, 0.3, zero, zero), std::invalid_argument);
    EXPECT_THROW(integrate_window(stream, 0.0, kInf, zero, zero), std::invalid_argument);
    EXPECT_THROW(integrate_window(stream, 0.2, 0.2, zero, zero), std::invalid_argument);
    EXPECT_THROW(integrate_window(stream, 0.3, 0.1, zero, zero), std::invalid_argument);
}

TEST(IntegrateWindow, NonFiniteBiasThrows) {
    const Eigen::Vector3d zero   = Eigen::Vector3d::Zero();
    const auto            stream = make_stream_5();
    constexpr double      kNaN   = std::numeric_limits<double>::quiet_NaN();

    EXPECT_THROW(integrate_window(stream, 0.0, 0.4, Eigen::Vector3d{kNaN, 0.0, 0.0}, zero),
                 std::invalid_argument);
    EXPECT_THROW(integrate_window(stream, 0.0, 0.4, zero, Eigen::Vector3d{0.0, kNaN, 0.0}),
                 std::invalid_argument);
}

// --- invalid stream ---

TEST(IntegrateWindow, EmptyStreamThrows) {
    const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
    EXPECT_THROW(integrate_window({}, 0.0, 0.4, zero, zero), std::invalid_argument);
}

TEST(IntegrateWindow, StreamNonIncreasingTimestampsThrow) {
    const Eigen::Vector3d zero = Eigen::Vector3d::Zero();

    // Duplicate timestamp
    std::vector<ImuMeasurement> dup = {make_timed_meas(0.0), make_timed_meas(0.1),
                                       make_timed_meas(0.1), make_timed_meas(0.2)};
    EXPECT_THROW(integrate_window(dup, 0.0, 0.2, zero, zero), std::invalid_argument);

    // Decreasing timestamp
    std::vector<ImuMeasurement> dec = {make_timed_meas(0.0), make_timed_meas(0.2),
                                       make_timed_meas(0.1), make_timed_meas(0.3)};
    EXPECT_THROW(integrate_window(dec, 0.0, 0.3, zero, zero), std::invalid_argument);
}

// Non-finite measurement outside the requested window: full stream is validated,
// not just the selected window.
TEST(IntegrateWindow, NonFiniteMeasurementOutsideWindowThrows) {
    const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
    constexpr double      kNaN = std::numeric_limits<double>::quiet_NaN();

    std::vector<ImuMeasurement> stream = make_stream_5();
    stream[4].accel_mps2.x()           = kNaN;  // outside window [0.0, 0.3]

    EXPECT_THROW(integrate_window(stream, 0.0, 0.3, zero, zero), std::invalid_argument);
}

// --- window coverage ---

TEST(IntegrateWindow, WindowOutsideStreamRangeThrows) {
    const Eigen::Vector3d zero   = Eigen::Vector3d::Zero();
    const auto            stream = make_stream_5();  // t in [0.0, 0.4]

    // Entirely before
    EXPECT_THROW(integrate_window(stream, -0.3, -0.1, zero, zero), std::invalid_argument);
    // Entirely after
    EXPECT_THROW(integrate_window(stream, 0.5, 0.9, zero, zero), std::invalid_argument);
}

TEST(IntegrateWindow, TooFewMeasurementsInWindowThrows) {
    const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
    // Stream: t = 0.0, 0.1, 0.5, 0.6
    // Window [0.2, 0.4] contains no measurements -> fewer than 2
    std::vector<ImuMeasurement> stream = {make_timed_meas(0.0), make_timed_meas(0.1),
                                          make_timed_meas(0.5), make_timed_meas(0.6)};
    EXPECT_THROW(integrate_window(stream, 0.2, 0.4, zero, zero), std::invalid_argument);

    // Window [0.1, 0.49] contains exactly 1 measurement (t=0.1)
    EXPECT_THROW(integrate_window(stream, 0.1, 0.49, zero, zero), std::invalid_argument);
}

// --- correct accumulation ---

// Window [0.1, 0.3] selects measurements at t=0.1, 0.2, 0.3 from a 5-element
// stream. Result must equal integrate_sequence on that explicit slice.
TEST(IntegrateWindow, MatchesManualSlice) {
    const Eigen::Vector3d gyro_bias{0.01, -0.02, 0.03};
    const Eigen::Vector3d accel_bias{0.1, 0.2, -0.1};

    const std::vector<ImuMeasurement> stream = {
        make_timed_meas(0.00, 0.1, 0.0, -0.05, 0.5, -0.3, 0.2),
        make_timed_meas(0.10, 0.0, 0.2, 0.10, -0.1, 0.4, -0.3),
        make_timed_meas(0.20, -0.05, 0.1, 0.00, 0.2, 0.1, 0.5),
        make_timed_meas(0.30, 0.2, -0.1, 0.05, -0.2, -0.1, 0.3),
        make_timed_meas(0.40, 0.0, 0.0, 0.10, 0.3, 0.2, -0.1),
    };

    // Manual slice: indices 1..3
    const std::vector<ImuMeasurement> slice    = {stream[1], stream[2], stream[3]};
    const PreintegratedImu            expected = integrate_sequence(slice, gyro_bias, accel_bias);

    const PreintegratedImu result = integrate_window(stream, 0.10, 0.30, gyro_bias, accel_bias);

    EXPECT_TRUE(result.delta_R.isApprox(expected.delta_R, 1e-12));
    EXPECT_TRUE(result.delta_v.isApprox(expected.delta_v, 1e-12));
    EXPECT_TRUE(result.delta_p.isApprox(expected.delta_p, 1e-12));
    EXPECT_NEAR(result.delta_t_s, expected.delta_t_s, 1e-12);
}

// Large accel outside the window, zero motion inside: result must be zero.
// Confirms that out-of-window measurements are excluded.
TEST(IntegrateWindow, ExcludesOutsideMeasurements) {
    const Eigen::Vector3d zero = Eigen::Vector3d::Zero();

    std::vector<ImuMeasurement> stream = {
        make_timed_meas(0.0, 0, 0, 0, 100.0, 100.0, 100.0),  // outside: before window
        make_timed_meas(0.1),                                // inside: zero motion
        make_timed_meas(0.2),                                // inside: zero motion
        make_timed_meas(0.3),                                // inside: zero motion
        make_timed_meas(0.4, 0, 0, 0, 100.0, 100.0, 100.0),  // outside: after window
    };

    const PreintegratedImu result = integrate_window(stream, 0.1, 0.3, zero, zero);

    EXPECT_TRUE(result.delta_R.isApprox(Eigen::Matrix3d::Identity(), 1e-9));
    EXPECT_TRUE(result.delta_v.isApprox(Eigen::Vector3d::Zero(), 1e-9));
    EXPECT_TRUE(result.delta_p.isApprox(Eigen::Vector3d::Zero(), 1e-9));
    EXPECT_NEAR(result.delta_t_s, 0.2, 1e-9);
}

// Measurement == bias for all in-window samples -> zero net motion.
TEST(IntegrateWindow, BiasCancellationOverWindow) {
    const Eigen::Vector3d gyro_bias{0.1, 0.2, 0.3};
    const Eigen::Vector3d accel_bias{1.0, 2.0, 3.0};

    std::vector<ImuMeasurement> stream;
    for (int i = 0; i < 5; ++i) {
        stream.push_back(make_timed_meas(i * 0.05, gyro_bias.x(), gyro_bias.y(), gyro_bias.z(),
                                         accel_bias.x(), accel_bias.y(), accel_bias.z()));
    }

    // Window covers indices 1..3 (t = 0.05, 0.10, 0.15)
    const PreintegratedImu result = integrate_window(stream, 0.05, 0.15, gyro_bias, accel_bias);

    EXPECT_TRUE(result.delta_R.isApprox(Eigen::Matrix3d::Identity(), 1e-9));
    EXPECT_TRUE(result.delta_v.isApprox(Eigen::Vector3d::Zero(), 1e-9));
    EXPECT_TRUE(result.delta_p.isApprox(Eigen::Vector3d::Zero(), 1e-9));
}

// --- FG-3: covariance propagation and bias-correction Jacobians ---

using slam_core::geometry::log_so3;
using slam_core::imu::delta_p_corrected;
using slam_core::imu::delta_R_corrected;
using slam_core::imu::delta_v_corrected;
using slam_core::imu::ImuNoiseParams;

namespace {

// Deterministic N(0,1) via Box-Muller on raw mt19937 draws; avoids
// std::normal_distribution implementation differences across platforms.
class Gaussian {
public:
    explicit Gaussian(std::uint32_t seed) : rng_(seed) {}
    double operator()() {
        const double u1 = (static_cast<double>(rng_()) + 1.0) / 4294967297.0;
        const double u2 = (static_cast<double>(rng_()) + 1.0) / 4294967297.0;
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
    }
    Eigen::Vector3d vec3() { return {(*this)(), (*this)(), (*this)()}; }

private:
    std::mt19937 rng_;
};

// Time-varying synthetic trajectory: rotating and accelerating.
std::vector<ImuMeasurement> varying_sequence(int n_meas, double dt) {
    std::vector<ImuMeasurement> seq;
    seq.reserve(n_meas);
    for (int i = 0; i < n_meas; ++i) {
        const double   t = i * dt;
        ImuMeasurement m;
        m.timestamp_s = t;
        m.gyro_radps  = {0.3 * std::sin(2.0 * t), -0.2 + 0.1 * std::cos(3.0 * t), 0.5};
        m.accel_mps2  = {1.0 * std::cos(t), -0.5 * std::sin(2.0 * t), 9.81 + 0.3 * t};
        seq.push_back(m);
    }
    return seq;
}

PreintegratedImu integrate_all(const std::vector<ImuMeasurement>& seq,
                               const Eigen::Vector3d&             bg,
                               const Eigen::Vector3d&             ba,
                               double                             dt,
                               const ImuNoiseParams&              noise = ImuNoiseParams{}) {
    PreintegratedImu preint;
    for (const ImuMeasurement& m : seq) integrate(preint, m, bg, ba, dt, noise);
    return preint;
}

}  // namespace

TEST(PreintegrationCovariance, StaysZeroWithoutNoise) {
    const auto             seq = varying_sequence(100, 0.005);
    const PreintegratedImu p = integrate_all(seq, {0.01, -0.02, 0.015}, {0.05, -0.1, 0.08}, 0.005);
    EXPECT_LT(p.covariance.cwiseAbs().maxCoeff(), 1e-300);
}

TEST(PreintegrationCovariance, SymmetricPsdAndGrowing) {
    const ImuNoiseParams noise{1.7e-4, 2.0e-3};  // EuRoC-like densities
    const auto           seq = varying_sequence(200, 0.005);

    PreintegratedImu preint;
    double           prev_trace = 0.0;
    for (const ImuMeasurement& m : seq) {
        integrate(preint, m, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), 0.005, noise);
        const double trace = preint.covariance.trace();
        EXPECT_GT(trace, prev_trace);
        prev_trace = trace;
    }

    EXPECT_TRUE(preint.covariance.isApprox(preint.covariance.transpose(), 1e-12));
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 9, 9>> eig(preint.covariance);
    EXPECT_GE(eig.eigenvalues().minCoeff(), -1e-18);
    EXPECT_TRUE(preint.covariance.allFinite());
}

TEST(PreintegrationBiasJacobians, MatchNumericDerivatives) {
    const double          dt  = 0.005;
    const auto            seq = varying_sequence(200, dt);  // 1 s of data
    const Eigen::Vector3d bg(0.01, -0.02, 0.015);
    const Eigen::Vector3d ba(0.05, -0.1, 0.08);

    const PreintegratedImu nominal = integrate_all(seq, bg, ba, dt);

    const double    h = 1e-6;
    Eigen::Matrix3d num_dR_dbg, num_dv_dbg, num_dv_dba, num_dp_dbg, num_dp_dba;
    Eigen::Matrix3d num_dR_dba;  // must be ~0: accel bias cannot rotate

    for (int i = 0; i < 3; ++i) {
        Eigen::Vector3d e = Eigen::Vector3d::Zero();
        e(i)              = h;

        const PreintegratedImu gp = integrate_all(seq, bg + e, ba, dt);
        const PreintegratedImu gm = integrate_all(seq, bg - e, ba, dt);
        const PreintegratedImu ap = integrate_all(seq, bg, ba + e, dt);
        const PreintegratedImu am = integrate_all(seq, bg, ba - e, dt);

        num_dR_dbg.col(i) = (log_so3(nominal.delta_R.transpose() * gp.delta_R) -
                             log_so3(nominal.delta_R.transpose() * gm.delta_R)) /
                            (2.0 * h);
        num_dR_dba.col(i) = (log_so3(nominal.delta_R.transpose() * ap.delta_R) -
                             log_so3(nominal.delta_R.transpose() * am.delta_R)) /
                            (2.0 * h);
        num_dv_dbg.col(i) = (gp.delta_v - gm.delta_v) / (2.0 * h);
        num_dv_dba.col(i) = (ap.delta_v - am.delta_v) / (2.0 * h);
        num_dp_dbg.col(i) = (gp.delta_p - gm.delta_p) / (2.0 * h);
        num_dp_dba.col(i) = (ap.delta_p - am.delta_p) / (2.0 * h);
    }

    EXPECT_LT((nominal.d_delta_R_d_bg - num_dR_dbg).cwiseAbs().maxCoeff(), 1e-5);
    EXPECT_LT((nominal.d_delta_v_d_bg - num_dv_dbg).cwiseAbs().maxCoeff(), 1e-5);
    EXPECT_LT((nominal.d_delta_v_d_ba - num_dv_dba).cwiseAbs().maxCoeff(), 1e-5);
    EXPECT_LT((nominal.d_delta_p_d_bg - num_dp_dbg).cwiseAbs().maxCoeff(), 1e-5);
    EXPECT_LT((nominal.d_delta_p_d_ba - num_dp_dba).cwiseAbs().maxCoeff(), 1e-5);
    EXPECT_LT(num_dR_dba.cwiseAbs().maxCoeff(), 1e-9);
}

TEST(PreintegrationBiasJacobians, CorrectedDeltasMatchReintegration) {
    const double          dt  = 0.005;
    const auto            seq = varying_sequence(200, dt);
    const Eigen::Vector3d bg(0.01, -0.02, 0.015);
    const Eigen::Vector3d ba(0.05, -0.1, 0.08);
    const Eigen::Vector3d dbg(2e-3, -1e-3, 1.5e-3);
    const Eigen::Vector3d dba(5e-3, 3e-3, -4e-3);

    const PreintegratedImu nominal = integrate_all(seq, bg, ba, dt);
    const PreintegratedImu exact   = integrate_all(seq, bg + dbg, ba + dba, dt);

    const Eigen::Matrix3d R_corr = delta_R_corrected(nominal, dbg);
    const Eigen::Vector3d v_corr = delta_v_corrected(nominal, dbg, dba);
    const Eigen::Vector3d p_corr = delta_p_corrected(nominal, dbg, dba);

    const double R_err_corr   = log_so3(exact.delta_R.transpose() * R_corr).norm();
    const double R_err_uncorr = log_so3(exact.delta_R.transpose() * nominal.delta_R).norm();
    const double v_err_corr   = (v_corr - exact.delta_v).norm();
    const double v_err_uncorr = (nominal.delta_v - exact.delta_v).norm();
    const double p_err_corr   = (p_corr - exact.delta_p).norm();
    const double p_err_uncorr = (nominal.delta_p - exact.delta_p).norm();

    // First-order correction: residual error is O(|db|^2).
    EXPECT_LT(R_err_corr, 1e-5);
    EXPECT_LT(v_err_corr, 1e-4);
    EXPECT_LT(p_err_corr, 1e-4);
    // And it must actually help, by a wide margin.
    EXPECT_GT(R_err_uncorr, 10.0 * R_err_corr);
    EXPECT_GT(v_err_uncorr, 10.0 * v_err_corr);
    EXPECT_GT(p_err_uncorr, 10.0 * p_err_corr);
}

TEST(PreintegrationCovariance, MonteCarloConsistency) {
    // Error convention (matches the propagation model):
    //   e_theta = log_so3(delta_R_refᵀ · delta_R_trial), e_v/e_p additive.
    const double         dt       = 0.005;
    const int            n_steps  = 100;  // 0.5 s
    const int            n_trials = 500;
    const ImuNoiseParams noise{1.7e-3, 2.0e-2};  // 10x EuRoC: cleaner statistics

    const auto             seq = varying_sequence(n_steps, dt);
    const PreintegratedImu ref =
        integrate_all(seq, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), dt, noise);

    const double sigma_gd = noise.gyro_noise_density / std::sqrt(dt);
    const double sigma_ad = noise.accel_noise_density / std::sqrt(dt);

    Gaussian                          gaussian(1234);
    Eigen::Matrix<double, 9, 9>       sample_cov = Eigen::Matrix<double, 9, 9>::Zero();
    double                            nees_sum   = 0.0;
    const Eigen::Matrix<double, 9, 9> info       = ref.covariance.inverse();

    for (int trial = 0; trial < n_trials; ++trial) {
        PreintegratedImu p;
        for (const ImuMeasurement& m : seq) {
            ImuMeasurement noisy = m;
            noisy.gyro_radps += sigma_gd * gaussian.vec3();
            noisy.accel_mps2 += sigma_ad * gaussian.vec3();
            integrate(p, noisy, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), dt);
        }
        Eigen::Matrix<double, 9, 1> e;
        e.segment<3>(0) = log_so3(ref.delta_R.transpose() * p.delta_R);
        e.segment<3>(3) = p.delta_v - ref.delta_v;
        e.segment<3>(6) = p.delta_p - ref.delta_p;

        sample_cov += e * e.transpose();
        nees_sum += e.dot(info * e);
    }
    sample_cov /= n_trials;

    // Mean NEES for a 9-dim consistent estimator is 9; SE ≈ sqrt(2*9/500).
    const double mean_nees = nees_sum / n_trials;
    EXPECT_GT(mean_nees, 7.8);
    EXPECT_LT(mean_nees, 10.2);

    // Per-axis variance ratios (sampling error ~ sqrt(2/500) ≈ 6%).
    for (int i = 0; i < 9; ++i) {
        const double ratio = sample_cov(i, i) / ref.covariance(i, i);
        EXPECT_GT(ratio, 0.75) << "axis " << i;
        EXPECT_LT(ratio, 1.30) << "axis " << i;
    }
}
