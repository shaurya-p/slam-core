#include <gtest/gtest.h>
#include <Eigen/Core>
#include <cmath>
#include <limits>
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
