#include <gtest/gtest.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "slam_core/geometry/so3.hpp"
#include "slam_core/imu/imu_state.hpp"

using slam_core::geometry::is_valid_rotation;
using slam_core::imu::ImuMeasurement;
using slam_core::imu::ImuState;
using slam_core::imu::propagate_imu_state;

// --- helpers ---

static ImuState make_state() {
    ImuState s;
    s.timestamp_s     = 1.0;
    s.R_W_B           = Eigen::Matrix3d::Identity();
    s.p_W_B           = Eigen::Vector3d::Zero();
    s.v_W_B           = Eigen::Vector3d::Zero();
    s.gyro_bias_radps = Eigen::Vector3d::Zero();
    s.accel_bias_mps2 = Eigen::Vector3d::Zero();
    return s;
}

static ImuMeasurement make_meas() {
    ImuMeasurement m;
    m.timestamp_s = 1.0;
    m.gyro_radps  = Eigen::Vector3d::Zero();
    m.accel_mps2  = Eigen::Vector3d::Zero();
    return m;
}

// --- 1. zero accel, zero gravity: velocity and position unchanged ---

TEST(ImuStatePropagation, GravityDisabledZeroAccelKeepsVelocityAndPosition) {
    ImuState state = make_state();
    state.p_W_B    = {1.0, 2.0, 3.0};
    // v = 0, a_W = 0 -> v and p stay at their initial values

    const ImuState next = propagate_imu_state(state, make_meas(), Eigen::Vector3d::Zero(), 0.01);

    EXPECT_TRUE(next.v_W_B.isApprox(state.v_W_B, 1e-12));
    EXPECT_TRUE(next.p_W_B.isApprox(state.p_W_B, 1e-12));
}

// --- 2. specific force cancels gravity: static state unchanged ---

TEST(ImuStatePropagation, SpecificForceCancelsGravityKeepsStaticState) {
    // a_B = [0, 0, 9.81], gravity_W = [0, 0, -9.81], R = I
    // a_W = R * a_B + gravity_W = [0,0,9.81] + [0,0,-9.81] = 0
    // v = 0, p = 0 -> both remain zero
    ImuState       state = make_state();
    ImuMeasurement meas  = make_meas();
    meas.accel_mps2      = {0.0, 0.0, 9.81};
    const Eigen::Vector3d gravity_W{0.0, 0.0, -9.81};

    const ImuState next = propagate_imu_state(state, meas, gravity_W, 0.1);

    EXPECT_TRUE(next.v_W_B.isApprox(Eigen::Vector3d::Zero(), 1e-9));
    EXPECT_TRUE(next.p_W_B.isApprox(Eigen::Vector3d::Zero(), 1e-9));
}

// --- 3. constant acceleration with identity orientation ---

TEST(ImuStatePropagation, ConstantAccelerationIdentityOrientation) {
    // R = I, gravity = 0, a_B = [2, 0, 0], v0 = [1, 0, 0], p0 = 0, dt = 0.5
    // a_W = [2, 0, 0]
    // v_next = [1,0,0] + [2,0,0]*0.5 = [2, 0, 0]
    // p_next = [0,0,0] + [1,0,0]*0.5 + 0.5*[2,0,0]*0.25 = [0.75, 0, 0]
    ImuState state      = make_state();
    state.v_W_B         = {1.0, 0.0, 0.0};
    ImuMeasurement meas = make_meas();
    meas.accel_mps2     = {2.0, 0.0, 0.0};
    const double dt     = 0.5;

    const ImuState next = propagate_imu_state(state, meas, Eigen::Vector3d::Zero(), dt);

    EXPECT_TRUE(next.v_W_B.isApprox(Eigen::Vector3d(2.0, 0.0, 0.0), 1e-9));
    EXPECT_TRUE(next.p_W_B.isApprox(Eigen::Vector3d(0.75, 0.0, 0.0), 1e-9));
}

// --- 4. non-identity orientation rotates body accel into world frame ---

TEST(ImuStatePropagation, RotatedBodyAccelerationIntoWorld) {
    // R_W_B = 90-deg rotation about z: body x-axis -> world y-axis
    // a_B = [1, 0, 0] -> a_W = [0, 1, 0]
    // gravity = 0, v0 = 0, p0 = 0, dt = 1.0
    // v_next = [0, 1, 0]
    // p_next = [0, 0.5, 0]
    ImuState state = make_state();
    state.R_W_B    = Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    ImuMeasurement meas = make_meas();
    meas.accel_mps2     = {1.0, 0.0, 0.0};

    const ImuState next = propagate_imu_state(state, meas, Eigen::Vector3d::Zero(), 1.0);

    EXPECT_TRUE(next.v_W_B.isApprox(Eigen::Vector3d(0.0, 1.0, 0.0), 1e-9));
    EXPECT_TRUE(next.p_W_B.isApprox(Eigen::Vector3d(0.0, 0.5, 0.0), 1e-9));
}

// --- 5. gyro updates orientation ---

TEST(ImuStatePropagation, GyroUpdatesOrientation) {
    // gyro_z = pi/2 rad/s, dt = 1.0 s, zero bias -> 90-deg yaw about z
    ImuState       state = make_state();
    ImuMeasurement meas  = make_meas();
    meas.gyro_radps      = {0.0, 0.0, M_PI / 2.0};

    const ImuState next = propagate_imu_state(state, meas, Eigen::Vector3d::Zero(), 1.0);

    const Eigen::Matrix3d expected =
        Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    EXPECT_TRUE(next.R_W_B.isApprox(expected, 1e-9));
}

// --- 6. gyro bias affects orientation ---

TEST(ImuStatePropagation, GyroBiasAffectsOrientation) {
    // gyro = [0, 0, 1.0], bias = [0, 0, 0.25], dt = 2.0
    // omega = [0, 0, 0.75] -> angle = 1.5 rad about z
    ImuState state        = make_state();
    state.gyro_bias_radps = {0.0, 0.0, 0.25};
    ImuMeasurement meas   = make_meas();
    meas.gyro_radps       = {0.0, 0.0, 1.0};

    const ImuState next = propagate_imu_state(state, meas, Eigen::Vector3d::Zero(), 2.0);

    const Eigen::Matrix3d expected =
        Eigen::AngleAxisd(1.5, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    EXPECT_TRUE(next.R_W_B.isApprox(expected, 1e-9));
}

// --- 7. accel bias affects velocity and position ---

TEST(ImuStatePropagation, AccelBiasAffectsVelocityAndPosition) {
    // accel = [2.0, 0, 0], bias = [0.5, 0, 0], gravity = 0
    // corrected a_W = [1.5, 0, 0], dt = 1.0, v0 = 0, p0 = 0
    // v_next = [1.5, 0, 0]
    // p_next = [0.75, 0, 0]
    ImuState state        = make_state();
    state.accel_bias_mps2 = {0.5, 0.0, 0.0};
    ImuMeasurement meas   = make_meas();
    meas.accel_mps2       = {2.0, 0.0, 0.0};

    const ImuState next = propagate_imu_state(state, meas, Eigen::Vector3d::Zero(), 1.0);

    EXPECT_TRUE(next.v_W_B.isApprox(Eigen::Vector3d(1.5, 0.0, 0.0), 1e-9));
    EXPECT_TRUE(next.p_W_B.isApprox(Eigen::Vector3d(0.75, 0.0, 0.0), 1e-9));
}

// --- 8. biases are carried forward unchanged ---

TEST(ImuStatePropagation, BiasesAreCarriedForwardUnchanged) {
    ImuState state        = make_state();
    state.gyro_bias_radps = {0.01, -0.02, 0.03};
    state.accel_bias_mps2 = {0.1, -0.2, 0.05};

    const ImuState next = propagate_imu_state(state, make_meas(), Eigen::Vector3d::Zero(), 0.01);

    EXPECT_TRUE(next.gyro_bias_radps.isApprox(state.gyro_bias_radps, 1e-12));
    EXPECT_TRUE(next.accel_bias_mps2.isApprox(state.accel_bias_mps2, 1e-12));
}

// --- 9. timestamp advances by dt ---

TEST(ImuStatePropagation, TimestampAdvancesByDt) {
    ImuState state    = make_state();
    state.timestamp_s = 5.0;
    const double dt   = 0.02;

    const ImuState next = propagate_imu_state(state, make_meas(), Eigen::Vector3d::Zero(), dt);

    EXPECT_DOUBLE_EQ(next.timestamp_s, 5.02);
}

// --- 10. invalid dt throws ---

TEST(ImuStatePropagation, InvalidDtThrows) {
    ImuState              state = make_state();
    ImuMeasurement        meas  = make_meas();
    const Eigen::Vector3d g     = Eigen::Vector3d::Zero();
    constexpr double      kNaN  = std::numeric_limits<double>::quiet_NaN();
    constexpr double      kInf  = std::numeric_limits<double>::infinity();

    EXPECT_THROW(propagate_imu_state(state, meas, g, 0.0), std::invalid_argument);
    EXPECT_THROW(propagate_imu_state(state, meas, g, -1.0), std::invalid_argument);
    EXPECT_THROW(propagate_imu_state(state, meas, g, kNaN), std::invalid_argument);
    EXPECT_THROW(propagate_imu_state(state, meas, g, kInf), std::invalid_argument);
    EXPECT_THROW(propagate_imu_state(state, meas, g, -kInf), std::invalid_argument);
}

// --- 11. non-finite inputs throw ---

TEST(ImuStatePropagation, NonFiniteInputsThrow) {
    constexpr double      kNaN = std::numeric_limits<double>::quiet_NaN();
    const Eigen::Vector3d g    = Eigen::Vector3d::Zero();
    const double          dt   = 0.01;

    {
        ImuState s  = make_state();
        s.p_W_B.x() = kNaN;
        EXPECT_THROW(propagate_imu_state(s, make_meas(), g, dt), std::invalid_argument);
    }
    {
        ImuState s  = make_state();
        s.v_W_B.y() = kNaN;
        EXPECT_THROW(propagate_imu_state(s, make_meas(), g, dt), std::invalid_argument);
    }
    {
        ImuState s            = make_state();
        s.gyro_bias_radps.z() = kNaN;
        EXPECT_THROW(propagate_imu_state(s, make_meas(), g, dt), std::invalid_argument);
    }
    {
        ImuState s            = make_state();
        s.accel_bias_mps2.x() = kNaN;
        EXPECT_THROW(propagate_imu_state(s, make_meas(), g, dt), std::invalid_argument);
    }
    {
        ImuMeasurement m = make_meas();
        m.accel_mps2.x() = kNaN;
        EXPECT_THROW(propagate_imu_state(make_state(), m, g, dt), std::invalid_argument);
    }
    {
        ImuMeasurement m = make_meas();
        m.gyro_radps.z() = kNaN;
        EXPECT_THROW(propagate_imu_state(make_state(), m, g, dt), std::invalid_argument);
    }
    {
        const Eigen::Vector3d bad_g{0.0, 0.0, kNaN};
        EXPECT_THROW(propagate_imu_state(make_state(), make_meas(), bad_g, dt),
                     std::invalid_argument);
    }
}

// --- 12. output rotation remains valid ---

TEST(ImuStatePropagation, OutputRotationRemainsValid) {
    ImuState       state = make_state();
    ImuMeasurement meas  = make_meas();
    meas.gyro_radps      = {0.3, -0.5, 0.8};
    meas.accel_mps2      = {0.1, 0.2, 9.81};
    const Eigen::Vector3d gravity_W{0.0, 0.0, -9.81};

    const ImuState next = propagate_imu_state(state, meas, gravity_W, 0.01);

    EXPECT_TRUE(is_valid_rotation(next.R_W_B, 1e-6));
}
