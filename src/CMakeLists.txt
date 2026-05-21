add_library(slam_core STATIC
    geometry/so3.cpp
    geometry/se3.cpp
    imu/imu_measurement.cpp
    imu/gyro_propagation.cpp
    imu/imu_state.cpp
    imu/preintegration.cpp
    imu/initialization.cpp
    camera/pinhole_camera.cpp
)

target_include_directories(slam_core PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(slam_core PUBLIC Eigen3::Eigen)
