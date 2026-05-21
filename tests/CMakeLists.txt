include(GoogleTest)

add_executable(test_placeholder test_placeholder.cpp)
target_include_directories(test_placeholder PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(test_placeholder PRIVATE GTest::gtest_main Eigen3::Eigen)
gtest_discover_tests(test_placeholder)

add_executable(test_geometry test_geometry.cpp)
target_link_libraries(test_geometry PRIVATE GTest::gtest_main slam_core)
gtest_discover_tests(test_geometry)

add_executable(test_imu test_imu.cpp)
target_link_libraries(test_imu PRIVATE GTest::gtest_main slam_core)
gtest_discover_tests(test_imu)

add_executable(test_gyro_propagation test_gyro_propagation.cpp)
target_link_libraries(test_gyro_propagation PRIVATE GTest::gtest_main slam_core)
gtest_discover_tests(test_gyro_propagation)

add_executable(test_imu_state test_imu_state.cpp)
target_link_libraries(test_imu_state PRIVATE GTest::gtest_main slam_core)
gtest_discover_tests(test_imu_state)

add_executable(test_preintegration test_preintegration.cpp)
target_link_libraries(test_preintegration PRIVATE GTest::gtest_main slam_core)
gtest_discover_tests(test_preintegration)

add_executable(test_imu_initialization test_imu_initialization.cpp)
target_link_libraries(test_imu_initialization PRIVATE GTest::gtest_main slam_core)
gtest_discover_tests(test_imu_initialization)
