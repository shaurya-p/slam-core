include(GoogleTest)

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

add_executable(test_pinhole_camera test_pinhole_camera.cpp)
target_link_libraries(test_pinhole_camera PRIVATE GTest::gtest_main slam_core)
gtest_discover_tests(test_pinhole_camera)

add_executable(test_euroc_csv test_euroc_csv.cpp)
target_link_libraries(test_euroc_csv PRIVATE GTest::gtest_main slam_core)
gtest_discover_tests(test_euroc_csv)

add_executable(test_so3_jacobians test_so3_jacobians.cpp)
target_link_libraries(test_so3_jacobians PRIVATE GTest::gtest_main slam_core)
gtest_discover_tests(test_so3_jacobians)

add_executable(test_optim test_optim.cpp)
target_link_libraries(test_optim PRIVATE GTest::gtest_main slam_core)
gtest_discover_tests(test_optim)
