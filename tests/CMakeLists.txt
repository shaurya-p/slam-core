include(GoogleTest)

add_executable(test_placeholder test_placeholder.cpp)
target_include_directories(test_placeholder PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(test_placeholder PRIVATE GTest::gtest_main Eigen3::Eigen)
gtest_discover_tests(test_placeholder)

add_executable(test_geometry test_geometry.cpp)
target_link_libraries(test_geometry PRIVATE GTest::gtest_main slam_core)
gtest_discover_tests(test_geometry)
