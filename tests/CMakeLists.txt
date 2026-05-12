add_executable(test_placeholder test_placeholder.cpp)

target_include_directories(test_placeholder PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(test_placeholder PRIVATE GTest::gtest_main Eigen3::Eigen)

include(GoogleTest)
gtest_discover_tests(test_placeholder)
