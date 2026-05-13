add_library(slam_core STATIC
    geometry/so3.cpp
    geometry/se3.cpp
)

target_include_directories(slam_core PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(slam_core PUBLIC Eigen3::Eigen)
