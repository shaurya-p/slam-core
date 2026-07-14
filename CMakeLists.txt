cmake_minimum_required(VERSION 3.24)
project(slam_core VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_TESTING "Build unit tests" ON)

# Eigen's version file enforces same-major compatibility, so a bare minimum
# of 3.4 would reject Eigen 5.x; the range accepts 3.4 through any 5.x.
find_package(Eigen3 3.4...<6 REQUIRED NO_MODULE)

if(BUILD_TESTING)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
  )
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  enable_testing()
endif()

# Warnings apply to project targets only; googletest is added above this line.
if(NOT MSVC)
  add_compile_options(-Wall -Wextra)
endif()

add_subdirectory(src)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
add_subdirectory(tools)
