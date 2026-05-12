#include <gtest/gtest.h>
#include <Eigen/Core>
#include "slam_core/version.hpp"

TEST(Placeholder, TestRunnerWorks) {
    EXPECT_EQ(1 + 1, 2);
}

TEST(Placeholder, EigenIncludeWorks) {
    Eigen::Vector3d v(1.0, 2.0, 3.0);
    EXPECT_DOUBLE_EQ(v.norm(), std::sqrt(14.0));
}

TEST(Placeholder, VersionHeaderIncludeWorks) {
    EXPECT_EQ(slam_core::VERSION_MAJOR, 0);
}
