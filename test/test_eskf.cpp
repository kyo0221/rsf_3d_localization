// Copyright 2026 kyo, SPDX-License-Identifier: Apache-2.0

#include "rsf_3d_localization/eskf.hpp"

#include <gtest/gtest.h>

using rsf_3d_localization::Eskf;
using rsf_3d_localization::EskfNoise;
using rsf_3d_localization::Pose6;
using rsf_3d_localization::to_pose;
using rsf_3d_localization::to_transform;

TEST(Eskf, StationaryPredictionKeepsPose)
{
  Eskf eskf;
  eskf.reset(to_transform(Pose6{ 1.0, 2.0, 3.0, 0.0, 0.0, 0.5 }), 0.0);
  for (int i = 0; i < 1000; i++)
    eskf.predict(Eigen::Vector3d::Zero(), Eigen::Vector3d(0.0, 0.0, 9.80), 0.005, EskfNoise{});
  Pose6 p = to_pose(eskf.pose());
  EXPECT_NEAR(p.x, 1.0, 1e-6);
  EXPECT_NEAR(p.y, 2.0, 1e-6);
  EXPECT_NEAR(p.z, 3.0, 1e-6);
  EXPECT_NEAR(p.yaw, 0.5, 1e-6);
}

TEST(Eskf, PoseUpdateMovesTowardMeasurement)
{
  Eskf eskf;
  eskf.reset(to_transform(Pose6{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }), 0.0);
  Eigen::Matrix<double, 6, 1> sigma;
  sigma << 0.1, 0.1, 0.1, 0.01, 0.01, 0.01;
  for (int i = 0; i < 20; i++)
    eskf.update_pose(to_transform(Pose6{ 0.5, -0.3, 0.0, 0.0, 0.0, 0.05 }), sigma, 0.0, false);
  Pose6 p = to_pose(eskf.pose());
  EXPECT_NEAR(p.x, 0.5, 0.02);
  EXPECT_NEAR(p.y, -0.3, 0.02);
  EXPECT_NEAR(p.yaw, 0.05, 0.005);
}

TEST(Eskf, GatedPositionIsRejected)
{
  Eskf eskf;
  eskf.reset(to_transform(Pose6{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }), 0.0);
  double nis = eskf.update_position(Eigen::Vector3d(50.0, 0.0, 0.0), Eigen::Vector3d::Zero(), 0.05, 0.1, 30.0, false);
  EXPECT_GT(nis, 30.0);
  EXPECT_NEAR(to_pose(eskf.pose()).x, 0.0, 1e-9);
}

TEST(Eskf, SpeedUpdateSetsVelocity)
{
  Eskf eskf;
  eskf.reset(to_transform(Pose6{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }), 0.0);
  for (int i = 0; i < 50; i++)
    eskf.update_forward_speed(1.0, 0.1, 0.0, false);
  EXPECT_NEAR(eskf.forward_speed(), 1.0, 0.1);
}
