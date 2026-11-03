// Copyright 2026 kyo, SPDX-License-Identifier: Apache-2.0

#include "rsf_3d_localization/pose_math.hpp"

#include <gtest/gtest.h>

#include <cmath>

using rsf_3d_localization::Pose6;
using rsf_3d_localization::Transform;

TEST(PoseMath, NormalizeAngle)
{
  EXPECT_NEAR(rsf_3d_localization::normalize_angle(3.0 * M_PI), M_PI, 1e-9);
  EXPECT_NEAR(rsf_3d_localization::normalize_angle(-0.5 * M_PI), -0.5 * M_PI, 1e-9);
}

TEST(PoseMath, PoseRoundTrip)
{
  Pose6 p;
  p.x = 1.0;
  p.y = -2.0;
  p.z = 0.5;
  p.roll = 0.1;
  p.pitch = -0.2;
  p.yaw = 2.5;
  Pose6 q = rsf_3d_localization::to_pose(rsf_3d_localization::to_transform(p));
  EXPECT_NEAR(q.x, p.x, 1e-9);
  EXPECT_NEAR(q.y, p.y, 1e-9);
  EXPECT_NEAR(q.z, p.z, 1e-9);
  EXPECT_NEAR(q.roll, p.roll, 1e-9);
  EXPECT_NEAR(q.pitch, p.pitch, 1e-9);
  EXPECT_NEAR(q.yaw, p.yaw, 1e-9);
}

TEST(PoseMath, ComposeWithInverseIsIdentity)
{
  Pose6 p;
  p.x = 3.0;
  p.y = 1.0;
  p.roll = 0.3;
  p.yaw = -1.2;
  Transform t = rsf_3d_localization::to_transform(p);
  Transform i = rsf_3d_localization::compose(t, rsf_3d_localization::inverse(t));
  for (int k = 0; k < 9; k++)
    EXPECT_NEAR(i.r[k], k % 4 == 0 ? 1.0 : 0.0, 1e-9);
  for (int k = 0; k < 3; k++)
    EXPECT_NEAR(i.t[k], 0.0, 1e-9);
}

TEST(PoseMath, QuaternionMatchesRotation)
{
  auto q = rsf_3d_localization::quaternion_from_rpy(0.2, -0.1, 1.0);
  auto r = rsf_3d_localization::rotation_from_quaternion(q[0], q[1], q[2], q[3]);
  auto e = rsf_3d_localization::rotation_from_rpy(0.2, -0.1, 1.0);
  for (int k = 0; k < 9; k++)
    EXPECT_NEAR(r[k], e[k], 1e-9);
}
