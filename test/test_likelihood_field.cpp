// Copyright 2026 kyo, SPDX-License-Identifier: Apache-2.0

#include "rsf_3d_localization/likelihood_field.hpp"

#include <gtest/gtest.h>

#include <vector>

using rsf_3d_localization::LikelihoodField;
using rsf_3d_localization::Point3f;

namespace
{

std::vector<Point3f> make_plane()
{
  std::vector<Point3f> points;
  for (int i = -50; i <= 50; i++)
    for (int j = -50; j <= 50; j++)
      points.push_back({ i * 0.1f, j * 0.1f, 0.0f });
  return points;
}

double distance(const LikelihoodField& field, float x, float y, float z)
{
  return field.level(x, y, z) * field.max_distance() / 255.0;
}

}  // namespace

TEST(LikelihoodField, DistanceToPlane)
{
  LikelihoodField field;
  field.build(make_plane(), 0.1, 1.0, 2);
  ASSERT_FALSE(field.empty());
  EXPECT_NEAR(distance(field, 0.0f, 0.0f, 0.05f), 0.0, 0.02);
  EXPECT_NEAR(distance(field, 0.0f, 0.0f, 0.45f), 0.4, 0.02);
  EXPECT_NEAR(distance(field, 1.0f, -2.0f, -0.65f), 0.7, 0.02);
}

TEST(LikelihoodField, FarPointsAreTruncated)
{
  LikelihoodField field;
  field.build(make_plane(), 0.1, 1.0, 2);
  EXPECT_DOUBLE_EQ(distance(field, 0.0f, 0.0f, 3.0f), 1.0);
  EXPECT_DOUBLE_EQ(distance(field, 100.0f, 0.0f, 0.0f), 1.0);
}
