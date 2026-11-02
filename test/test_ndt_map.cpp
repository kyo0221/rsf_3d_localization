// Copyright 2026 kyo
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "rsf_3d_localization/ndt_map.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

using rsf_3d_localization::Point3f;

namespace
{

// Ground plane with two perpendicular walls so that all six degrees of freedom are observable.
std::vector<Point3f> make_scene()
{
  std::mt19937 rng(0);
  std::uniform_real_distribution<float> u(-10.0f, 10.0f);
  std::normal_distribution<float> n(0.0f, 0.02f);
  std::vector<Point3f> points;
  for (int i = 0; i < 20000; i++)
  {
    float a = u(rng), b = u(rng), h = (u(rng) + 10.0f) * 0.15f;
    points.push_back({ a, b, n(rng) });
    points.push_back({ a, 8.0f + n(rng), h });
    points.push_back({ 8.0f + n(rng), b, h });
    points.push_back({ 0.3f * a - 3.0f + n(rng), 0.3f * b + 2.0f, h });
  }
  return points;
}

std::vector<Point3f> subsample(const std::vector<Point3f>& points, size_t step)
{
  std::vector<Point3f> out;
  for (size_t i = 0; i < points.size(); i += step)
    out.push_back(points[i]);
  return out;
}

}  // namespace

TEST(NdtMap, BuildsVoxels)
{
  rsf_3d_localization::NdtMap map;
  EXPECT_TRUE(map.empty());
  map.build(make_scene(), 1.0, 6, 0.01);
  EXPECT_FALSE(map.empty());
  EXPECT_GT(map.voxels().size(), 100u);
}

TEST(NdtMap, ScoreIsHighestAtTruePose)
{
  std::vector<Point3f> scene = make_scene();
  rsf_3d_localization::NdtMap map;
  map.build(scene, 1.0, 6, 0.01);
  std::vector<Point3f> scan = subsample(scene, 199);
  rsf_3d_localization::Pose6 offset;
  offset.x = 0.4;
  double aligned = map.score(rsf_3d_localization::Transform(), scan);
  double shifted = map.score(rsf_3d_localization::to_transform(offset), scan);
  EXPECT_GT(aligned, shifted);
}

TEST(NdtMap, RefineRecoversOffset)
{
  std::vector<Point3f> scene = make_scene();
  rsf_3d_localization::NdtMap map;
  map.build(scene, 1.0, 6, 0.01);
  rsf_3d_localization::Pose6 start;
  start.x = 0.3;
  start.y = -0.25;
  start.z = 0.1;
  start.yaw = 0.05;
  rsf_3d_localization::Transform t = rsf_3d_localization::to_transform(start);
  std::vector<Point3f> scan = subsample(scene, 199);
  double initial = map.score(t, scan, true);
  EXPECT_GT(map.refine(t, scan, 30, 1e-3), initial);
  rsf_3d_localization::Pose6 p = rsf_3d_localization::to_pose(t);
  EXPECT_NEAR(p.x, 0.0, 0.05);
  EXPECT_NEAR(p.y, 0.0, 0.05);
  EXPECT_NEAR(p.z, 0.0, 0.05);
  EXPECT_NEAR(p.yaw, 0.0, 0.01);
}
