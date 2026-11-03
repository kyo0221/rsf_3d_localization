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

#include "rsf_3d_localization/particle_filter.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using rsf_3d_localization::LikelihoodField;
using rsf_3d_localization::MotionNoise;
using rsf_3d_localization::ParticleFilter;
using rsf_3d_localization::Point3f;
using rsf_3d_localization::Pose6;
using rsf_3d_localization::PoseSpread;
using rsf_3d_localization::ScanModel;
using rsf_3d_localization::to_pose;
using rsf_3d_localization::to_transform;

namespace
{

std::vector<Point3f> make_room()
{
  std::vector<Point3f> points;
  for (int i = -100; i <= 100; i++)
    for (int k = 0; k <= 20; k++)
    {
      float a = i * 0.1f, h = k * 0.1f;
      points.push_back({ a, 6.0f, h });
      points.push_back({ -4.0f, a, h });
      points.push_back({ 8.0f + 0.5f * std::sin(a), a, h });
      points.push_back({ a, -5.0f + 0.3f * std::cos(a), h });
    }
  return points;
}

std::vector<Point3f> scan_from(const std::vector<Point3f>& map, const Pose6& pose)
{
  auto inv = rsf_3d_localization::inverse(to_transform(pose));
  std::vector<Point3f> scan;
  for (size_t i = 0; i < map.size(); i += 15)
  {
    auto q = rsf_3d_localization::apply(inv, { map[i].x, map[i].y, map[i].z });
    scan.push_back({ static_cast<float>(q[0]), static_cast<float>(q[1]), static_cast<float>(q[2]) });
  }
  return scan;
}

}  // namespace

TEST(ParticleFilter, ScanWeightingConvergesToTruePose)
{
  std::vector<Point3f> map = make_room();
  LikelihoodField field;
  field.build(map, 0.1, 1.0, 2);
  Pose6 truth{ 0.6, -0.4, 0.0, 0.0, 0.0, 0.08 };
  std::vector<Point3f> scan = scan_from(map, truth);
  ParticleFilter pf;
  pf.reset(to_transform(Pose6{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }), 2000, PoseSpread{ 0.8, 0.02, 0.0, 0.1 }, 3U);
  ScanModel model;
  model.sigma = 0.15;
  for (int i = 0; i < 5; i++)
  {
    pf.weight_scan(field, scan, model);
    pf.resample(0.5);
    pf.predict(to_transform(Pose6{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }), 0.0, MotionNoise{});
  }
  Pose6 e = to_pose(pf.estimate(1.0).pose);
  EXPECT_NEAR(e.x, truth.x, 0.1);
  EXPECT_NEAR(e.y, truth.y, 0.1);
  EXPECT_NEAR(e.yaw, truth.yaw, 0.02);
}

TEST(ParticleFilter, PositionWeightingSelectsNearParticles)
{
  ParticleFilter pf;
  pf.reset(to_transform(Pose6{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }), 1000, PoseSpread{ 2.0, 0.0, 0.0, 0.0 }, 5U);
  pf.weight_position({ 1.0, -0.5, 0.0 }, { 0.0, 0.0, 0.0 }, 0.2, 1.0, 0.0);
  auto e = pf.estimate(1.0);
  EXPECT_NEAR(e.pose.t[0], 1.0, 0.15);
  EXPECT_NEAR(e.pose.t[1], -0.5, 0.15);
  EXPECT_GT(pf.mass_near({ 1.0, -0.5, 0.0 }, { 0.0, 0.0, 0.0 }, 0.6), 0.9);
}

TEST(ParticleFilter, ResampleKeepsCountAndInjectReplaces)
{
  ParticleFilter pf;
  pf.reset(to_transform(Pose6{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }), 500, PoseSpread{ 1.0, 0.0, 0.0, 0.0 }, 7U);
  pf.weight_position({ 0.5, 0.0, 0.0 }, { 0.0, 0.0, 0.0 }, 0.1, 1.0, 0.0);
  EXPECT_TRUE(pf.resample(0.5));
  EXPECT_EQ(pf.size(), 500U);
  pf.inject(to_transform(Pose6{ 20.0, 0.0, 0.0, 0.0, 0.0, 0.0 }), 0.3, PoseSpread{ 0.1, 0.0, 0.0, 0.0 });
  EXPECT_NEAR(pf.mass_near({ 20.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0 }, 1.0), 0.3, 0.01);
}

TEST(ParticleFilter, UniformYawResetCoversAllHeadings)
{
  ParticleFilter pf;
  pf.reset_uniform_yaw(to_transform(Pose6{ 3.0, -2.0, 1.0, 0.0, 0.0, 0.7 }), 720, PoseSpread{ 0.5, 0.1, 0.0, 0.0 },
                       11U);
  EXPECT_EQ(pf.size(), 720U);
  int bins[8] = {};
  double sx = 0.0, sy = 0.0;
  for (const auto& t : pf.poses())
  {
    Pose6 p = to_pose(t);
    bins[static_cast<int>(std::floor((p.yaw + M_PI) / (M_PI / 4.0))) % 8]++;
    sx += p.x, sy += p.y;
  }
  for (int b : bins)
    EXPECT_NEAR(b, 90, 15);
  EXPECT_NEAR(sx / 720.0, 3.0, 0.1);
  EXPECT_NEAR(sy / 720.0, -2.0, 0.1);
  EXPECT_LT(pf.yaw_concentration(), 0.1);
}

TEST(ParticleFilter, YawConcentrationIsHighForNarrowHeadings)
{
  ParticleFilter pf;
  pf.reset(to_transform(Pose6{ 0.0, 0.0, 0.0, 0.0, 0.0, 3.1 }), 500, PoseSpread{ 0.5, 0.0, 0.0, 0.05 }, 13U);
  EXPECT_GT(pf.yaw_concentration(), 0.99);
}

TEST(ParticleFilter, UniformYawConvergesToTrueHeadingWithScans)
{
  std::vector<Point3f> map = make_room();
  LikelihoodField field;
  field.build(map, 0.1, 1.0, 2);
  Pose6 truth{ 0.3, -0.2, 0.0, 0.0, 0.0, 2.6 };
  std::vector<Point3f> scan = scan_from(map, truth);
  ParticleFilter pf;
  pf.reset_uniform_yaw(to_transform(Pose6{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }), 2000, PoseSpread{ 0.5, 0.02, 0.0, 0.0 },
                       17U);
  ScanModel model;
  model.sigma = 0.15;
  for (int i = 0; i < 8; i++)
  {
    pf.weight_scan(field, scan, model);
    pf.resample(0.5);
    pf.predict(to_transform(Pose6{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }), 0.0, MotionNoise{});
  }
  Pose6 e = to_pose(pf.estimate(1.0).pose);
  EXPECT_NEAR(e.x, truth.x, 0.15);
  EXPECT_NEAR(e.y, truth.y, 0.15);
  EXPECT_NEAR(rsf_3d_localization::normalize_angle(e.yaw - truth.yaw), 0.0, 0.05);
  EXPECT_GT(pf.yaw_concentration(), 0.99);
}
