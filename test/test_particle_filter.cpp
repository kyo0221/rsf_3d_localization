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
using rsf_3d_localization::odom_sigma;
using rsf_3d_localization::OdomMotion;
using rsf_3d_localization::OdomNoise;
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

namespace
{

// 粒子の位置・姿勢の平均と標準偏差 (x, y, yaw, pitch)
struct Spread
{
  double mx, my, myaw, mpitch, sx, sy, syaw;
};

Spread spread_of(const ParticleFilter& pf)
{
  double n = static_cast<double>(pf.size()), mx = 0, my = 0, myaw = 0, mpitch = 0;
  for (const auto& t : pf.poses())
  {
    Pose6 p = to_pose(t);
    mx += p.x / n, my += p.y / n, myaw += p.yaw / n, mpitch += p.pitch / n;
  }
  double vx = 0, vy = 0, vyaw = 0;
  for (const auto& t : pf.poses())
  {
    Pose6 p = to_pose(t);
    vx += (p.x - mx) * (p.x - mx) / n, vy += (p.y - my) * (p.y - my) / n, vyaw += (p.yaw - myaw) * (p.yaw - myaw) / n;
  }
  return { mx, my, myaw, mpitch, std::sqrt(vx), std::sqrt(vy), std::sqrt(vyaw) };
}

ParticleFilter at(const Pose6& pose, int count = 4000)
{
  ParticleFilter pf;
  pf.reset(to_transform(pose), count, PoseSpread{ 0.0, 0.0, 0.0, 0.0 }, 11U);
  return pf;
}

}  // namespace

TEST(OdomMotionModel, SigmaFollowsEmclVariancePerTravelAndRotation)
{
  OdomNoise noise;
  noise.ff = 0.08, noise.fr = 0.01, noise.fl = 0.004, noise.rf = 0.0004, noise.rr = 0.002;
  auto s = odom_sigma(OdomMotion{ -0.25, 0.2, 0.0, 0.0 }, noise);  // 後退でも移動量の絶対値で決まる
  EXPECT_NEAR(s.along, std::sqrt(0.08 * 0.25 + 0.01 * 0.2), 1e-12);
  EXPECT_NEAR(s.lateral, std::sqrt(0.004 * 0.25), 1e-12);
  EXPECT_NEAR(s.yaw, std::sqrt(0.0004 * 0.25 + 0.002 * 0.2), 1e-12);
}

TEST(OdomMotionModel, StationaryRobotKeepsParticlesInPlace)
{
  // 止まっている間 (車輪もジャイロも 0) は粒子をばらまかない
  ParticleFilter pf = at(Pose6{ 1.0, 2.0, 0.0, 0.0, 0.0, 0.3 }, 200);
  for (int i = 0; i < 60; i++)
    pf.predict_odometry(OdomMotion{ 0.0, 0.0, 0.0, 0.0 }, OdomNoise{});
  Spread s = spread_of(pf);
  EXPECT_NEAR(s.mx, 1.0, 1e-9);
  EXPECT_NEAR(s.my, 2.0, 1e-9);
  EXPECT_NEAR(s.myaw, 0.3, 1e-9);
  EXPECT_LT(s.sx + s.sy + s.syaw, 1e-9);
}

TEST(OdomMotionModel, MovesAlongEachParticleHeadingWithEmclSpread)
{
  OdomNoise noise;
  noise.ff = 0.04, noise.fr = 0.0, noise.fl = 0.001, noise.rf = 0.0004, noise.rr = 0.004;
  ParticleFilter pf = at(Pose6{ 0.0, 0.0, 0.0, 0.0, 0.0, M_PI / 2 });  // +y を向いた粒子
  pf.predict_odometry(OdomMotion{ 1.0, 0.1, 0.0, 0.0 }, noise);
  Spread s = spread_of(pf);
  EXPECT_NEAR(s.mx, 0.0, 0.01);
  EXPECT_NEAR(s.my, 1.0, 0.02);
  EXPECT_NEAR(s.myaw, M_PI / 2 + 0.1, 0.005);
  EXPECT_NEAR(s.sy, std::sqrt(0.04), 0.02);  // 進行方向
  EXPECT_NEAR(s.syaw, std::sqrt(0.0004 + 0.004 * 0.1), 0.002);
}

TEST(OdomMotionModel, BackwardMotionAndSlopeTilt)
{
  OdomNoise noise;
  noise.ff = noise.fr = noise.fl = noise.rf = noise.rr = noise.vertical = noise.tilt = 0.0;
  ParticleFilter pf = at(Pose6{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }, 10);
  pf.predict_odometry(OdomMotion{ -0.5, 0.0, 0.0, 0.05 }, noise);
  Spread s = spread_of(pf);
  EXPECT_NEAR(s.mx, -0.5, 1e-9);
  EXPECT_NEAR(s.mpitch, 0.05, 1e-9);
}

TEST(OdomMotionModel, UpdateIsSkippedWhileStationary)
{
  using rsf_3d_localization::odom_update_due;
  using rsf_3d_localization::OdomUpdateRule;
  OdomUpdateRule rule{ 0.25, 0.2, 1.0, 0.02, 0.01 };
  EXPECT_TRUE(odom_update_due(OdomMotion{ 0.25, 0.0, 0.0, 0.0 }, 0.3, rule));
  EXPECT_TRUE(odom_update_due(OdomMotion{ -0.3, 0.0, 0.0, 0.0 }, 0.3, rule));
  EXPECT_TRUE(odom_update_due(OdomMotion{ 0.0, -0.2, 0.0, 0.0 }, 0.3, rule));
  EXPECT_FALSE(odom_update_due(OdomMotion{ 0.1, 0.05, 0.0, 0.0 }, 0.5, rule));
  // ゆっくり動いているときは時間でも更新する
  EXPECT_TRUE(odom_update_due(OdomMotion{ 0.03, 0.0, 0.0, 0.0 }, 1.0, rule));
  EXPECT_TRUE(odom_update_due(OdomMotion{ 0.0, 0.012, 0.0, 0.0 }, 1.5, rule));
  // 止まっている間 (車輪の数 mm とジャイロの雑音だけ) は何秒経っても更新しない
  EXPECT_FALSE(odom_update_due(OdomMotion{ 0.005, 0.004, 0.0, 0.0 }, 1.0, rule));
  EXPECT_FALSE(odom_update_due(OdomMotion{ 0.0, 0.0, 0.0, 0.0 }, 60.0, rule));
}
