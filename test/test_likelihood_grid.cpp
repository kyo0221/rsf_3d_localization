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

#include "rsf_3d_localization/likelihood_grid.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

using rsf_3d_localization::GridParams;
using rsf_3d_localization::LikelihoodGrid;
using rsf_3d_localization::Point3f;

namespace
{

std::vector<Point3f> make_map()
{
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  std::vector<Point3f> points;
  for (int i = 0; i < 40000; i++)
  {
    float a = u(rng) * 40.0f - 20.0f, b = u(rng) * 40.0f - 20.0f, h = u(rng) * 2.5f;
    points.push_back({ a, b, 0.0f });
    points.push_back({ a, 10.0f, h });
    points.push_back({ -6.0f, b, h });
    points.push_back({ 7.0f + 0.3f * std::sin(b), b * 0.5f - 4.0f, h });
  }
  return points;
}

}  // namespace

TEST(LikelihoodGrid, GroundIsLowestPoint)
{
  LikelihoodGrid grid;
  grid.build(make_map(), GridParams{});
  EXPECT_NEAR(grid.ground(0.0, 0.0), 0.0, 1e-3);
}

TEST(LikelihoodGrid, SearchFindsTruePose)
{
  std::vector<Point3f> map = make_map();
  LikelihoodGrid grid;
  grid.build(map, GridParams{});
  const double tx = 1.0, ty = -2.0, tyaw = 0.1;
  std::vector<Point3f> scan;
  for (size_t i = 0; i < map.size(); i += 97)
  {
    const auto& p = map[i];
    if (p.z < 0.6f || std::hypot(p.x - tx, p.y - ty) > 15.0)
      continue;
    double dx = p.x - tx, dy = p.y - ty, c = std::cos(tyaw), s = std::sin(tyaw);
    scan.push_back({ static_cast<float>(c * dx + s * dy), static_cast<float>(-s * dx + c * dy), p.z });
  }
  auto peaks = grid.search({ tx + 3.0, ty + 2.0, tyaw - 0.1, 0.0 }, 5.0, 0.2, 0.25, 0.025, scan, 1, 1.0);
  ASSERT_EQ(peaks.size(), 1u);
  EXPECT_NEAR(peaks[0].x, tx, 0.3);
  EXPECT_NEAR(peaks[0].y, ty, 0.3);
  EXPECT_NEAR(peaks[0].yaw, tyaw, 0.03);
}
