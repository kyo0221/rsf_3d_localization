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

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace rsf_3d_localization
{

namespace
{

constexpr float kNoGround = std::numeric_limits<float>::infinity();

void chamfer(std::vector<float>& d, int nx, int ny, float step)
{
  const float diagonal = step * std::sqrt(2.0f);
  auto relax = [&](int i, int j, int di, int dj, float cost) {
    int a = i + di, b = j + dj;
    if (a < 0 || b < 0 || a >= nx || b >= ny)
      return;
    float& v = d[static_cast<size_t>(i) * ny + j];
    v = std::min(v, d[static_cast<size_t>(a) * ny + b] + cost);
  };
  for (int i = 0; i < nx; i++)
    for (int j = 0; j < ny; j++)
    {
      relax(i, j, -1, 0, step);
      relax(i, j, 0, -1, step);
      relax(i, j, -1, -1, diagonal);
      relax(i, j, -1, 1, diagonal);
    }
  for (int i = nx - 1; i >= 0; i--)
    for (int j = ny - 1; j >= 0; j--)
    {
      relax(i, j, 1, 0, step);
      relax(i, j, 0, 1, step);
      relax(i, j, 1, 1, diagonal);
      relax(i, j, 1, -1, diagonal);
    }
}

}  // namespace

void LikelihoodGrid::build(const std::vector<Point3f>& points, const GridParams& params)
{
  params_ = params;
  likelihood_.clear();
  if (points.empty())
    return;
  float min_x = points[0].x, max_x = points[0].x, min_y = points[0].y, max_y = points[0].y;
  for (const auto& p : points)
  {
    min_x = std::min(min_x, p.x), max_x = std::max(max_x, p.x);
    min_y = std::min(min_y, p.y), max_y = std::max(max_y, p.y);
  }
  x0_ = std::floor(min_x) - 1.0;
  y0_ = std::floor(min_y) - 1.0;
  gx_ = static_cast<int>(std::ceil((max_x - x0_) / params.ground_resolution)) + 2;
  gy_ = static_cast<int>(std::ceil((max_y - y0_) / params.ground_resolution)) + 2;
  std::vector<float> lowest(static_cast<size_t>(gx_) * gy_, kNoGround);
  for (const auto& p : points)
  {
    size_t k = static_cast<size_t>((p.x - x0_) / params.ground_resolution) * gy_ +
               static_cast<size_t>((p.y - y0_) / params.ground_resolution);
    lowest[k] = std::min(lowest[k], p.z);
  }
  ground_.assign(lowest.size(), kNoGround);
  for (int i = 0; i < gx_; i++)
    for (int j = 0; j < gy_; j++)
      for (int a = std::max(i - 1, 0); a <= std::min(i + 1, gx_ - 1); a++)
        for (int b = std::max(j - 1, 0); b <= std::min(j + 1, gy_ - 1); b++)
          ground_[static_cast<size_t>(i) * gy_ + j] =
              std::min(ground_[static_cast<size_t>(i) * gy_ + j], lowest[static_cast<size_t>(a) * gy_ + b]);

  nx_ = static_cast<int>(std::ceil(gx_ * params.ground_resolution / params.resolution));
  ny_ = static_cast<int>(std::ceil(gy_ * params.ground_resolution / params.resolution));
  std::vector<float> distance(static_cast<size_t>(nx_) * ny_, std::numeric_limits<float>::max() / 4);
  for (const auto& p : points)
  {
    double h = p.z - ground(p.x, p.y);
    if (h <= params.band_min || h >= params.band_max)
      continue;
    distance[static_cast<size_t>((p.x - x0_) / params.resolution) * ny_ +
             static_cast<size_t>((p.y - y0_) / params.resolution)] = 0.0f;
  }
  chamfer(distance, nx_, ny_, static_cast<float>(params.resolution));
  likelihood_.resize(distance.size());
  const float inv = static_cast<float>(1.0 / (2.0 * params.sigma * params.sigma));
  for (size_t k = 0; k < distance.size(); k++)
    likelihood_[k] = std::exp(-distance[k] * distance[k] * inv);
}

bool LikelihoodGrid::empty() const
{
  return likelihood_.empty();
}

double LikelihoodGrid::ground(double x, double y) const
{
  int i = static_cast<int>((x - x0_) / params_.ground_resolution);
  int j = static_cast<int>((y - y0_) / params_.ground_resolution);
  if (i < 0 || j < 0 || i >= gx_ || j >= gy_)
    return kNoGround;
  return ground_[static_cast<size_t>(i) * gy_ + j];
}

float LikelihoodGrid::at(double x, double y) const
{
  int i = static_cast<int>((x - x0_) / params_.resolution), j = static_cast<int>((y - y0_) / params_.resolution);
  if (i < 0 || j < 0 || i >= nx_ || j >= ny_)
    return 0.0f;
  return likelihood_[static_cast<size_t>(i) * ny_ + j];
}

double LikelihoodGrid::score(double x, double y, double yaw, const std::vector<Point3f>& points) const
{
  double c = std::cos(yaw), s = std::sin(yaw), sum = 0.0;
  for (const auto& p : points)
    sum += at(x + c * p.x - s * p.y, y + s * p.x + c * p.y);
  return points.empty() ? 0.0 : sum / static_cast<double>(points.size());
}

std::vector<PlanarPose> LikelihoodGrid::search(const PlanarPose& center, double radius, double yaw_range,
                                               double step, double yaw_step, const std::vector<Point3f>& points,
                                               size_t count, double separation) const
{
  std::vector<PlanarPose> all;
  int n = static_cast<int>(std::floor(radius / step)), m = static_cast<int>(std::floor(yaw_range / yaw_step));
  std::vector<std::array<double, 2>> rotated(points.size());
  for (int a = -m; a <= m; a++)
  {
    double yaw = center.yaw + a * yaw_step, c = std::cos(yaw), s = std::sin(yaw);
    for (size_t k = 0; k < points.size(); k++)
      rotated[k] = { c * points[k].x - s * points[k].y, s * points[k].x + c * points[k].y };
    for (int i = -n; i <= n; i++)
      for (int j = -n; j <= n; j++)
      {
        double x = center.x + i * step, y = center.y + j * step, sum = 0.0;
        for (const auto& q : rotated)
          sum += at(x + q[0], y + q[1]);
        all.push_back({ x, y, yaw, sum });
      }
  }
  std::sort(all.begin(), all.end(), [](const PlanarPose& a, const PlanarPose& b) { return a.score > b.score; });
  std::vector<PlanarPose> peaks;
  for (const auto& c : all)
  {
    bool distinct = std::all_of(peaks.begin(), peaks.end(), [&](const PlanarPose& p) {
      return std::hypot(p.x - c.x, p.y - c.y) >= separation;
    });
    if (!distinct)
      continue;
    peaks.push_back(c);
    if (peaks.size() >= count)
      break;
  }
  return peaks;
}

}  // namespace rsf_3d_localization
