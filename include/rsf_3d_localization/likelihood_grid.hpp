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

#ifndef RSF_3D_LOCALIZATION__LIKELIHOOD_GRID_HPP_
#define RSF_3D_LOCALIZATION__LIKELIHOOD_GRID_HPP_

#include "rsf_3d_localization/pcd_io.hpp"

#include <vector>

namespace rsf_3d_localization
{

struct GridParams
{
  double resolution = 0.25;
  double ground_resolution = 1.0;
  double band_min = 0.5;
  double band_max = 3.0;
  double sigma = 0.3;
};

struct PlanarPose
{
  double x, y, yaw, score;
};

class LikelihoodGrid
{
public:
  void build(const std::vector<Point3f>& points, const GridParams& params);
  bool empty() const;
  double ground(double x, double y) const;
  double score(double x, double y, double yaw, const std::vector<Point3f>& points) const;
  std::vector<PlanarPose> search(const PlanarPose& center, double radius, double yaw_range, double step,
                                 double yaw_step, const std::vector<Point3f>& points, size_t count,
                                 double separation) const;

private:
  GridParams params_;
  double x0_ = 0.0, y0_ = 0.0;
  int nx_ = 0, ny_ = 0, gx_ = 0, gy_ = 0;
  std::vector<float> likelihood_;
  std::vector<float> ground_;

  float at(double x, double y) const;
};

}  // namespace rsf_3d_localization

#endif  // RSF_3D_LOCALIZATION__LIKELIHOOD_GRID_HPP_
