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

#ifndef RSF_3D_LOCALIZATION__NDT_MAP_HPP_
#define RSF_3D_LOCALIZATION__NDT_MAP_HPP_

#include "rsf_3d_localization/pcd_io.hpp"
#include "rsf_3d_localization/pose_math.hpp"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace rsf_3d_localization
{

struct NdtVoxel
{
  float mean[3];
  float icov[6];
};

struct VoxelKeyHash
{
  size_t operator()(int64_t k) const
  {
    uint64_t x = static_cast<uint64_t>(k) + 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return static_cast<size_t>(x ^ (x >> 31));
  }
};

inline int64_t voxel_key(int64_t ix, int64_t iy, int64_t iz)
{
  constexpr int64_t offset = 1 << 20;
  return ((ix + offset) << 42) | ((iy + offset) << 21) | (iz + offset);
}

class NdtMap
{
public:
  void build(const std::vector<Point3f>& points, double resolution, int min_points, double eigen_ratio);
  bool empty() const;

  const std::vector<NdtVoxel>& voxels() const;

  double score(const Transform& t, const std::vector<Point3f>& points, bool neighbors = false) const;
  double refine(Transform& t, const std::vector<Point3f>& points, int max_iterations, double epsilon) const;

private:
  double resolution_ = 1.0;
  double inv_resolution_ = 1.0;
  std::vector<NdtVoxel> voxels_;
  std::unordered_map<int64_t, uint32_t, VoxelKeyHash> index_;

  int find(double x, double y, double z, int count, const NdtVoxel** out) const;
  void gradient_hessian(const Pose6& p, const std::vector<Point3f>& points, double g[6], double hess[6][6]) const;
  bool lm_step(Pose6& p, const double g[6], const double hess[6][6], const std::vector<Point3f>& points,
               double& current, double& lambda, double& step_norm) const;
};

}  // namespace rsf_3d_localization

#endif  // RSF_3D_LOCALIZATION__NDT_MAP_HPP_
