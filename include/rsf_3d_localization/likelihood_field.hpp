// Copyright 2026 kyo, SPDX-License-Identifier: Apache-2.0

#ifndef RSF_3D_LOCALIZATION__LIKELIHOOD_FIELD_HPP_
#define RSF_3D_LOCALIZATION__LIKELIHOOD_FIELD_HPP_

#include "rsf_3d_localization/pcd_io.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace rsf_3d_localization
{

class LikelihoodField
{
public:
  static constexpr int block_bits = 3;
  static constexpr int block_size = 1 << block_bits;
  static constexpr int block_cells = block_size * block_size * block_size;
  using Block = std::array<uint8_t, block_cells>;

  void build(const std::vector<Point3f>& points, double resolution, double max_distance, int threads);
  bool empty() const;
  double max_distance() const;
  size_t block_count() const;

  uint8_t level(float x, float y, float z) const
  {
    int ix = static_cast<int>(std::floor((x - origin_[0]) * inv_resolution_));
    int iy = static_cast<int>(std::floor((y - origin_[1]) * inv_resolution_));
    int iz = static_cast<int>(std::floor((z - origin_[2]) * inv_resolution_));
    if (ix < 0 || iy < 0 || iz < 0)
      return 255;
    int bx = ix >> block_bits, by = iy >> block_bits, bz = iz >> block_bits;
    if (bx >= blocks_dim_[0] || by >= blocks_dim_[1] || bz >= blocks_dim_[2])
      return 255;
    int32_t index = block_index_[(static_cast<size_t>(bx) * blocks_dim_[1] + by) * blocks_dim_[2] + bz];
    if (index < 0)
      return 255;
    int local =
        (((ix & (block_size - 1)) << block_bits) | (iy & (block_size - 1))) << block_bits | (iz & (block_size - 1));
    return blocks_[static_cast<size_t>(index)][static_cast<size_t>(local)];
  }

private:
  float origin_[3] = { 0.0f, 0.0f, 0.0f };
  float inv_resolution_ = 1.0f;
  double resolution_ = 1.0, max_distance_ = 0.0;
  int blocks_dim_[3] = { 0, 0, 0 };
  std::vector<int32_t> block_index_;
  std::vector<Block> blocks_;
};

}  // namespace rsf_3d_localization

#endif  // RSF_3D_LOCALIZATION__LIKELIHOOD_FIELD_HPP_
