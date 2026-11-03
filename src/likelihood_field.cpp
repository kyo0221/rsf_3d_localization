// Copyright 2026 kyo, SPDX-License-Identifier: Apache-2.0

#include "rsf_3d_localization/likelihood_field.hpp"

#include <algorithm>
#include <atomic>
#include <limits>
#include <thread>

namespace rsf_3d_localization
{

namespace
{

using Occupancy = std::array<uint64_t, LikelihoodField::block_cells / 64>;

struct BuildGrid
{
  int dim[3];
  std::vector<int32_t> occupied_index;
  std::vector<Occupancy> occupied;

  size_t flat(int bx, int by, int bz) const
  {
    return (static_cast<size_t>(bx) * dim[1] + by) * dim[2] + bz;
  }

  bool inside(int bx, int by, int bz) const
  {
    return bx >= 0 && by >= 0 && bz >= 0 && bx < dim[0] && by < dim[1] && bz < dim[2];
  }

  bool cell(int ix, int iy, int iz) const
  {
    constexpr int bits = LikelihoodField::block_bits, mask = LikelihoodField::block_size - 1;
    int bx = ix >> bits, by = iy >> bits, bz = iz >> bits;
    if (ix < 0 || iy < 0 || iz < 0 || !inside(bx, by, bz))
      return false;
    int32_t index = occupied_index[flat(bx, by, bz)];
    if (index < 0)
      return false;
    int local = (((ix & mask) << bits) | (iy & mask)) << bits | (iz & mask);
    return (occupied[static_cast<size_t>(index)][static_cast<size_t>(local >> 6)] >> (local & 63)) & 1ULL;
  }
};

bool compute_block(const BuildGrid& grid, int bx, int by, int bz, int radius, double scale,
                   LikelihoodField::Block& block, std::vector<int>& g1, std::vector<int>& g2)
{
  constexpr int size = LikelihoodField::block_size;
  const int span = size + 2 * radius, inf = std::numeric_limits<int>::max() / 4;
  const int x0 = bx * size - radius, y0 = by * size - radius, z0 = bz * size - radius;
  g1.assign(static_cast<size_t>(size * span * span), inf);
  g2.assign(static_cast<size_t>(size * size * span), inf);
  for (int x = 0; x < size; x++)
    for (int y = 0; y < span; y++)
      for (int z = 0; z < span; z++)
        for (int d = -radius; d <= radius; d++)
          if (grid.cell(x0 + radius + x + d, y0 + y, z0 + z))
            g1[static_cast<size_t>((x * span + y) * span + z)] =
                std::min(g1[static_cast<size_t>((x * span + y) * span + z)], d * d);
  for (int x = 0; x < size; x++)
    for (int y = 0; y < size; y++)
      for (int z = 0; z < span; z++)
        for (int d = -radius; d <= radius; d++)
          g2[static_cast<size_t>((x * size + y) * span + z)] =
              std::min(g2[static_cast<size_t>((x * size + y) * span + z)],
                       g1[static_cast<size_t>((x * span + y + radius + d) * span + z)] + d * d);
  bool any = false;
  for (int x = 0; x < size; x++)
    for (int y = 0; y < size; y++)
      for (int z = 0; z < size; z++)
      {
        int best = inf;
        for (int d = -radius; d <= radius; d++)
          best = std::min(best, g2[static_cast<size_t>((x * size + y) * span + z + radius + d)] + d * d);
        double level = best >= inf ? 255.0 : std::min(255.0, std::sqrt(static_cast<double>(best)) * scale);
        block[static_cast<size_t>(((x * size) + y) * size + z)] = static_cast<uint8_t>(std::lround(level));
        any = any || level < 255.0;
      }
  return any;
}

void mark_occupied(const std::vector<Point3f>& points, const float origin[3], float inv_resolution, BuildGrid& grid)
{
  constexpr int bits = LikelihoodField::block_bits, mask = LikelihoodField::block_size - 1;
  grid.occupied_index.assign(static_cast<size_t>(grid.dim[0]) * grid.dim[1] * grid.dim[2], -1);
  for (const auto& p : points)
  {
    int ix = static_cast<int>(std::floor((p.x - origin[0]) * inv_resolution));
    int iy = static_cast<int>(std::floor((p.y - origin[1]) * inv_resolution));
    int iz = static_cast<int>(std::floor((p.z - origin[2]) * inv_resolution));
    int32_t& index = grid.occupied_index[grid.flat(ix >> bits, iy >> bits, iz >> bits)];
    if (index < 0)
    {
      index = static_cast<int32_t>(grid.occupied.size());
      grid.occupied.push_back(Occupancy{});
    }
    int local = (((ix & mask) << bits) | (iy & mask)) << bits | (iz & mask);
    grid.occupied[static_cast<size_t>(index)][static_cast<size_t>(local >> 6)] |= 1ULL << (local & 63);
  }
}

std::vector<std::array<int, 3>> dilate_blocks(const BuildGrid& grid)
{
  std::vector<uint8_t> candidate(grid.occupied_index.size(), 0);
  std::vector<std::array<int, 3>> targets;
  for (int bx = 0; bx < grid.dim[0]; bx++)
    for (int by = 0; by < grid.dim[1]; by++)
      for (int bz = 0; bz < grid.dim[2]; bz++)
      {
        if (grid.occupied_index[grid.flat(bx, by, bz)] < 0)
          continue;
        for (int dx = -1; dx <= 1; dx++)
          for (int dy = -1; dy <= 1; dy++)
            for (int dz = -1; dz <= 1; dz++)
            {
              if (!grid.inside(bx + dx, by + dy, bz + dz) || candidate[grid.flat(bx + dx, by + dy, bz + dz)])
                continue;
              candidate[grid.flat(bx + dx, by + dy, bz + dz)] = 1;
              targets.push_back({ bx + dx, by + dy, bz + dz });
            }
      }
  return targets;
}

std::vector<uint8_t> compute_blocks(const BuildGrid& grid, const std::vector<std::array<int, 3>>& targets, int radius,
                                    double scale, int threads, std::vector<LikelihoodField::Block>& results)
{
  results.resize(targets.size());
  std::vector<uint8_t> used(targets.size(), 0);
  std::atomic<size_t> next{ 0 };
  auto worker = [&]() {
    std::vector<int> g1, g2;
    for (size_t i = next++; i < targets.size(); i = next++)
      used[i] = compute_block(grid, targets[i][0], targets[i][1], targets[i][2], radius, scale, results[i], g1, g2);
  };
  std::vector<std::thread> pool;
  for (int i = 0; i < std::max(1, threads); i++)
    pool.emplace_back(worker);
  for (auto& t : pool)
    t.join();
  return used;
}

}  // namespace

void LikelihoodField::build(const std::vector<Point3f>& points, double resolution, double max_distance, int threads)
{
  blocks_.clear();
  block_index_.clear();
  if (points.empty() || resolution <= 0.0 || max_distance <= 0.0)
    return;
  resolution_ = resolution;
  max_distance_ = max_distance;
  inv_resolution_ = static_cast<float>(1.0 / resolution);
  float lo[3] = { points[0].x, points[0].y, points[0].z }, hi[3] = { lo[0], lo[1], lo[2] };
  for (const auto& p : points)
  {
    lo[0] = std::min(lo[0], p.x), lo[1] = std::min(lo[1], p.y), lo[2] = std::min(lo[2], p.z);
    hi[0] = std::max(hi[0], p.x), hi[1] = std::max(hi[1], p.y), hi[2] = std::max(hi[2], p.z);
  }
  BuildGrid grid;
  float pad = static_cast<float>(block_size * resolution);
  for (int a = 0; a < 3; a++)
  {
    origin_[a] = lo[a] - pad;
    grid.dim[a] = static_cast<int>(std::ceil((hi[a] + pad - origin_[a]) / pad)) + 1;
    blocks_dim_[a] = grid.dim[a];
  }
  mark_occupied(points, origin_, inv_resolution_, grid);
  std::vector<std::array<int, 3>> targets = dilate_blocks(grid);
  std::vector<Block> results;
  int radius = std::min(block_size, static_cast<int>(std::ceil(max_distance / resolution)));
  std::vector<uint8_t> used =
      compute_blocks(grid, targets, radius, 255.0 * resolution / max_distance, threads, results);
  block_index_.assign(grid.occupied_index.size(), -1);
  for (size_t i = 0; i < targets.size(); i++)
  {
    if (!used[i])
      continue;
    block_index_[grid.flat(targets[i][0], targets[i][1], targets[i][2])] = static_cast<int32_t>(blocks_.size());
    blocks_.push_back(results[i]);
  }
}

bool LikelihoodField::empty() const
{
  return blocks_.empty();
}

double LikelihoodField::max_distance() const
{
  return max_distance_;
}

size_t LikelihoodField::block_count() const
{
  return blocks_.size();
}

}  // namespace rsf_3d_localization
