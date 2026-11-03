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

#ifndef RSF_3D_LOCALIZATION__VOXEL_KEY_HPP_
#define RSF_3D_LOCALIZATION__VOXEL_KEY_HPP_

#include <cstddef>
#include <cstdint>

namespace rsf_3d_localization
{

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

}  // namespace rsf_3d_localization

#endif  // RSF_3D_LOCALIZATION__VOXEL_KEY_HPP_
