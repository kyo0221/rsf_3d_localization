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

#ifndef RSF_3D_LOCALIZATION__PCD_IO_HPP_
#define RSF_3D_LOCALIZATION__PCD_IO_HPP_

#include <string>
#include <vector>

namespace rsf_3d_localization
{

struct Point3f
{
  float x;
  float y;
  float z;
};

// Reads x, y, z of a binary or ascii PCD file, dropping non-finite points.
bool load_pcd(const std::string& path, std::vector<Point3f>& points, std::string& error);

}  // namespace rsf_3d_localization

#endif  // RSF_3D_LOCALIZATION__PCD_IO_HPP_
