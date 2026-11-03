// Copyright 2026 kyo, SPDX-License-Identifier: Apache-2.0

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
