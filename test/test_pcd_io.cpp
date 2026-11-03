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

#include "rsf_3d_localization/pcd_io.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace
{

std::string write_file(const std::string& name, const std::string& body)
{
  std::string path = testing::TempDir() + name;
  std::ofstream file(path, std::ios::binary);
  file << body;
  return path;
}

const char header[] =
    "VERSION .7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n"
    "WIDTH 3\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS 3\n";

}  // namespace

TEST(PcdIo, ReadsAsciiAndDropsNan)
{
  std::string path = write_file("ascii.pcd", std::string(header) + "DATA ascii\n1 2 3 0\nnan 0 0 0\n4 5 6 0\n");
  std::vector<rsf_3d_localization::Point3f> points;
  std::string error;
  ASSERT_TRUE(rsf_3d_localization::load_pcd(path, points, error)) << error;
  ASSERT_EQ(points.size(), 2u);
  EXPECT_FLOAT_EQ(points[1].x, 4.0f);
  EXPECT_FLOAT_EQ(points[1].z, 6.0f);
  std::remove(path.c_str());
}

TEST(PcdIo, ReadsBinary)
{
  float data[3][4] = { { 1, 2, 3, 9 }, { -1, -2, -3, 9 }, { 7, 8, 9, 9 } };
  std::string body(header);
  body += "DATA binary\n";
  body.append(reinterpret_cast<const char*>(data), sizeof(data));
  std::string path = write_file("binary.pcd", body);
  std::vector<rsf_3d_localization::Point3f> points;
  std::string error;
  ASSERT_TRUE(rsf_3d_localization::load_pcd(path, points, error)) << error;
  ASSERT_EQ(points.size(), 3u);
  EXPECT_FLOAT_EQ(points[1].y, -2.0f);
  EXPECT_FLOAT_EQ(points[2].x, 7.0f);
  std::remove(path.c_str());
}

TEST(PcdIo, FailsOnMissingFile)
{
  std::vector<rsf_3d_localization::Point3f> points;
  std::string error;
  EXPECT_FALSE(rsf_3d_localization::load_pcd("/nonexistent/map.pcd", points, error));
  EXPECT_FALSE(error.empty());
}
