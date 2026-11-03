// Copyright 2026 kyo, SPDX-License-Identifier: Apache-2.0

#include "rsf_3d_localization/pcd_io.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

namespace rsf_3d_localization
{

namespace
{

struct PcdHeader
{
  std::vector<std::string> fields;
  std::vector<int> sizes;
  std::vector<int> counts;
  std::vector<char> types;
  size_t num_points = 0;
  std::string data_type;
};

struct PcdLayout
{
  size_t xyz_offset[3];
  size_t xyz_token[3];
  size_t step = 0;
  size_t tokens = 0;
};

bool parse_header_line(const std::string& line, PcdHeader& header)
{
  std::istringstream ss(line);
  std::string key;
  ss >> key;
  if (key.empty() || key[0] == '#')
    return false;
  if (key == "FIELDS")
  {
    std::string v;
    while (ss >> v)
      header.fields.push_back(v);
  }
  else if (key == "SIZE" || key == "COUNT")
  {
    std::vector<int>& values = key == "SIZE" ? header.sizes : header.counts;
    int v;
    while (ss >> v)
      values.push_back(v);
  }
  else if (key == "TYPE")
  {
    char v;
    while (ss >> v)
      header.types.push_back(v);
  }
  else if (key == "POINTS")
  {
    ss >> header.num_points;
  }
  else if (key == "DATA")
  {
    ss >> header.data_type;
    return true;
  }
  return false;
}

bool read_pcd_header(std::ifstream& file, PcdHeader& header, std::string& error)
{
  std::string line;
  while (std::getline(file, line))
  {
    if (parse_header_line(line, header))
      break;
  }
  if (header.counts.empty())
    header.counts.assign(header.fields.size(), 1);
  size_t n = header.fields.size();
  if (n == 0 || header.sizes.size() != n || header.types.size() != n || header.counts.size() != n)
  {
    error = "invalid PCD header";
    return false;
  }
  return true;
}

bool find_xyz_layout(const PcdHeader& header, PcdLayout& layout, std::string& error)
{
  int xyz_index[3] = { -1, -1, -1 };
  for (size_t i = 0; i < header.fields.size(); i++)
  {
    for (int k = 0; k < 3; k++)
    {
      if (header.fields[i] == std::string(1, static_cast<char>('x' + k)))
      {
        xyz_index[k] = static_cast<int>(i);
        layout.xyz_offset[k] = layout.step;
        layout.xyz_token[k] = layout.tokens;
      }
    }
    layout.step += static_cast<size_t>(header.sizes[i] * header.counts[i]);
    layout.tokens += static_cast<size_t>(header.counts[i]);
  }
  for (int k = 0; k < 3; k++)
  {
    if (xyz_index[k] < 0 || header.types[xyz_index[k]] != 'F' || header.sizes[xyz_index[k]] != 4)
    {
      error = "PCD needs float32 x y z fields";
      return false;
    }
  }
  return true;
}

void read_binary_points(std::ifstream& file, size_t num_points, const PcdLayout& layout, std::vector<Point3f>& points)
{
  size_t step = layout.step;
  std::vector<char> buffer(num_points * step);
  file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
  size_t n = static_cast<size_t>(file.gcount()) / step;
  for (size_t i = 0; i < n; i++)
  {
    const char* base = buffer.data() + i * step;
    Point3f p;
    std::memcpy(&p.x, base + layout.xyz_offset[0], 4);
    std::memcpy(&p.y, base + layout.xyz_offset[1], 4);
    std::memcpy(&p.z, base + layout.xyz_offset[2], 4);
    if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z))
      points.push_back(p);
  }
}

void read_ascii_points(std::ifstream& file, const PcdLayout& layout, std::vector<Point3f>& points)
{
  std::string line;
  while (std::getline(file, line))
  {
    std::vector<float> values;
    const char* s = line.c_str();
    char* end;
    while (true)
    {
      float v = std::strtof(s, &end);
      if (end == s)
        break;
      values.push_back(v);
      s = end;
    }
    if (values.size() < layout.tokens)
      continue;
    Point3f p{ values[layout.xyz_token[0]], values[layout.xyz_token[1]], values[layout.xyz_token[2]] };
    if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z))
      points.push_back(p);
  }
}

}  // namespace

bool load_pcd(const std::string& path, std::vector<Point3f>& points, std::string& error)
{
  std::ifstream file(path, std::ios::binary);
  if (!file)
  {
    error = "cannot open " + path;
    return false;
  }
  PcdHeader header;
  PcdLayout layout;
  if (!read_pcd_header(file, header, error) || !find_xyz_layout(header, layout, error))
    return false;
  points.clear();
  points.reserve(header.num_points);
  if (header.data_type == "binary")
  {
    read_binary_points(file, header.num_points, layout, points);
  }
  else if (header.data_type == "ascii")
  {
    read_ascii_points(file, layout, points);
  }
  else
  {
    error = "unsupported PCD DATA type: " + header.data_type;
    return false;
  }
  return true;
}

}  // namespace rsf_3d_localization
