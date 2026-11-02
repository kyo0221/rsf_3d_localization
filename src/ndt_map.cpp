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

#include "rsf_3d_localization/ndt_map.hpp"

#include <cmath>
#include <cstring>

namespace rsf_3d_localization
{

namespace
{

const int neighbor_offsets[7][3] = { { 0, 0, 0 },  { 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 },
                                     { 0, -1, 0 }, { 0, 0, 1 }, { 0, 0, -1 } };

struct Accumulator
{
  double sum[3] = { 0, 0, 0 };
  double sq[6] = { 0, 0, 0, 0, 0, 0 };
  uint32_t n = 0;
};

struct RotationDerivatives
{
  Mat3 r;
  Mat3 d[3];
  Mat3 h[3][3];
};

float mahalanobis(const NdtVoxel* v, float x, float y, float z)
{
  float dx = x - v->mean[0], dy = y - v->mean[1], dz = z - v->mean[2];
  return dx * (v->icov[0] * dx + 2.0f * (v->icov[1] * dy + v->icov[2] * dz)) +
         dy * (v->icov[3] * dy + 2.0f * v->icov[4] * dz) + dz * v->icov[5] * dz;
}

bool solve6(double a[6][6], double b[6], double x[6])
{
  for (int c = 0; c < 6; c++)
  {
    int pivot = c;
    for (int r = c + 1; r < 6; r++)
    {
      if (std::fabs(a[r][c]) > std::fabs(a[pivot][c]))
        pivot = r;
    }
    if (std::fabs(a[pivot][c]) < 1e-12)
      return false;
    std::swap(a[c], a[pivot]);
    std::swap(b[c], b[pivot]);
    for (int r = c + 1; r < 6; r++)
    {
      double f = a[r][c] / a[c][c];
      for (int k = c; k < 6; k++)
        a[r][k] -= f * a[c][k];
      b[r] -= f * b[c];
    }
  }
  for (int r = 5; r >= 0; r--)
  {
    double s = b[r];
    for (int k = r + 1; k < 6; k++)
      s -= a[r][k] * x[k];
    x[r] = s / a[r][r];
  }
  return true;
}

Mat3 elementary_rotation(int axis, int order, double angle)
{
  double c = std::cos(angle), s = std::sin(angle);
  double cc[3] = { c, -s, -c }, ss[3] = { s, c, -s };
  double ci = cc[order], si = ss[order], one = order == 0 ? 1.0 : 0.0;
  if (axis == 0)
    return { one, 0, 0, 0, ci, -si, 0, si, ci };
  if (axis == 1)
    return { ci, 0, si, 0, one, 0, -si, 0, ci };
  return { ci, -si, 0, si, ci, 0, 0, 0, one };
}

std::unordered_map<int64_t, Accumulator, VoxelKeyHash> accumulate_cells(const std::vector<Point3f>& points,
                                                                        double inv_resolution)
{
  std::unordered_map<int64_t, Accumulator, VoxelKeyHash> cells;
  cells.reserve(points.size() / 16);
  for (const auto& p : points)
  {
    int64_t key = voxel_key(static_cast<int64_t>(std::floor(p.x * inv_resolution)),
                            static_cast<int64_t>(std::floor(p.y * inv_resolution)),
                            static_cast<int64_t>(std::floor(p.z * inv_resolution)));
    Accumulator& a = cells[key];
    double x = p.x, y = p.y, z = p.z;
    a.sum[0] += x, a.sum[1] += y, a.sum[2] += z;
    a.sq[0] += x * x, a.sq[1] += x * y, a.sq[2] += x * z, a.sq[3] += y * y, a.sq[4] += y * z, a.sq[5] += z * z;
    a.n++;
  }
  return cells;
}

Mat3 voxel_covariance(const Accumulator& a, Vec3& m)
{
  double n = a.n;
  m = { a.sum[0] / n, a.sum[1] / n, a.sum[2] / n };
  double k = 1.0 / (n - 1.0);
  Mat3 c;
  c[0] = (a.sq[0] - n * m[0] * m[0]) * k;
  c[1] = c[3] = (a.sq[1] - n * m[0] * m[1]) * k;
  c[2] = c[6] = (a.sq[2] - n * m[0] * m[2]) * k;
  c[4] = (a.sq[3] - n * m[1] * m[1]) * k;
  c[5] = c[7] = (a.sq[4] - n * m[1] * m[2]) * k;
  c[8] = (a.sq[5] - n * m[2] * m[2]) * k;
  return c;
}

bool make_voxel(const Accumulator& a, double eigen_ratio, NdtVoxel& voxel)
{
  Vec3 m;
  Mat3 c = voxel_covariance(a, m);
  Vec3 values;
  Mat3 vectors;
  eigen_symmetric(c, values, vectors);
  double max_value = std::max({ values[0], values[1], values[2] });
  if (max_value < 1e-8)
    return false;
  for (auto& v : values)
    v = 1.0 / std::max(v, max_value * eigen_ratio);
  int idx = 0;
  for (int i = 0; i < 3; i++)
  {
    voxel.mean[i] = static_cast<float>(m[i]);
    for (int j = i; j < 3; j++)
    {
      double s = 0.0;
      for (int e = 0; e < 3; e++)
        s += vectors[i * 3 + e] * values[e] * vectors[j * 3 + e];
      voxel.icov[idx++] = static_cast<float>(s);
    }
  }
  return true;
}

RotationDerivatives rotation_derivatives(const Pose6& p)
{
  double angles[3] = { p.roll, p.pitch, p.yaw };
  auto rot = [&](int kx, int ky, int kz) {
    return mul(elementary_rotation(2, kz, angles[2]),
               mul(elementary_rotation(1, ky, angles[1]), elementary_rotation(0, kx, angles[0])));
  };
  RotationDerivatives rd;
  rd.r = rot(0, 0, 0);
  rd.d[0] = rot(1, 0, 0);
  rd.d[1] = rot(0, 1, 0);
  rd.d[2] = rot(0, 0, 1);
  for (int a = 0; a < 3; a++)
  {
    for (int b = 0; b < 3; b++)
    {
      int k[3] = { 0, 0, 0 };
      k[a]++, k[b]++;
      rd.h[a][b] = rot(k[0], k[1], k[2]);
    }
  }
  return rd;
}

void add_voxel_term(const NdtVoxel* vx, const Vec3& q, const Vec3 jac[6], const Vec3 hv[3][3], double g[6],
                    double hess[6][6])
{
  Mat3 icov = { vx->icov[0], vx->icov[1], vx->icov[2], vx->icov[1], vx->icov[3],
                vx->icov[4], vx->icov[2], vx->icov[4], vx->icov[5] };
  Vec3 e = { q[0] - vx->mean[0], q[1] - vx->mean[1], q[2] - vx->mean[2] };
  Vec3 ae = mul(icov, e);
  double w = std::exp(-0.5 * (e[0] * ae[0] + e[1] * ae[1] + e[2] * ae[2]));
  if (w < 1e-12)
    return;
  Vec3 ajac[6];
  double aj[6];
  for (int k = 0; k < 6; k++)
  {
    ajac[k] = mul(icov, jac[k]);
    aj[k] = ae[0] * jac[k][0] + ae[1] * jac[k][1] + ae[2] * jac[k][2];
    g[k] -= w * aj[k];
  }
  for (int k = 0; k < 6; k++)
  {
    for (int l = k; l < 6; l++)
    {
      double term = jac[k][0] * ajac[l][0] + jac[k][1] * ajac[l][1] + jac[k][2] * ajac[l][2];
      if (k >= 3 && l >= 3)
      {
        const Vec3& h = hv[k - 3][l - 3];
        term += ae[0] * h[0] + ae[1] * h[1] + ae[2] * h[2];
      }
      hess[k][l] += w * (aj[k] * aj[l] - term);
    }
  }
}

double solve_step(const double g[6], const double hess[6][6], double lambda, double step[6])
{
  double a[6][6], b[6];
  for (int k = 0; k < 6; k++)
  {
    for (int l = 0; l < 6; l++)
      a[k][l] = -hess[k][l];
    a[k][k] += lambda * std::max(std::fabs(hess[k][k]), 1e-6);
    b[k] = g[k];
  }
  double ascent = 0.0;
  if (solve6(a, b, step))
  {
    for (int k = 0; k < 6; k++)
      ascent += g[k] * step[k];
  }
  return ascent;
}

}  // namespace

void NdtMap::build(const std::vector<Point3f>& points, double resolution, int min_points, double eigen_ratio)
{
  resolution_ = resolution;
  inv_resolution_ = 1.0 / resolution;
  std::unordered_map<int64_t, Accumulator, VoxelKeyHash> cells = accumulate_cells(points, inv_resolution_);
  voxels_.clear();
  index_.clear();
  index_.reserve(cells.size());
  for (const auto& [key, a] : cells)
  {
    if (a.n < static_cast<uint32_t>(std::max(min_points, 3)))
      continue;
    NdtVoxel voxel;
    if (!make_voxel(a, eigen_ratio, voxel))
      continue;
    index_.emplace(key, static_cast<uint32_t>(voxels_.size()));
    voxels_.push_back(voxel);
  }
}

bool NdtMap::empty() const
{
  return voxels_.empty();
}

const std::vector<NdtVoxel>& NdtMap::voxels() const
{
  return voxels_;
}

int NdtMap::find(double x, double y, double z, int count, const NdtVoxel** out) const
{
  int64_t ix = static_cast<int64_t>(std::floor(x * inv_resolution_)),
          iy = static_cast<int64_t>(std::floor(y * inv_resolution_)),
          iz = static_cast<int64_t>(std::floor(z * inv_resolution_));
  int n = 0;
  for (int i = 0; i < count; i++)
  {
    auto it =
        index_.find(voxel_key(ix + neighbor_offsets[i][0], iy + neighbor_offsets[i][1], iz + neighbor_offsets[i][2]));
    if (it != index_.end())
      out[n++] = &voxels_[it->second];
  }
  return n;
}

double NdtMap::score(const Transform& t, const std::vector<Point3f>& points, bool neighbors) const
{
  float r[9], o[3];
  for (int i = 0; i < 9; i++)
    r[i] = static_cast<float>(t.r[i]);
  for (int i = 0; i < 3; i++)
    o[i] = static_cast<float>(t.t[i]);
  int count = neighbors ? 7 : 1;
  const NdtVoxel* found[7];
  double sum = 0.0;
  for (const auto& p : points)
  {
    float x = r[0] * p.x + r[1] * p.y + r[2] * p.z + o[0];
    float y = r[3] * p.x + r[4] * p.y + r[5] * p.z + o[1];
    float z = r[6] * p.x + r[7] * p.y + r[8] * p.z + o[2];
    int n = find(x, y, z, count, found);
    for (int i = 0; i < n; i++)
      sum += std::exp(-0.5f * mahalanobis(found[i], x, y, z));
  }
  return sum;
}

void NdtMap::gradient_hessian(const Pose6& p, const std::vector<Point3f>& points, double g[6], double hess[6][6]) const
{
  RotationDerivatives rd = rotation_derivatives(p);
  for (const auto& pt : points)
  {
    Vec3 v = { pt.x, pt.y, pt.z };
    Vec3 rv = mul(rd.r, v);
    Vec3 jac[6] = { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, mul(rd.d[0], v), mul(rd.d[1], v), mul(rd.d[2], v) };
    Vec3 q = { rv[0] + p.x, rv[1] + p.y, rv[2] + p.z };
    const NdtVoxel* found[7];
    int found_count = find(q[0], q[1], q[2], 7, found);
    if (found_count == 0)
      continue;
    Vec3 hv[3][3];
    for (int a = 0; a < 3; a++)
    {
      for (int b = a; b < 3; b++)
        hv[a][b] = mul(rd.h[a][b], v);
    }
    for (int f = 0; f < found_count; f++)
      add_voxel_term(found[f], q, jac, hv, g, hess);
  }
  for (int k = 0; k < 6; k++)
  {
    for (int l = 0; l < k; l++)
      hess[k][l] = hess[l][k];
  }
}

bool NdtMap::lm_step(Pose6& p, const double g[6], const double hess[6][6], const std::vector<Point3f>& points,
                     double& current, double& lambda, double& step_norm) const
{
  for (int trial = 0; trial < 10; trial++)
  {
    double step[6];
    if (solve_step(g, hess, lambda, step) > 0.0)
    {
      double trans_norm = std::sqrt(step[0] * step[0] + step[1] * step[1] + step[2] * step[2]);
      double rot_norm = std::sqrt(step[3] * step[3] + step[4] * step[4] + step[5] * step[5]);
      double scale = std::min({ 1.0, 0.5 / std::max(trans_norm, 1e-12), 0.1 / std::max(rot_norm, 1e-12) });
      Pose6 c = p;
      c.x += scale * step[0], c.y += scale * step[1], c.z += scale * step[2];
      c.roll += scale * step[3], c.pitch += scale * step[4], c.yaw += scale * step[5];
      double s = score(to_transform(c), points, true);
      if (s > current)
      {
        current = s;
        p = c;
        step_norm = scale * std::max(trans_norm, rot_norm);
        lambda *= 0.1;
        return true;
      }
    }
    lambda = std::max(lambda * 10.0, 1e-3);
  }
  return false;
}

double NdtMap::refine(Transform& t, const std::vector<Point3f>& points, int max_iterations, double epsilon) const
{
  Pose6 p = to_pose(t);
  double current = score(t, points, true);
  double initial = current;
  double lambda = 0.0;
  for (int iteration = 0; iteration < max_iterations; iteration++)
  {
    double g[6] = { 0 }, hess[6][6] = { { 0 } };
    gradient_hessian(p, points, g, hess);
    double step_norm = 0.0;
    if (!lm_step(p, g, hess, points, current, lambda, step_norm) || step_norm < epsilon)
      break;
  }
  if (current <= initial)
    return initial;
  p.roll = normalize_angle(p.roll);
  p.pitch = normalize_angle(p.pitch);
  p.yaw = normalize_angle(p.yaw);
  t = to_transform(p);
  return current;
}

}  // namespace rsf_3d_localization
