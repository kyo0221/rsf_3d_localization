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

#ifndef RSF_3D_LOCALIZATION__POSE_MATH_HPP_
#define RSF_3D_LOCALIZATION__POSE_MATH_HPP_

#include <algorithm>
#include <array>
#include <cmath>

namespace rsf_3d_localization
{

using Vec3 = std::array<double, 3>;
using Mat3 = std::array<double, 9>;

struct Pose6
{
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double roll = 0.0;
  double pitch = 0.0;
  double yaw = 0.0;
};

struct Transform
{
  Mat3 r = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
  Vec3 t = { 0, 0, 0 };
};

inline double normalize_angle(double a)
{
  return std::atan2(std::sin(a), std::cos(a));
}

inline Mat3 mul(const Mat3& a, const Mat3& b)
{
  Mat3 c;
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      c[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
    }
  }
  return c;
}

inline Vec3 mul(const Mat3& a, const Vec3& v)
{
  return { a[0] * v[0] + a[1] * v[1] + a[2] * v[2], a[3] * v[0] + a[4] * v[1] + a[5] * v[2],
           a[6] * v[0] + a[7] * v[1] + a[8] * v[2] };
}

inline Mat3 transpose(const Mat3& a)
{
  return { a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8] };
}

inline Mat3 rotation_from_rpy(double roll, double pitch, double yaw)
{
  double cr = std::cos(roll), sr = std::sin(roll);
  double cp = std::cos(pitch), sp = std::sin(pitch);
  double cy = std::cos(yaw), sy = std::sin(yaw);
  return { cy * cp,
           cy * sp * sr - sy * cr,
           cy * sp * cr + sy * sr,
           sy * cp,
           sy * sp * sr + cy * cr,
           sy * sp * cr - cy * sr,
           -sp,
           cp * sr,
           cp * cr };
}

inline void rpy_from_rotation(const Mat3& r, double& roll, double& pitch, double& yaw)
{
  pitch = std::asin(std::clamp(-r[6], -1.0, 1.0));
  roll = std::atan2(r[7], r[8]);
  yaw = std::atan2(r[3], r[0]);
}

inline Mat3 rotation_from_quaternion(double x, double y, double z, double w)
{
  double n = std::sqrt(x * x + y * y + z * z + w * w);
  if (n < 1e-12)
    return { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
  x /= n, y /= n, z /= n, w /= n;
  return { 1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w),
           2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
           2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y) };
}

inline std::array<double, 4> quaternion_from_rpy(double roll, double pitch, double yaw)
{
  double cr = std::cos(roll * 0.5), sr = std::sin(roll * 0.5);
  double cp = std::cos(pitch * 0.5), sp = std::sin(pitch * 0.5);
  double cy = std::cos(yaw * 0.5), sy = std::sin(yaw * 0.5);
  return { sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy,
           cr * cp * cy + sr * sp * sy };
}

inline Transform to_transform(const Pose6& p)
{
  Transform t;
  t.r = rotation_from_rpy(p.roll, p.pitch, p.yaw);
  t.t = { p.x, p.y, p.z };
  return t;
}

inline Pose6 to_pose(const Transform& t)
{
  Pose6 p;
  p.x = t.t[0];
  p.y = t.t[1];
  p.z = t.t[2];
  rpy_from_rotation(t.r, p.roll, p.pitch, p.yaw);
  return p;
}

inline Vec3 apply(const Transform& a, const Vec3& v)
{
  Vec3 r = mul(a.r, v);
  return { r[0] + a.t[0], r[1] + a.t[1], r[2] + a.t[2] };
}

inline Transform compose(const Transform& a, const Transform& b)
{
  Transform c;
  c.r = mul(a.r, b.r);
  c.t = apply(a, b.t);
  return c;
}

inline Transform inverse(const Transform& a)
{
  Transform b;
  b.r = transpose(a.r);
  Vec3 t = mul(b.r, a.t);
  b.t = { -t[0], -t[1], -t[2] };
  return b;
}

}  // namespace rsf_3d_localization

#endif  // RSF_3D_LOCALIZATION__POSE_MATH_HPP_
