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

#include "rsf_3d_localization/eskf.hpp"

#include <algorithm>
#include <cmath>

namespace rsf_3d_localization
{

namespace
{

using Eigen::Matrix3d;
using Eigen::Vector3d;

const Vector3d kGravity(0.0, 0.0, -9.80);

Matrix3d skew(const Vector3d& v)
{
  Matrix3d m;
  m << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
  return m;
}

Matrix3d exp_so3(const Vector3d& w)
{
  double angle = w.norm();
  if (angle < 1e-12)
    return Matrix3d::Identity() + skew(w);
  Matrix3d k = skew(w / angle);
  return Matrix3d::Identity() + std::sin(angle) * k + (1.0 - std::cos(angle)) * k * k;
}

Vector3d log_so3(const Matrix3d& r)
{
  double angle = std::acos(std::clamp((r.trace() - 1.0) / 2.0, -1.0, 1.0));
  Vector3d v(r(2, 1) - r(1, 2), r(0, 2) - r(2, 0), r(1, 0) - r(0, 1));
  if (angle < 1e-9)
    return 0.5 * v;
  return angle / (2.0 * std::sin(angle)) * v;
}

Matrix3d to_matrix(const Mat3& m)
{
  Matrix3d r;
  for (int i = 0; i < 9; i++)
    r(i / 3, i % 3) = m[i];
  return r;
}

}  // namespace

void Eskf::reset(const Transform& pose, double speed)
{
  r_ = to_matrix(pose.r);
  p_ = Vector3d(pose.t[0], pose.t[1], pose.t[2]);
  v_ = r_ * Vector3d(speed, 0.0, 0.0);
  gyro_bias_.setZero();
  acc_bias_.setZero();
  odom_scale_ = 1.0;
  cov_.setZero();
  cov_.block<2, 2>(0, 0) = Eigen::Matrix2d::Identity() * 0.25;
  cov_(2, 2) = 0.04;
  cov_.block<3, 3>(3, 3) = Matrix3d::Identity() * 0.04;
  cov_.block<3, 3>(6, 6) = Matrix3d::Identity() * 1e-3;
  cov_(8, 8) = 0.01;
  cov_.block<3, 3>(9, 9) = Matrix3d::Identity() * 1e-5;
  cov_.block<3, 3>(12, 12) = Matrix3d::Identity() * 0.01;
  cov_(15, 15) = 0.01;
}

void Eskf::predict(const Vector3d& gyro, const Vector3d& acc, double dt, const EskfNoise& noise)
{
  Vector3d w = gyro - gyro_bias_, a = acc - acc_bias_;
  Vector3d world_acc = r_ * a + kGravity;
  Cov f = Cov::Identity();
  f.block<3, 3>(0, 3) = Matrix3d::Identity() * dt;
  f.block<3, 3>(3, 6) = -r_ * skew(a) * dt;
  f.block<3, 3>(3, 12) = -r_ * dt;
  f.block<3, 3>(6, 6) = exp_so3(-w * dt);
  f.block<3, 3>(6, 9) = -Matrix3d::Identity() * dt;
  p_ += v_ * dt + 0.5 * world_acc * dt * dt;
  v_ += world_acc * dt;
  r_ = r_ * exp_so3(w * dt);
  Cov q = Cov::Zero();
  q.block<3, 3>(3, 3) = Matrix3d::Identity() * noise.acc * noise.acc * dt;
  q.block<3, 3>(6, 6) = Matrix3d::Identity() * noise.gyro * noise.gyro * dt;
  q.block<3, 3>(9, 9) = Matrix3d::Identity() * noise.gyro_bias * noise.gyro_bias * dt;
  q.block<3, 3>(12, 12) = Matrix3d::Identity() * noise.acc_bias * noise.acc_bias * dt;
  q(15, 15) = noise.odom_scale * noise.odom_scale * dt;
  cov_ = f * cov_ * f.transpose() + q;
}

template <int M>
double Eskf::correct(const Eigen::Matrix<double, M, kDim>& h, const Eigen::Matrix<double, M, 1>& residual,
                     Eigen::Matrix<double, M, M> noise, double gate, bool inflate)
{
  Eigen::Matrix<double, M, M> s = h * cov_ * h.transpose() + noise;
  double nis = residual.dot(s.ldlt().solve(residual));
  if (gate > 0.0 && nis > gate)
  {
    if (!inflate)
      return nis;
    noise *= nis / gate;
    s = h * cov_ * h.transpose() + noise;
  }
  Eigen::Matrix<double, kDim, M> k = cov_ * h.transpose() * s.inverse();
  Eigen::Matrix<double, kDim, 1> dx = k * residual;
  p_ += dx.template segment<3>(0);
  v_ += dx.template segment<3>(3);
  r_ = r_ * exp_so3(dx.template segment<3>(6));
  gyro_bias_ += dx.template segment<3>(9);
  acc_bias_ += dx.template segment<3>(12);
  odom_scale_ += dx(15);
  Cov ikh = Cov::Identity() - k * h;
  cov_ = ikh * cov_ * ikh.transpose() + k * noise * k.transpose();
  return -nis;
}

double Eskf::update_forward_speed(double speed, double sigma, double gate, bool inflate)
{
  Eigen::Matrix<double, 1, kDim> h = Eigen::Matrix<double, 1, kDim>::Zero();
  Vector3d body = r_.transpose() * v_;
  h.block<1, 3>(0, 3) = odom_scale_ * r_.transpose().row(0);
  h.block<1, 3>(0, 6) = odom_scale_ * skew(body).row(0);
  h(0, 15) = body.x();
  Eigen::Matrix<double, 1, 1> residual, noise;
  residual << speed - odom_scale_ * body.x();
  noise << sigma * sigma;
  return correct<1>(h, residual, noise, gate, inflate);
}

void Eskf::update_lateral_constraint(double sigma)
{
  Eigen::Matrix<double, 2, kDim> h = Eigen::Matrix<double, 2, kDim>::Zero();
  Vector3d body = r_.transpose() * v_;
  Matrix3d rt = r_.transpose(), sk = skew(body);
  h.block<1, 3>(0, 3) = rt.row(1);
  h.block<1, 3>(1, 3) = rt.row(2);
  h.block<1, 3>(0, 6) = sk.row(1);
  h.block<1, 3>(1, 6) = sk.row(2);
  Eigen::Vector2d residual(-body.y(), -body.z());
  Eigen::Matrix2d noise = Eigen::Matrix2d::Identity() * sigma * sigma;
  correct<2>(h, residual, noise, 0.0, false);
}

double Eskf::update_position(const Vector3d& measured, const Vector3d& lever, double sigma_xy, double sigma_z,
                             double gate, bool inflate)
{
  Eigen::Matrix<double, 3, kDim> h = Eigen::Matrix<double, 3, kDim>::Zero();
  h.block<3, 3>(0, 0) = Matrix3d::Identity();
  h.block<3, 3>(0, 6) = -r_ * skew(lever);
  Vector3d residual = measured - (p_ + r_ * lever);
  Matrix3d noise = Vector3d(sigma_xy * sigma_xy, sigma_xy * sigma_xy, sigma_z * sigma_z).asDiagonal();
  return correct<3>(h, residual, noise, gate, inflate);
}

double Eskf::update_pose(const Transform& measured, const Eigen::Matrix<double, 6, 1>& sigma, double gate, bool inflate)
{
  Eigen::Matrix<double, 6, kDim> h = Eigen::Matrix<double, 6, kDim>::Zero();
  h.block<3, 3>(0, 0) = Matrix3d::Identity();
  h.block<3, 3>(3, 6) = Matrix3d::Identity();
  Eigen::Matrix<double, 6, 1> residual;
  residual.segment<3>(0) = Vector3d(measured.t[0], measured.t[1], measured.t[2]) - p_;
  residual.segment<3>(3) = log_so3(r_.transpose() * to_matrix(measured.r));
  Eigen::Matrix<double, 6, 6> noise = sigma.cwiseProduct(sigma).asDiagonal();
  return correct<6>(h, residual, noise, gate, inflate);
}

Transform Eskf::pose() const
{
  Transform t;
  for (int i = 0; i < 9; i++)
    t.r[i] = r_(i / 3, i % 3);
  t.t = { p_.x(), p_.y(), p_.z() };
  return t;
}

double Eskf::forward_speed() const
{
  return (r_.transpose() * v_).x();
}

}  // namespace rsf_3d_localization
