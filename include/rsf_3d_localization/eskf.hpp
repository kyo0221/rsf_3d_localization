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

#ifndef RSF_3D_LOCALIZATION__ESKF_HPP_
#define RSF_3D_LOCALIZATION__ESKF_HPP_

#include "rsf_3d_localization/pose_math.hpp"

#include <Eigen/Dense>

namespace rsf_3d_localization
{

struct EskfNoise
{
  double acc = 0.3;
  double gyro = 0.01;
  double gyro_bias = 1e-4;
  double acc_bias = 1e-3;
  double odom_scale = 0.002;
};

class Eskf
{
public:
  static constexpr int kDim = 16;
  using Cov = Eigen::Matrix<double, kDim, kDim>;

  void reset(const Transform& pose, double speed);
  void predict(const Eigen::Vector3d& gyro, const Eigen::Vector3d& acc, double dt, const EskfNoise& noise);
  double update_forward_speed(double speed, double sigma, double gate, bool inflate);
  void update_lateral_constraint(double sigma);
  double update_position(const Eigen::Vector3d& measured, const Eigen::Vector3d& lever, double sigma_xy, double sigma_z,
                         double gate, bool inflate);
  double update_pose(const Transform& measured, const Eigen::Matrix<double, 6, 1>& sigma, double gate, bool inflate);

  Transform pose() const;
  double forward_speed() const;
  double position_variance() const;

private:
  Eigen::Vector3d p_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d v_ = Eigen::Vector3d::Zero();
  Eigen::Matrix3d r_ = Eigen::Matrix3d::Identity();
  Eigen::Vector3d gyro_bias_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d acc_bias_ = Eigen::Vector3d::Zero();
  double odom_scale_ = 1.0;
  Cov cov_ = Cov::Identity();

  template <int M>
  double correct(const Eigen::Matrix<double, M, kDim>& h, const Eigen::Matrix<double, M, 1>& residual,
                 Eigen::Matrix<double, M, M> noise, double gate, bool inflate);
};

}  // namespace rsf_3d_localization

#endif  // RSF_3D_LOCALIZATION__ESKF_HPP_
