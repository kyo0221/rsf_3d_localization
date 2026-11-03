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

#ifndef RSF_3D_LOCALIZATION__PARTICLE_FILTER_HPP_
#define RSF_3D_LOCALIZATION__PARTICLE_FILTER_HPP_

#include "rsf_3d_localization/likelihood_field.hpp"
#include "rsf_3d_localization/pose_math.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <random>
#include <vector>

namespace rsf_3d_localization
{

struct MotionNoise
{
  double along = 0.1;
  double along_time = 0.0;
  double lateral = 0.05;
  double vertical = 0.05;
  double yaw_per_d = 0.02;
  double yaw_per_a = 0.05;
  double tilt = 0.003;
  double xy_floor = 0.02;
  double yaw_floor = 0.005;
};

struct ScanModel
{
  double sigma = 0.2;
  double random_ratio = 0.1;
  double effective_points = 30.0;
  int threads = 4;
};

struct PoseSpread
{
  double xy, z, tilt, yaw;
};

struct ParticleEstimate
{
  Transform pose;
  std::array<double, 6> sigma;
  double mass;
};

std::optional<ParticleEstimate> information_gain(const ParticleEstimate& prior, const ParticleEstimate& posterior,
                                                 double min_gain);

class ParticleFilter
{
public:
  void reset(const Transform& center, int count, const PoseSpread& spread, uint32_t seed);
  void reset_uniform_yaw(const Transform& center, int count, const PoseSpread& spread, uint32_t seed);
  void predict(const Transform& delta, double dt, const MotionNoise& noise);
  double weight_scan(const LikelihoodField& field, const std::vector<Point3f>& scan, const ScanModel& model);
  void weight_position(const Vec3& measured, const Vec3& lever, double sigma_xy, double sigma_z, double outlier);
  double mass_near(const Vec3& measured, const Vec3& lever, double radius) const;
  void inject(const Transform& center, double ratio, const PoseSpread& spread);
  bool resample(double ratio);
  ParticleEstimate estimate(double radius) const;
  ParticleEstimate estimate_around(const Transform& center, double radius, const std::vector<double>& weights) const;
  double score(const LikelihoodField& field, const std::vector<Point3f>& scan, const Transform& pose,
               double sigma) const;
  double yaw_concentration() const;
  size_t size() const;
  const std::vector<Transform>& poses() const;
  const std::vector<double>& weights() const;

private:
  std::vector<Transform> poses_;
  std::vector<double> weights_;
  std::mt19937 rng_;

  Transform sample(const Transform& center, const PoseSpread& spread);
  void apply_log_weights(std::vector<double>& log_weights);
  ParticleEstimate cluster(double cx, double cy, double reach, const std::vector<double>& weights) const;
};

}  // namespace rsf_3d_localization

#endif  // RSF_3D_LOCALIZATION__PARTICLE_FILTER_HPP_
