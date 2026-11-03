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

#include "rsf_3d_localization/particle_filter.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <thread>
#include <utility>

namespace rsf_3d_localization
{

namespace
{

std::array<double, 256> hit_table(double max_distance, double sigma)
{
  std::array<double, 256> table;
  for (int l = 0; l < 256; l++)
  {
    double d = l * max_distance / 255.0;
    table[static_cast<size_t>(l)] = l == 255 ? 0.0 : std::exp(-0.5 * d * d / (sigma * sigma));
  }
  return table;
}

void scan_range(const LikelihoodField& field, const std::vector<Point3f>& scan, const std::vector<Transform>& poses,
                const std::array<double, 256>& hit, double random_ratio, size_t begin, size_t end,
                std::vector<double>& log_sum, std::vector<double>& hit_mean)
{
  for (size_t i = begin; i < end; i++)
  {
    const Transform& t = poses[i];
    float r[9], o[3];
    for (int k = 0; k < 9; k++)
      r[k] = static_cast<float>(t.r[static_cast<size_t>(k)]);
    for (int k = 0; k < 3; k++)
      o[k] = static_cast<float>(t.t[static_cast<size_t>(k)]);
    double sum = 0.0, hits = 0.0;
    for (const auto& p : scan)
    {
      float x = r[0] * p.x + r[1] * p.y + r[2] * p.z + o[0];
      float y = r[3] * p.x + r[4] * p.y + r[5] * p.z + o[1];
      float z = r[6] * p.x + r[7] * p.y + r[8] * p.z + o[2];
      double h = hit[field.level(x, y, z)];
      hits += h;
      sum += std::log((1.0 - random_ratio) * h + random_ratio);
    }
    log_sum[i] = sum;
    hit_mean[i] = hits / static_cast<double>(std::max<size_t>(scan.size(), 1));
  }
}

Vec3 antenna(const Transform& t, const Vec3& lever)
{
  Vec3 a = mul(t.r, lever);
  return { t.t[0] + a[0], t.t[1] + a[1], t.t[2] + a[2] };
}

}  // namespace

std::optional<ParticleEstimate> information_gain(const ParticleEstimate& prior, const ParticleEstimate& posterior,
                                                 double min_gain)
{
  if (prior.mass <= 0.0 || posterior.mass <= 0.0)
    return std::nullopt;
  Pose6 p1 = to_pose(posterior.pose), p0 = to_pose(prior.pose);
  double v1[6] = { p1.x, p1.y, p1.z, p1.roll, p1.pitch, p1.yaw };
  double v0[6] = { p0.x, p0.y, p0.z, p0.roll, p0.pitch, p0.yaw };
  double z[6];
  ParticleEstimate result = posterior;
  int informative = 0;
  for (size_t k = 0; k < 6; k++)
  {
    double s1 = std::max(posterior.sigma[k], 1e-4), s0 = std::max(prior.sigma[k], 1e-4);
    double d = k == 5 ? normalize_angle(v1[k] - v0[k]) : v1[k] - v0[k];
    double information = 1.0 / (s1 * s1) - 1.0 / (s0 * s0);
    if (information * s0 * s0 < min_gain)
    {
      z[k] = v0[k] + d;
      result.sigma[k] = 1e3;
      continue;
    }
    double variance = 1.0 / information;
    z[k] = v0[k] + variance * d / (s1 * s1);
    result.sigma[k] = std::sqrt(variance);
    informative++;
  }
  if (informative == 0)
    return std::nullopt;
  result.pose = to_transform(Pose6{ z[0], z[1], z[2], z[3], z[4], z[5] });
  return result;
}

void ParticleFilter::reset(const Transform& center, int count, const PoseSpread& spread, uint32_t seed)
{
  rng_.seed(seed);
  poses_.assign(static_cast<size_t>(std::max(count, 1)), center);
  weights_.assign(poses_.size(), 1.0 / static_cast<double>(poses_.size()));
  for (size_t i = 1; i < poses_.size(); i++)
    poses_[i] = sample(center, spread);
}

void ParticleFilter::reset_uniform_yaw(const Transform& center, int count, const PoseSpread& spread, uint32_t seed)
{
  rng_.seed(seed);
  size_t n = static_cast<size_t>(std::max(count, 1));
  std::uniform_real_distribution<double> jitter(0.0, 1.0);
  Pose6 c = to_pose(center);
  poses_.resize(n);
  weights_.assign(n, 1.0 / static_cast<double>(n));
  for (size_t i = 0; i < n; i++)
  {
    Pose6 p = to_pose(sample(center, spread));
    p.yaw = normalize_angle(c.yaw + 2.0 * M_PI * (static_cast<double>(i) + jitter(rng_)) / static_cast<double>(n));
    poses_[i] = to_transform(p);
  }
}

Transform ParticleFilter::sample(const Transform& center, const PoseSpread& spread)
{
  std::normal_distribution<double> n(0.0, 1.0);
  Pose6 p = to_pose(center);
  p.x += spread.xy * n(rng_);
  p.y += spread.xy * n(rng_);
  p.z += spread.z * n(rng_);
  p.roll += spread.tilt * n(rng_);
  p.pitch += spread.tilt * n(rng_);
  p.yaw += spread.yaw * n(rng_);
  return to_transform(p);
}

void ParticleFilter::predict(const Transform& delta, double dt, const MotionNoise& noise)
{
  std::normal_distribution<double> n(0.0, 1.0);
  Pose6 d = to_pose(delta);
  double travel = std::hypot(d.x, d.y);
  double s_along = noise.along * travel + noise.along_time * std::max(dt, 0.0) + noise.xy_floor,
         s_lateral = noise.lateral * travel + noise.xy_floor;
  double s_yaw = noise.yaw_per_d * travel + noise.yaw_per_a * std::fabs(d.yaw) + noise.yaw_floor;
  for (auto& pose : poses_)
  {
    Pose6 step{ d.x + s_along * n(rng_),       d.y + s_lateral * n(rng_),      d.z + noise.vertical * travel * n(rng_),
                d.roll + noise.tilt * n(rng_), d.pitch + noise.tilt * n(rng_), d.yaw + s_yaw * n(rng_) };
    pose = compose(pose, to_transform(step));
  }
}

void ParticleFilter::apply_log_weights(std::vector<double>& log_weights)
{
  double best = -1e300;
  for (size_t i = 0; i < poses_.size(); i++)
  {
    log_weights[i] += std::log(std::max(weights_[i], 1e-300));
    best = std::max(best, log_weights[i]);
  }
  double total = 0.0;
  for (size_t i = 0; i < poses_.size(); i++)
  {
    weights_[i] = std::exp(log_weights[i] - best);
    total += weights_[i];
  }
  for (auto& w : weights_)
    w /= total;
}

double ParticleFilter::weight_scan(const LikelihoodField& field, const std::vector<Point3f>& scan,
                                   const ScanModel& model)
{
  if (scan.empty() || poses_.empty())
    return 0.0;
  auto hit = hit_table(field.max_distance(), model.sigma);
  std::vector<double> log_sum(poses_.size()), hit_mean(poses_.size());
  size_t workers = static_cast<size_t>(std::max(1, model.threads));
  size_t chunk = (poses_.size() + workers - 1) / workers;
  std::vector<std::thread> pool;
  for (size_t w = 0; w < workers; w++)
  {
    size_t begin = w * chunk, end = std::min(poses_.size(), begin + chunk);
    if (begin < end)
      pool.emplace_back(scan_range, std::cref(field), std::cref(scan), std::cref(poses_), std::cref(hit),
                        model.random_ratio, begin, end, std::ref(log_sum), std::ref(hit_mean));
  }
  for (auto& t : pool)
    t.join();
  double scale = model.effective_points / static_cast<double>(scan.size());
  for (auto& v : log_sum)
    v *= scale;
  apply_log_weights(log_sum);
  return *std::max_element(hit_mean.begin(), hit_mean.end());
}

void ParticleFilter::weight_position(const Vec3& measured, const Vec3& lever, double sigma_xy, double sigma_z,
                                     double outlier)
{
  std::vector<double> log_weights(poses_.size());
  for (size_t i = 0; i < poses_.size(); i++)
  {
    Vec3 a = antenna(poses_[i], lever);
    double dx = (a[0] - measured[0]) / sigma_xy, dy = (a[1] - measured[1]) / sigma_xy;
    double dz = (a[2] - measured[2]) / sigma_z;
    log_weights[i] = std::log((1.0 - outlier) * std::exp(-0.5 * (dx * dx + dy * dy + dz * dz)) + outlier);
  }
  apply_log_weights(log_weights);
}

double ParticleFilter::mass_near(const Vec3& measured, const Vec3& lever, double radius) const
{
  double mass = 0.0;
  for (size_t i = 0; i < poses_.size(); i++)
  {
    Vec3 a = antenna(poses_[i], lever);
    if (std::hypot(a[0] - measured[0], a[1] - measured[1]) < radius)
      mass += weights_[i];
  }
  return mass;
}

void ParticleFilter::inject(const Transform& center, double ratio, const PoseSpread& spread)
{
  size_t count = static_cast<size_t>(std::clamp(ratio, 0.0, 1.0) * static_cast<double>(poses_.size()));
  std::vector<size_t> order(poses_.size());
  std::iota(order.begin(), order.end(), 0);
  std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(count), order.end(),
                    [this](size_t a, size_t b) { return weights_[a] < weights_[b]; });
  double uniform = 1.0 / static_cast<double>(poses_.size());
  for (size_t k = 0; k < count; k++)
  {
    poses_[order[k]] = sample(center, spread);
    weights_[order[k]] = uniform;
  }
  double total = std::accumulate(weights_.begin(), weights_.end(), 0.0);
  for (auto& w : weights_)
    w /= total;
}

bool ParticleFilter::resample(double ratio)
{
  double sum2 = 0.0;
  for (double w : weights_)
    sum2 += w * w;
  if (1.0 / sum2 >= ratio * static_cast<double>(poses_.size()))
    return false;
  std::uniform_real_distribution<double> u(0.0, 1.0 / static_cast<double>(poses_.size()));
  std::vector<Transform> next;
  next.reserve(poses_.size());
  double step = 1.0 / static_cast<double>(poses_.size()), target = u(rng_), cumulative = weights_[0];
  size_t i = 0;
  for (size_t k = 0; k < poses_.size(); k++)
  {
    while (target > cumulative && i + 1 < poses_.size())
      cumulative += weights_[++i];
    next.push_back(poses_[i]);
    target += step;
  }
  poses_.swap(next);
  weights_.assign(poses_.size(), step);
  return true;
}

ParticleEstimate ParticleFilter::estimate(double radius) const
{
  std::map<std::pair<int64_t, int64_t>, double> cells;
  for (size_t i = 0; i < poses_.size(); i++)
    cells[{ static_cast<int64_t>(std::floor(poses_[i].t[0] / radius)),
            static_cast<int64_t>(std::floor(poses_[i].t[1] / radius)) }] += weights_[i];
  std::pair<int64_t, int64_t> best_cell{ 0, 0 };
  double best_mass = -1.0;
  for (const auto& [cell, mass] : cells)
  {
    double sum = 0.0;
    for (int dx = -1; dx <= 1; dx++)
      for (int dy = -1; dy <= 1; dy++)
      {
        auto it = cells.find({ cell.first + dx, cell.second + dy });
        sum += it == cells.end() ? 0.0 : it->second;
      }
    if (sum > best_mass)
      best_mass = sum, best_cell = cell;
  }
  double cx = (static_cast<double>(best_cell.first) + 0.5) * radius;
  double cy = (static_cast<double>(best_cell.second) + 0.5) * radius;
  ParticleEstimate result = cluster(cx, cy, 1.5 * radius, weights_);
  for (int iteration = 0; iteration < 3 && result.mass > 0.0; iteration++)
    result = cluster(result.pose.t[0], result.pose.t[1], radius, weights_);
  return result;
}

ParticleEstimate ParticleFilter::estimate_around(const Transform& center, double radius,
                                                 const std::vector<double>& weights) const
{
  return cluster(center.t[0], center.t[1], radius, weights);
}

ParticleEstimate ParticleFilter::cluster(double cx, double cy, double reach, const std::vector<double>& weights) const
{
  ParticleEstimate result{};
  double m = 0.0, sx = 0.0, sy = 0.0, sz = 0.0, sr = 0.0, sp = 0.0, syc = 0.0, sys = 0.0;
  std::vector<std::pair<size_t, Pose6>> members;
  for (size_t i = 0; i < poses_.size(); i++)
  {
    if (std::hypot(poses_[i].t[0] - cx, poses_[i].t[1] - cy) > reach)
      continue;
    Pose6 p = to_pose(poses_[i]);
    double w = weights[i];
    m += w, sx += w * p.x, sy += w * p.y, sz += w * p.z, sr += w * p.roll, sp += w * p.pitch;
    syc += w * std::cos(p.yaw), sys += w * std::sin(p.yaw);
    members.push_back({ i, p });
  }
  if (m <= 0.0)
    return result;
  Pose6 mean{ sx / m, sy / m, sz / m, sr / m, sp / m, std::atan2(sys, syc) };
  std::array<double, 6> var{};
  for (const auto& [i, p] : members)
  {
    double w = weights[i] / m;
    double e[6] = { p.x - mean.x,       p.y - mean.y,         p.z - mean.z,
                    p.roll - mean.roll, p.pitch - mean.pitch, normalize_angle(p.yaw - mean.yaw) };
    for (int k = 0; k < 6; k++)
      var[static_cast<size_t>(k)] += w * e[k] * e[k];
  }
  for (int k = 0; k < 6; k++)
    result.sigma[static_cast<size_t>(k)] = std::sqrt(var[static_cast<size_t>(k)]);
  result.pose = to_transform(mean);
  result.mass = m;
  return result;
}

double ParticleFilter::score(const LikelihoodField& field, const std::vector<Point3f>& scan, const Transform& pose,
                             double sigma) const
{
  if (scan.empty())
    return 0.0;
  auto hit = hit_table(field.max_distance(), sigma);
  double hits = 0.0;
  for (const auto& p : scan)
  {
    Vec3 q = apply(pose, { p.x, p.y, p.z });
    hits += hit[field.level(static_cast<float>(q[0]), static_cast<float>(q[1]), static_cast<float>(q[2]))];
  }
  return hits / static_cast<double>(scan.size());
}

double ParticleFilter::yaw_concentration() const
{
  double c = 0.0, s = 0.0;
  for (size_t i = 0; i < poses_.size(); i++)
  {
    double yaw = std::atan2(poses_[i].r[3], poses_[i].r[0]);
    c += weights_[i] * std::cos(yaw);
    s += weights_[i] * std::sin(yaw);
  }
  return std::hypot(c, s);
}

size_t ParticleFilter::size() const
{
  return poses_.size();
}

const std::vector<Transform>& ParticleFilter::poses() const
{
  return poses_;
}

const std::vector<double>& ParticleFilter::weights() const
{
  return weights_;
}

}  // namespace rsf_3d_localization
