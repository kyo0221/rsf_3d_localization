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

#include "rsf_3d_localization_node.hpp"

#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <unordered_map>
#include <utility>

namespace rsf_3d_localization
{

namespace
{

Transform from_msg(const geometry_msgs::msg::Transform& m)
{
  Transform t;
  t.r = rotation_from_quaternion(m.rotation.x, m.rotation.y, m.rotation.z, m.rotation.w);
  t.t = { m.translation.x, m.translation.y, m.translation.z };
  return t;
}

geometry_msgs::msg::Pose to_msg(const Transform& t)
{
  Pose6 p = to_pose(t);
  auto q = quaternion_from_rpy(p.roll, p.pitch, p.yaw);
  geometry_msgs::msg::Pose m;
  m.position.x = p.x;
  m.position.y = p.y;
  m.position.z = p.z;
  m.orientation.x = q[0];
  m.orientation.y = q[1];
  m.orientation.z = q[2];
  m.orientation.w = q[3];
  return m;
}

Transform planar(const Transform& t)
{
  Pose6 p = to_pose(t);
  return to_transform(Pose6{ p.x, p.y, 0.0, 0.0, 0.0, p.yaw });
}

bool has_xyz_fields(const sensor_msgs::msg::PointCloud2& msg)
{
  int xyz_fields = 0;
  for (const auto& field : msg.fields)
  {
    if ((field.name == "x" || field.name == "y" || field.name == "z") &&
        field.datatype == sensor_msgs::msg::PointField::FLOAT32)
      xyz_fields++;
  }
  return xyz_fields == 3;
}

double stamp_of(const builtin_interfaces::msg::Time& t)
{
  return rclcpp::Time(t).seconds();
}

}  // namespace

Rsf3dLocalization::Rsf3dLocalization(const rclcpp::NodeOptions& options) : Node("rsf_3d_localization", options)
{
  init_param();
  init_interfaces();
  load_map();
  pending_pose_ = params_.initial_pose;
}

void Rsf3dLocalization::init_param()
{
  init_map_param();
  init_filter_param();
  init_gnss_param();
}

void Rsf3dLocalization::init_map_param()
{
  params_.map_path = this->declare_parameter<std::string>("map_path");
  params_.map_frame = this->declare_parameter<std::string>("map_frame");
  params_.odom_frame = this->declare_parameter<std::string>("odom_frame");
  params_.base_frame = this->declare_parameter<std::string>("base_frame");
  params_.map_resolution = this->declare_parameter<double>("map_resolution");
  params_.ndt_coarse_resolution = this->declare_parameter<double>("ndt_coarse_resolution");
  params_.map_min_points = this->declare_parameter<int>("map_min_points");
  params_.map_eigen_ratio = this->declare_parameter<double>("map_eigen_ratio");
  params_.map_visualization_voxel_size = this->declare_parameter<double>("map_visualization_voxel_size");
  params_.ndt_max_iterations = this->declare_parameter<int>("ndt_max_iterations");
  params_.ndt_epsilon = this->declare_parameter<double>("ndt_epsilon");
  params_.grid.resolution = this->declare_parameter<double>("grid_resolution");
  params_.grid.band_min = this->declare_parameter<double>("grid_band_min");
  params_.grid.band_max = this->declare_parameter<double>("grid_band_max");
  params_.grid.sigma = this->declare_parameter<double>("grid_sigma");
  params_.initial_pose.x = this->declare_parameter<double>("initial_pose_x");
  params_.initial_pose.y = this->declare_parameter<double>("initial_pose_y");
  params_.initial_pose.z = this->declare_parameter<double>("initial_pose_z");
  params_.initial_pose.yaw = this->declare_parameter<double>("initial_pose_yaw");
  params_.range_min = this->declare_parameter<double>("range_min");
  params_.range_max = this->declare_parameter<double>("range_max");
  params_.scan_accumulate_s = this->declare_parameter<double>("scan_accumulate_s");
  params_.scan_voxel_size = this->declare_parameter<double>("scan_voxel_size");
  params_.max_points = this->declare_parameter<int>("max_points");
  params_.output_rate_d = this->declare_parameter<double>("output_rate_d");
  params_.output_rate_a = this->declare_parameter<double>("output_rate_a");
  params_.output_grade = this->declare_parameter<double>("output_grade");
  std::vector<double> covariance = this->declare_parameter<std::vector<double>>("output_pose_covariance");
  params_.output_pose_covariance.fill(0.0);
  std::copy_n(covariance.begin(), std::min<size_t>(covariance.size(), 36), params_.output_pose_covariance.begin());
  params_.transform_tolerance = this->declare_parameter<double>("transform_tolerance");
  params_.tf_timeout = this->declare_parameter<double>("tf_timeout");
  params_.imu_wait = this->declare_parameter<double>("imu_wait");
}

void Rsf3dLocalization::init_filter_param()
{
  params_.update_min_d = this->declare_parameter<double>("update_min_d");
  params_.update_min_a = this->declare_parameter<double>("update_min_a");
  params_.update_max_dt = this->declare_parameter<double>("update_max_dt");
  params_.alpha_threshold = this->declare_parameter<double>("alpha_threshold");
  params_.ndt_gate = this->declare_parameter<double>("ndt_gate");
  std::vector<double> sigma = this->declare_parameter<std::vector<double>>("ndt_sigma");
  for (int i = 0; i < 6; i++)
    params_.ndt_sigma(i) = i < static_cast<int>(sigma.size()) ? sigma[i] : 0.1;
  params_.search_radius = this->declare_parameter<double>("search_radius");
  params_.search_yaw = this->declare_parameter<double>("search_yaw");
  params_.search_step = this->declare_parameter<double>("search_step");
  params_.search_yaw_step = this->declare_parameter<double>("search_yaw_step");
  params_.search_ratio = this->declare_parameter<double>("search_ratio");
  std::vector<double> rotation = this->declare_parameter<std::vector<double>>("imu_rotation");
  params_.imu_rotation.setIdentity();
  for (int i = 0; i < 9 && i < static_cast<int>(rotation.size()); i++)
    params_.imu_rotation(i / 3, i % 3) = rotation[i];
  params_.imu_decimation = this->declare_parameter<int>("imu_decimation");
  params_.noise.acc = this->declare_parameter<double>("sigma_acc");
  params_.noise.gyro = this->declare_parameter<double>("sigma_gyro");
  params_.noise.acc_bias = this->declare_parameter<double>("sigma_acc_bias");
  params_.noise.gyro_bias = this->declare_parameter<double>("sigma_gyro_bias");
  params_.noise.odom_scale = this->declare_parameter<double>("sigma_odom_scale");
  params_.lateral_sigma = this->declare_parameter<double>("lateral_sigma");
  params_.odom_speed_period = this->declare_parameter<double>("odom_speed_period");
  params_.odom_speed_window = this->declare_parameter<double>("odom_speed_window");
  params_.odom_speed_sigma = this->declare_parameter<double>("odom_speed_sigma");
  params_.odom_speed_gate = this->declare_parameter<double>("odom_speed_gate");
  params_.odom_max_speed = this->declare_parameter<double>("odom_max_speed");
  params_.odom_max_lateral_speed = this->declare_parameter<double>("odom_max_lateral_speed");
  params_.restart_stop_s = this->declare_parameter<double>("restart_stop_s");
  params_.restart_hold_s = this->declare_parameter<double>("restart_hold_s");
}

void Rsf3dLocalization::init_gnss_param()
{
  params_.gnss_origin_lat = this->declare_parameter<double>("gnss_origin_lat");
  params_.gnss_origin_lon = this->declare_parameter<double>("gnss_origin_lon");
  std::vector<double> offset = this->declare_parameter<std::vector<double>>("gnss_offset");
  std::vector<double> lever = this->declare_parameter<std::vector<double>>("gnss_lever");
  for (int i = 0; i < 3; i++)
  {
    params_.gnss_offset(i) = i < static_cast<int>(offset.size()) ? offset[i] : 0.0;
    params_.gnss_lever(i) = i < static_cast<int>(lever.size()) ? lever[i] : 0.0;
  }
  params_.gnss_time_offset = this->declare_parameter<double>("gnss_time_offset");
  params_.gnss_max_variance = this->declare_parameter<double>("gnss_max_variance");
  params_.gnss_sigma_xy = this->declare_parameter<double>("gnss_sigma_xy");
  params_.gnss_sigma_z = this->declare_parameter<double>("gnss_sigma_z");
  params_.gnss_gate = this->declare_parameter<double>("gnss_gate");
  const double a = 6378137.0, flattening = 1.0 / 298.257222101, e2 = flattening * (2.0 - flattening);
  double s = std::sin(params_.gnss_origin_lat * M_PI / 180.0);
  gnss_n_ = a / std::sqrt(1.0 - e2 * s * s) * std::cos(params_.gnss_origin_lat * M_PI / 180.0);
  gnss_m_ = a * (1.0 - e2) / std::pow(1.0 - e2 * s * s, 1.5);
}

void Rsf3dLocalization::init_interfaces()
{
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);
  pose_publisher_ =
      this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("~/output/pose_with_covariance", 10);
  map_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "~/output/pointcloud_map", rclcpp::QoS(1).reliable().transient_local());
  cloud_subscriber_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "~/input/pointcloud", rclcpp::SensorDataQoS(),
      std::bind(&Rsf3dLocalization::on_cloud, this, std::placeholders::_1));
  imu_subscriber_ = this->create_subscription<sensor_msgs::msg::Imu>(
      "~/input/imu", rclcpp::SensorDataQoS().keep_last(2000),
      std::bind(&Rsf3dLocalization::on_imu, this, std::placeholders::_1));
  gnss_subscriber_ = this->create_subscription<sensor_msgs::msg::NavSatFix>(
      "~/input/gnss", rclcpp::SensorDataQoS(), std::bind(&Rsf3dLocalization::on_gnss, this, std::placeholders::_1));
  initial_pose_subscriber_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "~/input/initialpose", 1, std::bind(&Rsf3dLocalization::on_initial_pose, this, std::placeholders::_1));
}

void Rsf3dLocalization::load_map()
{
  auto start = std::chrono::steady_clock::now();
  std::vector<Point3f> points;
  std::string error;
  if (!load_pcd(params_.map_path, points, error))
  {
    RCLCPP_ERROR(this->get_logger(), "map load failed: %s", error.c_str());
    return;
  }
  map_.build(points, params_.map_resolution, params_.map_min_points, params_.map_eigen_ratio);
  coarse_map_.build(points, params_.ndt_coarse_resolution, params_.map_min_points, params_.map_eigen_ratio);
  grid_.build(points, params_.grid);
  publish_map(points);
  double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  RCLCPP_INFO(this->get_logger(), "map loaded: %zu points, %zu voxels, %.1f s", points.size(), map_.voxels().size(),
              elapsed);
}

void Rsf3dLocalization::publish_map(const std::vector<Point3f>& points)
{
  float inv_size = static_cast<float>(1.0 / params_.map_visualization_voxel_size);
  std::unordered_map<int64_t, std::array<float, 4>, VoxelKeyHash> cells;
  for (const auto& p : points)
  {
    auto& c = cells[voxel_key(static_cast<int64_t>(std::floor(p.x * inv_size)),
                              static_cast<int64_t>(std::floor(p.y * inv_size)),
                              static_cast<int64_t>(std::floor(p.z * inv_size)))];
    c[0] += p.x, c[1] += p.y, c[2] += p.z, c[3] += 1.0f;
  }
  sensor_msgs::msg::PointCloud2 msg;
  msg.header.frame_id = params_.map_frame;
  msg.header.stamp = this->now();
  sensor_msgs::PointCloud2Modifier modifier(msg);
  modifier.setPointCloud2FieldsByString(1, "xyz");
  modifier.resize(cells.size());
  sensor_msgs::PointCloud2Iterator<float> x(msg, "x"), y(msg, "y"), z(msg, "z");
  for (const auto& [key, c] : cells)
  {
    *x = c[0] / c[3];
    *y = c[1] / c[3];
    *z = c[2] / c[3];
    ++x, ++y, ++z;
  }
  map_publisher_->publish(msg);
  RCLCPP_INFO(this->get_logger(), "map published: %zu points", cells.size());
}

void Rsf3dLocalization::on_imu(const sensor_msgs::msg::Imu::SharedPtr msg)
{
  const auto& w = msg->angular_velocity;
  const auto& a = msg->linear_acceleration;
  ImuSample sample{ stamp_of(msg->header.stamp), params_.imu_rotation * Eigen::Vector3d(w.x, w.y, w.z),
                    params_.imu_rotation * Eigen::Vector3d(a.x, a.y, a.z) };
  imu_queue_.insert(std::upper_bound(imu_queue_.begin(), imu_queue_.end(), sample.stamp,
                                     [](double t, const ImuSample& s) { return t < s.stamp; }),
                    sample);
  imu_recent_.push_back(sample);
  if (++imu_count_ % 10000 == 0)
    RCLCPP_DEBUG(this->get_logger(), "imu %ld stamp %.3f", imu_count_, sample.stamp);
  while (imu_recent_.size() > 1 && imu_recent_.front().stamp < sample.stamp - 1.0)
    imu_recent_.pop_front();
  process_scans();
}

void Rsf3dLocalization::on_gnss(const sensor_msgs::msg::NavSatFix::SharedPtr msg)
{
  if (msg->status.status < sensor_msgs::msg::NavSatStatus::STATUS_FIX ||
      msg->position_covariance[0] > params_.gnss_max_variance || !std::isfinite(msg->latitude))
    return;
  Eigen::Vector3d position((msg->longitude - params_.gnss_origin_lon) * M_PI / 180.0 * gnss_n_,
                           (msg->latitude - params_.gnss_origin_lat) * M_PI / 180.0 * gnss_m_, msg->altitude);
  gnss_queue_.push_back({ stamp_of(msg->header.stamp) - params_.gnss_time_offset, position + params_.gnss_offset });
}

void Rsf3dLocalization::on_initial_pose(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
{
  const auto& p = msg->pose.pose;
  Pose6 pose;
  double roll, pitch;
  rpy_from_rotation(rotation_from_quaternion(p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w), roll,
                    pitch, pose.yaw);
  pose.x = p.position.x;
  pose.y = p.position.y;
  pose.z = p.position.z;
  pending_pose_ = pose;
  RCLCPP_INFO(this->get_logger(), "initial pose: %.2f %.2f %.2f yaw %.3f", pose.x, pose.y, pose.z, pose.yaw);
}

void Rsf3dLocalization::on_cloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  if (map_.empty())
    return;
  if (!has_xyz_fields(*msg))
  {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "point cloud needs float32 x, y, z fields");
    return;
  }
  rclcpp::Time stamp(msg->header.stamp);
  Transform base_sensor, odom_base;
  if (!lookup_transforms(*msg, stamp, base_sensor, odom_base))
    return;
  double t = stamp.seconds();
  if (!odom_history_.empty() && t < odom_history_.back().stamp)
  {
    odom_history_.clear();
    clouds_.clear();
  }
  odom_history_.push_back({ t, odom_base });
  while (odom_history_.front().stamp < t - 2.0)
    odom_history_.pop_front();
  accumulate_cloud(*msg, compose(odom_base, base_sensor), t);
  scan_queue_.push_back({ t, odom_base });
  process_scans();
}

void Rsf3dLocalization::process_scans()
{
  while (!scan_queue_.empty())
  {
    StampedTransform scan = scan_queue_.front();
    double latest = imu_recent_.empty() ? -1e18 : imu_recent_.back().stamp;
    if (latest < scan.stamp && scan_queue_.back().stamp - scan.stamp < params_.imu_wait)
      return;
    scan_queue_.pop_front();
    if (pending_pose_ && !initialize(scan.stamp))
      continue;
    if (!initialized_)
      continue;
    advance(scan.stamp);
    update_scan(scan.stamp, scan.transform);
    update_output(scan.stamp);
    publish(rclcpp::Time(static_cast<int64_t>(scan.stamp * 1e9), this->get_clock()->get_clock_type()), scan.transform);
  }
}

bool Rsf3dLocalization::lookup_transforms(const sensor_msgs::msg::PointCloud2& msg, const rclcpp::Time& stamp,
                                          Transform& base_sensor, Transform& odom_base)
{
  rclcpp::Duration timeout = rclcpp::Duration::from_seconds(params_.tf_timeout);
  if (!tf_buffer_->canTransform(params_.base_frame, msg.header.frame_id, tf2::TimePointZero) ||
      !tf_buffer_->canTransform(params_.odom_frame, params_.base_frame, stamp, timeout))
  {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "no transform from %s or %s to %s",
                         msg.header.frame_id.c_str(), params_.base_frame.c_str(), params_.odom_frame.c_str());
    return false;
  }
  base_sensor =
      from_msg(tf_buffer_->lookupTransform(params_.base_frame, msg.header.frame_id, tf2::TimePointZero).transform);
  odom_base = from_msg(tf_buffer_->lookupTransform(params_.odom_frame, params_.base_frame, stamp).transform);
  return true;
}

void Rsf3dLocalization::accumulate_cloud(const sensor_msgs::msg::PointCloud2& msg, const Transform& odom_sensor,
                                         double t)
{
  while (!clouds_.empty() && clouds_.front().stamp < t - params_.scan_accumulate_s)
    clouds_.pop_front();
  StampedCloud cloud;
  cloud.stamp = t;
  cloud.points.reserve(msg.width * msg.height);
  double min2 = params_.range_min * params_.range_min, max2 = params_.range_max * params_.range_max;
  for (sensor_msgs::PointCloud2ConstIterator<float> x(msg, "x"), y(msg, "y"), z(msg, "z"); x != x.end(); ++x, ++y, ++z)
  {
    double r2 = *x * *x + *y * *y + *z * *z;
    if (!std::isfinite(r2) || r2 < min2 || r2 > max2)
      continue;
    Vec3 p = apply(odom_sensor, { *x, *y, *z });
    cloud.points.push_back({ static_cast<float>(p[0]), static_cast<float>(p[1]), static_cast<float>(p[2]) });
  }
  clouds_.push_back(std::move(cloud));
}

bool Rsf3dLocalization::initialize(double t)
{
  if (imu_recent_.size() < 50)
    return false;
  Eigen::Vector3d acc = Eigen::Vector3d::Zero();
  for (const auto& s : imu_recent_)
    acc += s.acc;
  acc /= static_cast<double>(imu_recent_.size());
  Pose6 pose = *pending_pose_;
  pose.roll = std::atan2(acc.y(), acc.z());
  pose.pitch = std::atan2(-acc.x(), std::hypot(acc.y(), acc.z()));
  if (std::fabs(pose.z) < 1e-6 && std::isfinite(grid_.ground(pose.x, pose.y)))
    pose.z = grid_.ground(pose.x, pose.y);
  double speed_stamp;
  auto velocity = odom_velocity(t, speed_stamp);
  eskf_.reset(to_transform(pose), velocity ? velocity->x() : 0.0);
  while (!imu_queue_.empty() && imu_queue_.front().stamp <= t)
    imu_queue_.pop_front();
  while (!gnss_queue_.empty() && gnss_queue_.front().stamp <= t)
    gnss_queue_.pop_front();
  filter_time_ = t;
  position_history_.clear();
  last_update_time_ = t;
  last_update_pose_ = eskf_.pose();
  shadow_ = eskf_.pose();
  rejects_ = low_streak_ = gnss_rejects_ = 0;
  still_time_ = moving_time_ = 0.0;
  restart_until_ = -1e18;
  has_output_ = false;
  initialized_ = true;
  pending_pose_.reset();
  return true;
}

void Rsf3dLocalization::advance(double t)
{
  Eigen::Vector3d gyro = Eigen::Vector3d::Zero(), acc = Eigen::Vector3d::Zero();
  int n = 0;
  while (!imu_queue_.empty() && imu_queue_.front().stamp <= t)
  {
    ImuSample s = imu_queue_.front();
    imu_queue_.pop_front();
    if (s.stamp <= filter_time_)
      continue;
    gyro += s.gyro;
    acc += s.acc;
    if (++n < params_.imu_decimation)
      continue;
    last_gyro_ = gyro / n;
    last_acc_ = acc / n;
    predict_step(last_gyro_, last_acc_, s.stamp, s.stamp - filter_time_);
    gyro.setZero();
    acc.setZero();
    n = 0;
  }
  if (t > filter_time_ && t - filter_time_ < 0.1)
    predict_step(n > 0 ? Eigen::Vector3d(gyro / n) : last_gyro_, n > 0 ? Eigen::Vector3d(acc / n) : last_acc_, t,
                 t - filter_time_);
  else if (t > filter_time_)
    filter_time_ = t;
}

void Rsf3dLocalization::predict_step(const Eigen::Vector3d& gyro, const Eigen::Vector3d& acc, double t, double dt)
{
  if (dt <= 0.0 || dt > 0.5)
  {
    filter_time_ = std::max(filter_time_, t);
    return;
  }
  Transform before = eskf_.pose();
  eskf_.predict(gyro, acc, dt, params_.noise);
  shadow_ = compose(shadow_, compose(inverse(before), eskf_.pose()));
  filter_time_ = t;
  const Transform& now = eskf_.pose();
  position_history_.push_back({ t, Eigen::Vector3d(now.t[0], now.t[1], now.t[2]) });
  while (position_history_.size() > 1 && position_history_.front().stamp < t - 2.0)
    position_history_.pop_front();
  track_restart(dt);
  eskf_.update_lateral_constraint(params_.lateral_sigma);
  update_gnss(t);
  update_speed(t);
}

void Rsf3dLocalization::track_restart(double dt)
{
  double stamp;
  auto velocity = odom_velocity(filter_time_, stamp);
  bool still = std::fabs(eskf_.forward_speed()) < 0.2 && (!velocity || std::fabs(velocity->x()) < 0.15);
  moving_time_ = still ? 0.0 : moving_time_ + dt;
  if (still)
    still_time_ += dt;
  else if (moving_time_ >= 0.3)
  {
    if (still_time_ > params_.restart_stop_s)
      restart_until_ = filter_time_ + params_.restart_hold_s;
    still_time_ = 0.0;
  }
}

std::optional<Eigen::Vector3d> Rsf3dLocalization::odom_velocity(double t, double& stamp) const
{
  auto end = std::upper_bound(odom_history_.begin(), odom_history_.end(), t,
                              [](double v, const StampedTransform& s) { return v < s.stamp; });
  if (end == odom_history_.begin())
    return std::nullopt;
  auto b = std::prev(end);
  auto a = b;
  while (a != odom_history_.begin() && a->stamp > b->stamp - params_.odom_speed_window)
  {
    auto prev = std::prev(a);
    if (a->stamp - prev->stamp > 0.5)
      return std::nullopt;
    a = prev;
  }
  double dt = b->stamp - a->stamp;
  if (dt < 0.5 * params_.odom_speed_window)
    return std::nullopt;
  const Transform& tb = b->transform;
  Vec3 d = { tb.t[0] - a->transform.t[0], tb.t[1] - a->transform.t[1], tb.t[2] - a->transform.t[2] };
  Vec3 body = mul(transpose(tb.r), d);
  stamp = b->stamp;
  return Eigen::Vector3d(body[0] / dt, body[1] / dt, body[2] / dt);
}

void Rsf3dLocalization::update_speed(double t)
{
  double stamp;
  auto velocity = odom_velocity(t, stamp);
  if (!velocity || stamp - last_speed_update_ < params_.odom_speed_period)
    return;
  if (std::hypot(velocity->y(), velocity->z()) > params_.odom_max_lateral_speed ||
      std::fabs(velocity->x()) > params_.odom_max_speed || t < restart_until_)
    return;
  last_speed_update_ = stamp;
  eskf_.update_forward_speed(velocity->x(), params_.odom_speed_sigma, params_.odom_speed_gate);
}

void Rsf3dLocalization::update_gnss(double t)
{
  while (!gnss_queue_.empty() && gnss_queue_.front().stamp <= t)
  {
    Eigen::Vector3d position = gnss_queue_.front().position;
    double stamp = gnss_queue_.front().stamp;
    gnss_queue_.pop_front();
    auto past = std::lower_bound(position_history_.begin(), position_history_.end(), stamp,
                                 [](const StampedPosition& p, double v) { return p.stamp < v; });
    if (past != position_history_.end() && !position_history_.empty())
      position += position_history_.back().position - past->position;
    double nis = eskf_.update_position(position, params_.gnss_lever, params_.gnss_sigma_xy, params_.gnss_sigma_z,
                                       params_.gnss_gate, gnss_rejects_ >= 2);
    gnss_rejects_ = nis > 0.0 ? gnss_rejects_ + 1 : 0;
    RCLCPP_DEBUG(this->get_logger(), "gnss %.3f lag %.3f nis %.1f", t, t - stamp, std::fabs(nis));
  }
}

std::vector<Point3f> Rsf3dLocalization::build_scan(const Transform& odom_base) const
{
  Transform inv = inverse(odom_base);
  float r[9], o[3];
  for (int i = 0; i < 9; i++)
    r[i] = static_cast<float>(inv.r[i]);
  for (int i = 0; i < 3; i++)
    o[i] = static_cast<float>(inv.t[i]);
  float inv_size = static_cast<float>(1.0 / params_.scan_voxel_size);
  std::unordered_map<int64_t, std::array<float, 4>, VoxelKeyHash> cells;
  for (const auto& cloud : clouds_)
  {
    for (const auto& p : cloud.points)
    {
      float x = r[0] * p.x + r[1] * p.y + r[2] * p.z + o[0];
      float y = r[3] * p.x + r[4] * p.y + r[5] * p.z + o[1];
      float z = r[6] * p.x + r[7] * p.y + r[8] * p.z + o[2];
      auto& c = cells[voxel_key(static_cast<int64_t>(std::floor(x * inv_size)),
                                static_cast<int64_t>(std::floor(y * inv_size)),
                                static_cast<int64_t>(std::floor(z * inv_size)))];
      c[0] += x, c[1] += y, c[2] += z, c[3] += 1.0f;
    }
  }
  std::vector<Point3f> scan;
  scan.reserve(cells.size());
  for (const auto& [key, c] : cells)
    scan.push_back({ c[0] / c[3], c[1] / c[3], c[2] / c[3] });
  size_t n = static_cast<size_t>(params_.max_points);
  if (scan.size() <= n)
    return scan;
  std::vector<Point3f> picked(n);
  for (size_t i = 0; i < n; i++)
    picked[i] = scan[i * scan.size() / n];
  return picked;
}

double Rsf3dLocalization::search(const std::vector<Point3f>& scan, const Transform& current, Transform& estimate,
                                 double alpha) const
{
  std::vector<Point3f> band;
  for (const auto& p : scan)
    if (p.z > params_.grid.band_min && p.z < params_.grid.band_max)
      band.push_back(p);
  if (band.size() < 20)
    return alpha;
  Pose6 c = to_pose(current);
  double radius = std::max(params_.search_radius, 3.0 * std::sqrt(eskf_.position_variance()));
  auto peaks = grid_.search({ c.x, c.y, c.yaw, 0.0 }, radius, params_.search_yaw, params_.search_step,
                            params_.search_yaw_step, band, 3, 1.0);
  std::vector<std::pair<double, Transform>> candidates;
  for (const auto& peak : peaks)
  {
    Pose6 seed = c;
    seed.x = peak.x, seed.y = peak.y, seed.yaw = peak.yaw;
    Transform candidate = to_transform(seed);
    coarse_map_.refine(candidate, scan, params_.ndt_max_iterations, params_.ndt_epsilon);
    candidates.push_back({ map_.refine(candidate, scan, params_.ndt_max_iterations, params_.ndt_epsilon), candidate });
  }
  if (candidates.empty())
    return alpha;
  std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
  double runner_up = 0.0;
  for (size_t i = 1; i < candidates.size(); i++)
  {
    if (std::hypot(candidates[i].second.t[0] - candidates[0].second.t[0],
                   candidates[i].second.t[1] - candidates[0].second.t[1]) > 1.0)
    {
      runner_up = candidates[i].first;
      break;
    }
  }
  if (!(candidates[0].first > map_.score(estimate, scan, true) * 1.1 &&
        candidates[0].first > runner_up * params_.search_ratio))
    return alpha;
  double found = map_.score(candidates[0].second, scan) / static_cast<double>(scan.size());
  if (found <= alpha)
    return alpha;
  estimate = candidates[0].second;
  return found;
}

void Rsf3dLocalization::update_scan(double t, const Transform& odom_base)
{
  Transform current = eskf_.pose();
  Pose6 moved = to_pose(compose(inverse(last_update_pose_), current));
  if (std::hypot(moved.x, moved.y) < params_.update_min_d && std::fabs(moved.yaw) < params_.update_min_a &&
      t - last_update_time_ < params_.update_max_dt)
    return;
  std::vector<Point3f> scan = build_scan(odom_base);
  if (scan.empty())
    return;
  Transform estimate = current;
  coarse_map_.refine(estimate, scan, params_.ndt_max_iterations, params_.ndt_epsilon);
  if (!(map_.refine(estimate, scan, params_.ndt_max_iterations, params_.ndt_epsilon) >
        map_.score(current, scan, true)))
    estimate = current;
  double n = static_cast<double>(scan.size());
  double alpha = map_.score(estimate, scan) / n;
  bool searched = alpha < params_.alpha_threshold && (low_streak_ >= 3 || rejects_ > 0);
  if (searched)
    alpha = search(scan, current, estimate, alpha);
  low_streak_ = alpha < params_.alpha_threshold ? low_streak_ + 1 : 0;
  double nis = 0.0;
  Pose6 e = to_pose(estimate), c = to_pose(current);
  if (alpha >= params_.alpha_threshold)
  {
    nis = eskf_.update_pose(estimate, params_.ndt_sigma, params_.ndt_gate, rejects_ >= 2 || searched);
    rejects_ = nis > 0.0 ? rejects_ + 1 : 0;
  }
  last_update_time_ = t;
  last_update_pose_ = eskf_.pose();
  RCLCPP_DEBUG(this->get_logger(), "scan %.3f alpha %.3f nis %.1f rejects %d searched %d residual %.2f %.2f speed %.2f",
               t, alpha, nis, rejects_, searched, e.x - c.x, e.y - c.y, eskf_.forward_speed());
}

void Rsf3dLocalization::update_output(double t)
{
  Transform state = eskf_.pose();
  if (!has_output_)
  {
    output_ = state;
    output_shadow_ = shadow_;
    output_time_ = t;
    has_output_ = true;
    return;
  }
  Pose6 d = to_pose(compose(inverse(output_shadow_), shadow_));
  d.z = d.roll = d.pitch = 0.0;
  output_ = compose(output_, to_transform(d));
  output_shadow_ = shadow_;
  double dt = std::max(t - output_time_, 0.0);
  output_time_ = t;
  Pose6 c = to_pose(compose(inverse(output_), state));
  double scale = std::min({ 1.0, params_.output_rate_d * dt / std::max(std::hypot(c.x, c.y), 1e-9),
                            params_.output_rate_a * dt / std::max(std::fabs(c.yaw), 1e-9) });
  Pose6 before = to_pose(output_);
  Pose6 step{ c.x * scale, c.y * scale, c.z, c.roll, c.pitch, c.yaw * scale };
  Pose6 next = to_pose(compose(output_, to_transform(step)));
  double limit = params_.output_grade * (std::hypot(d.x, d.y) + std::hypot(next.x - before.x, next.y - before.y)) +
                 0.05 * dt;
  next.z = before.z + std::clamp(next.z - before.z, -limit, limit);
  output_ = to_transform(next);
}

void Rsf3dLocalization::publish(const rclcpp::Time& stamp, const Transform& odom_base)
{
  geometry_msgs::msg::Pose map_odom = to_msg(compose(planar(output_), inverse(planar(odom_base))));
  geometry_msgs::msg::TransformStamped tf_msg;
  tf_msg.header.stamp = stamp + rclcpp::Duration::from_seconds(params_.transform_tolerance);
  tf_msg.header.frame_id = params_.map_frame;
  tf_msg.child_frame_id = params_.odom_frame;
  tf_msg.transform.translation.x = map_odom.position.x;
  tf_msg.transform.translation.y = map_odom.position.y;
  tf_msg.transform.translation.z = map_odom.position.z;
  tf_msg.transform.rotation = map_odom.orientation;
  tf_broadcaster_->sendTransform(tf_msg);
  geometry_msgs::msg::PoseWithCovarianceStamped pose_msg;
  pose_msg.header.stamp = stamp;
  pose_msg.header.frame_id = params_.map_frame;
  pose_msg.pose.pose = to_msg(output_);
  pose_msg.pose.covariance = params_.output_pose_covariance;
  pose_publisher_->publish(pose_msg);
}

}  // namespace rsf_3d_localization

RCLCPP_COMPONENTS_REGISTER_NODE(rsf_3d_localization::Rsf3dLocalization)
