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

PoseSpread to_spread(const std::vector<double>& v)
{
  PoseSpread s{ 0.0, 0.0, 0.0, 0.0 };
  if (v.size() >= 4)
    s = PoseSpread{ v[0], v[1], v[2], v[3] };
  return s;
}

Vec3 to_vec(const Eigen::Vector3d& v)
{
  return { v.x(), v.y(), v.z() };
}

void level_with_gravity(Pose6& pose, const Eigen::Vector3d& acc)
{
  pose.roll = std::atan2(acc.y(), acc.z());
  pose.pitch = std::atan2(-acc.x(), std::hypot(acc.y(), acc.z()));
}

}  // namespace

Rsf3dLocalization::Rsf3dLocalization(const rclcpp::NodeOptions& options) : Node("rsf_3d_localization", options)
{
  init_param();
  init_interfaces();
  load_map();
  if (!params_.gnss_initialize)
    pending_pose_ = params_.initial_pose;
}

void Rsf3dLocalization::init_param()
{
  init_map_param();
  init_particle_param();
  init_filter_param();
  init_gnss_param();
}

void Rsf3dLocalization::init_map_param()
{
  params_.map_path = this->declare_parameter<std::string>("map_path");
  params_.map_frame = this->declare_parameter<std::string>("map_frame");
  params_.odom_frame = this->declare_parameter<std::string>("odom_frame");
  params_.base_frame = this->declare_parameter<std::string>("base_frame");
  params_.field_resolution = this->declare_parameter<double>("field_resolution");
  params_.field_max_distance = this->declare_parameter<double>("field_max_distance");
  params_.field_threads = this->declare_parameter<int>("field_threads");
  params_.map_visualization_voxel_size = this->declare_parameter<double>("map_visualization_voxel_size");
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
  params_.publish_tf = this->declare_parameter<bool>("publish_tf");
  params_.tf_timeout = this->declare_parameter<double>("tf_timeout");
  params_.imu_wait = this->declare_parameter<double>("imu_wait");
}

void Rsf3dLocalization::init_particle_param()
{
  params_.particle_count = this->declare_parameter<int>("particle_count");
  params_.initial_spread = to_spread(this->declare_parameter<std::vector<double>>("initial_spread"));
  params_.gnss_initialize = this->declare_parameter<bool>("gnss_initialize");
  params_.gnss_init_particle_count = this->declare_parameter<int>("gnss_init_particle_count");
  params_.gnss_init_concentration = this->declare_parameter<double>("gnss_init_concentration");
  params_.gnss_init_min_updates = this->declare_parameter<int>("gnss_init_min_updates");
  params_.motion.along = this->declare_parameter<double>("motion_along");
  params_.motion.along_time = this->declare_parameter<double>("motion_along_time");
  params_.motion.lateral = this->declare_parameter<double>("motion_lateral");
  params_.motion.vertical = this->declare_parameter<double>("motion_vertical");
  params_.motion.yaw_per_d = this->declare_parameter<double>("motion_yaw_per_d");
  params_.motion.yaw_per_a = this->declare_parameter<double>("motion_yaw_per_a");
  params_.motion.tilt = this->declare_parameter<double>("motion_tilt");
  params_.motion.xy_floor = this->declare_parameter<double>("motion_xy_floor");
  params_.motion.yaw_floor = this->declare_parameter<double>("motion_yaw_floor");
  params_.scan.sigma = this->declare_parameter<double>("scan_sigma");
  params_.scan.random_ratio = this->declare_parameter<double>("scan_random_ratio");
  params_.scan.effective_points = this->declare_parameter<double>("scan_effective_points");
  params_.scan.threads = this->declare_parameter<int>("particle_threads");
  params_.resample_ratio = this->declare_parameter<double>("resample_ratio");
  params_.cluster_radius = this->declare_parameter<double>("cluster_radius");
  params_.mass_threshold = this->declare_parameter<double>("mass_threshold");
  params_.fix_reset_radius = this->declare_parameter<double>("fix_reset_radius");
  params_.fix_reset_mass = this->declare_parameter<double>("fix_reset_mass");
  params_.fix_reset_count = this->declare_parameter<int>("fix_reset_count");
  params_.fix_reset_ratio = this->declare_parameter<double>("fix_reset_ratio");
  params_.fix_reset_spread = to_spread(this->declare_parameter<std::vector<double>>("fix_reset_spread"));
  params_.expand_count = this->declare_parameter<int>("expand_count");
  params_.expand_ratio = this->declare_parameter<double>("expand_ratio");
  params_.expand_spread = to_spread(this->declare_parameter<std::vector<double>>("expand_spread"));
}

void Rsf3dLocalization::init_filter_param()
{
  params_.update_min_d = this->declare_parameter<double>("update_min_d");
  params_.update_min_a = this->declare_parameter<double>("update_min_a");
  params_.update_max_dt = this->declare_parameter<double>("update_max_dt");
  params_.alpha_threshold = this->declare_parameter<double>("alpha_threshold");
  params_.pose_gate = this->declare_parameter<double>("pose_gate");
  params_.information_min_gain = this->declare_parameter<double>("information_min_gain");
  std::vector<double> floor = this->declare_parameter<std::vector<double>>("pose_sigma_floor");
  for (int i = 0; i < 6; i++)
    params_.pose_sigma_floor(i) = i < static_cast<int>(floor.size()) ? floor[static_cast<size_t>(i)] : 0.1;
  std::vector<double> rotation = this->declare_parameter<std::vector<double>>("imu_rotation");
  params_.imu_rotation.setIdentity();
  for (int i = 0; i < 9 && i < static_cast<int>(rotation.size()); i++)
    params_.imu_rotation(i / 3, i % 3) = rotation[i];
  params_.imu_decimation = this->declare_parameter<int>("imu_decimation");
  params_.noise.acc = this->declare_parameter<double>("sigma_acc");
  params_.noise.gyro = this->declare_parameter<double>("sigma_gyro");
  params_.noise.acc_bias = this->declare_parameter<double>("sigma_acc_bias");
  params_.noise.gyro_bias = this->declare_parameter<double>("sigma_gyro_bias");
  params_.noise.wheel_scale = this->declare_parameter<double>("sigma_wheel_scale");
  params_.lateral_sigma = this->declare_parameter<double>("lateral_sigma");
  params_.wheel_speed_sigma = this->declare_parameter<double>("wheel_speed_sigma");
  params_.wheel_speed_gate = this->declare_parameter<double>("wheel_speed_gate");
  params_.wheel_max_speed = this->declare_parameter<double>("wheel_max_speed");
  params_.wheel_timeout = this->declare_parameter<double>("wheel_timeout");
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
  params_.particle_fix_sigma_xy = this->declare_parameter<double>("particle_fix_sigma_xy");
  params_.particle_fix_sigma_z = this->declare_parameter<double>("particle_fix_sigma_z");
  params_.particle_fix_outlier = this->declare_parameter<double>("particle_fix_outlier");
  params_.particle_float_sigma_xy = this->declare_parameter<double>("particle_float_sigma_xy");
  params_.particle_float_sigma_z = this->declare_parameter<double>("particle_float_sigma_z");
  params_.particle_float_outlier = this->declare_parameter<double>("particle_float_outlier");
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
  map_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("~/output/pointcloud_map",
                                                                         rclcpp::QoS(1).reliable().transient_local());
  particle_publisher_ = this->create_publisher<geometry_msgs::msg::PoseArray>("~/debug/particles", 1);
  cloud_subscriber_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "~/input/pointcloud", rclcpp::SensorDataQoS(),
      std::bind(&Rsf3dLocalization::on_cloud, this, std::placeholders::_1));
  imu_subscriber_ = this->create_subscription<sensor_msgs::msg::Imu>(
      "~/input/imu", rclcpp::SensorDataQoS().keep_last(2000),
      std::bind(&Rsf3dLocalization::on_imu, this, std::placeholders::_1));
  gnss_subscriber_ = this->create_subscription<sensor_msgs::msg::NavSatFix>(
      "~/input/gnss", rclcpp::SensorDataQoS(), std::bind(&Rsf3dLocalization::on_gnss, this, std::placeholders::_1));
  wheel_subscriber_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "~/input/wheel_odom", rclcpp::SensorDataQoS().keep_last(100),
      std::bind(&Rsf3dLocalization::on_wheel, this, std::placeholders::_1));
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
  field_.build(points, params_.field_resolution, params_.field_max_distance, params_.field_threads);
  build_ground(points);
  publish_map(points);
  double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  RCLCPP_INFO(this->get_logger(), "map loaded: %zu points, %zu field blocks, %.1f s", points.size(),
              field_.block_count(), elapsed);
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
  std::vector<float> floors;
  floors.reserve(ground_.size());
  for (const auto& [key, z] : ground_)
    floors.push_back(z);
  float ground = 0.0f;
  if (!floors.empty())
  {
    std::nth_element(floors.begin(), floors.begin() + floors.size() / 2, floors.end());
    ground = floors[floors.size() / 2];
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
    *z = c[2] / c[3] - ground;
    ++x, ++y, ++z;
  }
  map_publisher_->publish(msg);
  RCLCPP_INFO(this->get_logger(), "map published: %zu points, lowered by %.2f m", cells.size(), ground);
}

void Rsf3dLocalization::build_ground(const std::vector<Point3f>& points)
{
  ground_.clear();
  for (const auto& p : points)
  {
    int64_t column =
        voxel_key(static_cast<int64_t>(std::floor(p.x * 0.5f)), static_cast<int64_t>(std::floor(p.y * 0.5f)), 0);
    auto [it, inserted] = ground_.try_emplace(column, p.z);
    if (!inserted)
      it->second = std::min(it->second, p.z);
  }
}

std::optional<double> Rsf3dLocalization::ground_height(double x, double y) const
{
  auto it =
      ground_.find(voxel_key(static_cast<int64_t>(std::floor(x * 0.5)), static_cast<int64_t>(std::floor(y * 0.5)), 0));
  if (it == ground_.end())
    return std::nullopt;
  return static_cast<double>(it->second);
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
  while (imu_recent_.size() > 1 && imu_recent_.front().stamp < sample.stamp - 1.0)
    imu_recent_.pop_front();
  process_scans();
}

void Rsf3dLocalization::on_gnss(const sensor_msgs::msg::NavSatFix::SharedPtr msg)
{
  if (msg->status.status < sensor_msgs::msg::NavSatStatus::STATUS_FIX || !std::isfinite(msg->latitude) ||
      !std::isfinite(msg->longitude) || !std::isfinite(msg->altitude))
    return;
  bool fix = msg->status.status >= sensor_msgs::msg::NavSatStatus::STATUS_GBAS_FIX &&
             msg->position_covariance[0] <= params_.gnss_max_variance;
  Eigen::Vector3d position((msg->longitude - params_.gnss_origin_lon) * M_PI / 180.0 * gnss_n_,
                           (msg->latitude - params_.gnss_origin_lat) * M_PI / 180.0 * gnss_m_, msg->altitude);
  gnss_queue_.push_back(
      { stamp_of(msg->header.stamp) - params_.gnss_time_offset, position + params_.gnss_offset, fix });
}

void Rsf3dLocalization::on_wheel(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  double speed = msg->twist.twist.linear.x;
  if (!std::isfinite(speed))
    return;
  WheelSample sample{ stamp_of(msg->header.stamp), speed };
  wheel_queue_.insert(std::upper_bound(wheel_queue_.begin(), wheel_queue_.end(), sample.stamp,
                                       [](double t, const WheelSample& w) { return t < w.stamp; }),
                      sample);
  last_wheel_stamp_ = std::max(last_wheel_stamp_, sample.stamp);
  while (wheel_queue_.size() > 200)
    wheel_queue_.pop_front();
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
  if (field_.empty())
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
  scan_queue_.push_back({ stamp.seconds(), base_sensor, odom_base, read_points(*msg) });
  process_scans();
}

void Rsf3dLocalization::process_scans()
{
  while (!scan_queue_.empty())
  {
    PendingScan scan = std::move(scan_queue_.front());
    double latest = imu_recent_.empty() ? -1e18 : imu_recent_.back().stamp;
    if (latest < scan.stamp && scan_queue_.back().stamp - scan.stamp < params_.imu_wait)
      return;
    scan_queue_.pop_front();
    if (pending_pose_ && !initialize(scan.stamp))
      continue;
    if (!initialized_)
    {
      accumulate_cloud(scan, scan.odom_base);
      if (searching_)
        step_search(scan.stamp, scan.odom_base);
      else if (params_.gnss_initialize)
        start_search(scan.stamp, scan.odom_base);
      continue;
    }
    advance(scan.stamp);
    accumulate_cloud(scan, shadow_);
    update_particles(scan.stamp);
    update_output(scan.stamp);
    publish(rclcpp::Time(static_cast<int64_t>(scan.stamp * 1e9), this->get_clock()->get_clock_type()), scan.odom_base);
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

std::vector<Point3f> Rsf3dLocalization::read_points(const sensor_msgs::msg::PointCloud2& msg) const
{
  std::vector<Point3f> points;
  points.reserve(msg.width * msg.height);
  double min2 = params_.range_min * params_.range_min, max2 = params_.range_max * params_.range_max;
  for (sensor_msgs::PointCloud2ConstIterator<float> x(msg, "x"), y(msg, "y"), z(msg, "z"); x != x.end(); ++x, ++y, ++z)
  {
    double r2 = *x * *x + *y * *y + *z * *z;
    if (std::isfinite(r2) && r2 >= min2 && r2 <= max2)
      points.push_back({ *x, *y, *z });
  }
  return points;
}

void Rsf3dLocalization::accumulate_cloud(const PendingScan& scan, const Transform& frame_base)
{
  while (!clouds_.empty() &&
         (clouds_.front().stamp < scan.stamp - params_.scan_accumulate_s || clouds_.front().stamp > scan.stamp))
    clouds_.pop_front();
  Transform frame_sensor = compose(frame_base, scan.base_sensor);
  StampedCloud cloud;
  cloud.stamp = scan.stamp;
  cloud.points.reserve(scan.points.size());
  for (const auto& p : scan.points)
  {
    Vec3 q = apply(frame_sensor, { p.x, p.y, p.z });
    cloud.points.push_back({ static_cast<float>(q[0]), static_cast<float>(q[1]), static_cast<float>(q[2]) });
  }
  clouds_.push_back(std::move(cloud));
}

std::optional<double> Rsf3dLocalization::initial_speed(double t) const
{
  for (auto it = wheel_queue_.rbegin(); it != wheel_queue_.rend(); ++it)
    if (it->stamp <= t + 0.1)
      return t - it->stamp < params_.wheel_timeout ? std::optional<double>(it->speed) : std::nullopt;
  return std::nullopt;
}

std::optional<Eigen::Vector3d> Rsf3dLocalization::mean_acc() const
{
  if (imu_recent_.size() < 50)
    return std::nullopt;
  Eigen::Vector3d acc = Eigen::Vector3d::Zero();
  for (const auto& s : imu_recent_)
    acc += s.acc;
  return Eigen::Vector3d(acc / static_cast<double>(imu_recent_.size()));
}

void Rsf3dLocalization::start_search(double t, const Transform& odom_base)
{
  auto acc = mean_acc();
  if (!acc)
    return;
  std::optional<GnssSample> latest;
  while (!gnss_queue_.empty() && gnss_queue_.front().stamp <= t)
  {
    if (gnss_queue_.front().fix)
      latest = gnss_queue_.front();
    gnss_queue_.pop_front();
  }
  if (!latest)
  {
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "waiting for gnss fix to initialize");
    return;
  }
  Pose6 center;
  center.x = latest->position.x();
  center.y = latest->position.y();
  center.z = ground_height(center.x, center.y).value_or(latest->position.z());
  level_with_gravity(center, *acc);
  particles_.reset_uniform_yaw(to_transform(center), params_.gnss_init_particle_count, params_.initial_spread, 1U);
  search_odom_ = odom_base;
  search_time_ = t;
  search_updates_ = 0;
  searching_ = true;
  RCLCPP_INFO(this->get_logger(), "gnss initialization started at %.2f %.2f", center.x, center.y);
}

void Rsf3dLocalization::step_search(double t, const Transform& odom_base)
{
  Transform delta = compose(inverse(search_odom_), odom_base);
  if (!update_due(to_pose(delta), t - search_time_))
    return;
  std::vector<Point3f> scan = build_scan(odom_base);
  if (scan.empty())
    return;
  particles_.predict(delta, t - search_time_, params_.motion);
  search_odom_ = odom_base;
  search_time_ = t;
  particles_.weight_scan(field_, scan, params_.scan);
  Vec3 lever = to_vec(params_.gnss_lever);
  for (; !gnss_queue_.empty() && gnss_queue_.front().stamp <= t; gnss_queue_.pop_front())
  {
    const GnssSample& sample = gnss_queue_.front();
    if (sample.fix)
      particles_.weight_position(to_vec(sample.position), lever, params_.particle_fix_sigma_xy,
                                 params_.particle_fix_sigma_z, params_.particle_fix_outlier);
  }
  particles_.resample(params_.resample_ratio);
  publish_particles(rclcpp::Time(static_cast<int64_t>(t * 1e9), this->get_clock()->get_clock_type()));
  ParticleEstimate estimate = particles_.estimate(params_.cluster_radius);
  if (++search_updates_ < params_.gnss_init_min_updates ||
      particles_.yaw_concentration() < params_.gnss_init_concentration || estimate.mass < params_.mass_threshold)
    return;
  Pose6 pose = to_pose(estimate.pose);
  RCLCPP_INFO(this->get_logger(), "gnss initialization converged at %.2f %.2f yaw %.3f after %d updates", pose.x,
              pose.y, pose.yaw, search_updates_);
  pending_pose_ = pose;
  initialize(t);
}

bool Rsf3dLocalization::update_due(const Pose6& moved, double dt) const
{
  return std::hypot(moved.x, moved.y) >= params_.update_min_d || std::fabs(moved.yaw) >= params_.update_min_a ||
         dt >= params_.update_max_dt;
}

bool Rsf3dLocalization::initialize(double t)
{
  auto acc = mean_acc();
  if (!acc)
    return false;
  Pose6 pose = *pending_pose_;
  level_with_gravity(pose, *acc);
  auto ground = ground_height(pose.x, pose.y);
  if (std::fabs(pose.z) < 1e-6 && ground)
    pose.z = *ground;
  eskf_.reset(to_transform(pose), initial_speed(t).value_or(0.0));
  particles_.reset(to_transform(pose), params_.particle_count, params_.initial_spread, 1U);
  particle_gnss_.clear();
  clouds_.clear();
  while (!imu_queue_.empty() && imu_queue_.front().stamp <= t)
    imu_queue_.pop_front();
  while (!gnss_queue_.empty() && gnss_queue_.front().stamp <= t)
    gnss_queue_.pop_front();
  filter_time_ = t;
  position_history_.clear();
  last_update_time_ = t;
  last_update_pose_ = eskf_.pose();
  shadow_ = eskf_.pose();
  particle_shadow_ = shadow_;
  rejects_ = low_streak_ = gnss_rejects_ = fix_misses_ = wheel_rejects_ = 0;
  has_output_ = false;
  initialized_ = true;
  searching_ = false;
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
  eskf_.update_lateral_constraint(params_.lateral_sigma);
  update_gnss(t);
  update_wheel(t);
}

void Rsf3dLocalization::update_wheel(double t)
{
  if (t - last_wheel_stamp_ > params_.wheel_timeout)
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "no wheel odometry for %.1f s",
                         t - last_wheel_stamp_);
  while (!wheel_queue_.empty() && wheel_queue_.front().stamp <= t)
  {
    WheelSample sample = wheel_queue_.front();
    wheel_queue_.pop_front();
    if (t - sample.stamp > params_.wheel_timeout || std::fabs(sample.speed) > params_.wheel_max_speed)
      continue;
    double nis = eskf_.update_forward_speed(sample.speed, params_.wheel_speed_sigma, params_.wheel_speed_gate,
                                            wheel_rejects_ >= 2);
    wheel_rejects_ = nis > 0.0 ? wheel_rejects_ + 1 : 0;
  }
}

void Rsf3dLocalization::update_gnss(double t)
{
  while (!gnss_queue_.empty() && gnss_queue_.front().stamp <= t)
  {
    GnssSample sample = gnss_queue_.front();
    gnss_queue_.pop_front();
    auto past = std::lower_bound(position_history_.begin(), position_history_.end(), sample.stamp,
                                 [](const StampedPosition& p, double v) { return p.stamp < v; });
    if (past != position_history_.end() && !position_history_.empty())
      sample.position += position_history_.back().position - past->position;
    particle_gnss_.push_back(sample);
    if (!sample.fix)
      continue;
    double nis = eskf_.update_position(sample.position, params_.gnss_lever, params_.gnss_sigma_xy, params_.gnss_sigma_z,
                                       params_.gnss_gate, gnss_rejects_ >= 2);
    gnss_rejects_ = nis > 0.0 ? gnss_rejects_ + 1 : 0;
  }
}

std::vector<Point3f> Rsf3dLocalization::build_scan(const Transform& frame_base) const
{
  Transform inv = inverse(frame_base);
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

void Rsf3dLocalization::weight_gnss()
{
  Vec3 lever = to_vec(params_.gnss_lever);
  for (const auto& sample : particle_gnss_)
  {
    Vec3 measured = to_vec(sample.position);
    if (!sample.fix)
    {
      particles_.weight_position(measured, lever, params_.particle_float_sigma_xy, params_.particle_float_sigma_z,
                                 params_.particle_float_outlier);
      continue;
    }
    double near = particles_.mass_near(measured, lever, params_.fix_reset_radius);
    fix_misses_ = near < params_.fix_reset_mass ? fix_misses_ + 1 : 0;
    particles_.weight_position(measured, lever, params_.particle_fix_sigma_xy, params_.particle_fix_sigma_z,
                               params_.particle_fix_outlier);
    if (fix_misses_ < params_.fix_reset_count)
      continue;
    Pose6 center = to_pose(particles_.estimate(params_.cluster_radius).pose);
    center.yaw = to_pose(eskf_.pose()).yaw;
    Vec3 offset = mul(to_transform(center).r, lever);
    center.x = measured[0] - offset[0], center.y = measured[1] - offset[1], center.z = measured[2] - offset[2];
    particles_.inject(to_transform(center), params_.fix_reset_ratio, params_.fix_reset_spread);
    fix_misses_ = 0;
    RCLCPP_INFO(this->get_logger(), "fix reset at %.2f %.2f", center.x, center.y);
  }
  particle_gnss_.clear();
}

void Rsf3dLocalization::update_from_particles(const ParticleEstimate& estimate, const ParticleEstimate& prior,
                                              double alpha)
{
  low_streak_ = alpha < params_.alpha_threshold ? low_streak_ + 1 : 0;
  auto gain = information_gain(prior, estimate, params_.information_min_gain);
  if (gain && estimate.mass >= params_.mass_threshold && alpha >= params_.alpha_threshold)
  {
    Eigen::Matrix<double, 6, 1> sigma;
    for (int i = 0; i < 6; i++)
      sigma(i) = std::max(params_.pose_sigma_floor(i), gain->sigma[static_cast<size_t>(i)]);
    double nis = eskf_.update_pose(gain->pose, sigma, params_.pose_gate, rejects_ >= 2);
    rejects_ = nis > 0.0 ? rejects_ + 1 : 0;
  }
  if (low_streak_ >= params_.expand_count)
  {
    particles_.inject(eskf_.pose(), params_.expand_ratio, params_.expand_spread);
    low_streak_ = 0;
    RCLCPP_WARN(this->get_logger(), "low map agreement, particles expanded");
  }
}

void Rsf3dLocalization::update_particles(double t)
{
  Transform current = eskf_.pose();
  if (!update_due(to_pose(compose(inverse(last_update_pose_), current)), t - last_update_time_))
    return;
  std::vector<Point3f> scan = build_scan(shadow_);
  if (scan.empty())
    return;
  particles_.predict(compose(inverse(particle_shadow_), shadow_), t - last_update_time_, params_.motion);
  particle_shadow_ = shadow_;
  std::vector<double> prior_weights = particles_.weights();
  particles_.weight_scan(field_, scan, params_.scan);
  weight_gnss();
  ParticleEstimate estimate = particles_.estimate(params_.cluster_radius);
  ParticleEstimate prior = particles_.estimate_around(estimate.pose, params_.cluster_radius, prior_weights);
  particles_.resample(params_.resample_ratio);
  double alpha = particles_.score(field_, scan, estimate.pose, params_.scan.sigma);
  update_from_particles(estimate, prior, alpha);
  last_update_time_ = t;
  last_update_pose_ = eskf_.pose();
  publish_particles(rclcpp::Time(static_cast<int64_t>(t * 1e9), this->get_clock()->get_clock_type()));
}

void Rsf3dLocalization::publish_particles(const rclcpp::Time& stamp)
{
  if (particle_publisher_->get_subscription_count() == 0)
    return;
  geometry_msgs::msg::PoseArray msg;
  msg.header.stamp = stamp;
  msg.header.frame_id = params_.map_frame;
  msg.poses.reserve(particles_.size());
  for (const auto& pose : particles_.poses())
    msg.poses.push_back(to_msg(pose));
  particle_publisher_->publish(msg);
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
  double limit =
      params_.output_grade * (std::hypot(d.x, d.y) + std::hypot(next.x - before.x, next.y - before.y)) + 0.05 * dt;
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
  if (params_.publish_tf)
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
