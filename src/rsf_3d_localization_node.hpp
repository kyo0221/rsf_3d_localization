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

#ifndef RSF_3D_LOCALIZATION_NODE_HPP_
#define RSF_3D_LOCALIZATION_NODE_HPP_

#include "rsf_3d_localization/eskf.hpp"
#include "rsf_3d_localization/likelihood_field.hpp"
#include "rsf_3d_localization/particle_filter.hpp"
#include "rsf_3d_localization/pose_math.hpp"
#include "rsf_3d_localization/voxel_key.hpp"

#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2_ros/transform_listener.h"

#include "geometry_msgs/msg/pose_array.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/nav_sat_fix.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

#include <array>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace rsf_3d_localization
{

struct PendingScan
{
  double stamp;
  Transform base_sensor, odom_base;
  std::vector<Point3f> points;
};

struct StampedCloud
{
  double stamp;
  std::vector<Point3f> points;
};

struct ImuSample
{
  double stamp;
  Eigen::Vector3d gyro, acc;
};

struct StampedPosition
{
  double stamp;
  Eigen::Vector3d position;
};

struct GnssSample
{
  double stamp;
  Eigen::Vector3d position;
  bool fix;
};

struct Params
{
  std::string map_path, map_frame, odom_frame, base_frame;
  double field_resolution, field_max_distance;
  int field_threads, particle_count, max_points;
  Pose6 initial_pose;
  bool gnss_initialize;
  int gnss_init_particle_count, gnss_init_min_updates;
  double gnss_init_concentration;
  PoseSpread initial_spread, fix_reset_spread, expand_spread;
  MotionNoise motion;  // 初期化の探索中 (停止中も時間で更新する)
  OdomNoise odom_noise;  // 初期化後: 車輪の距離とジャイロの回転に比例する emcl 方式
  double update_still_d, update_still_a;
  ScanModel scan;
  double map_visualization_voxel_size;
  double range_min, range_max, scan_accumulate_s, scan_voxel_size;
  double update_min_d, update_min_a, update_max_dt, alpha_threshold, mass_threshold, cluster_radius;
  double resample_ratio, pose_gate, information_min_gain;
  Eigen::Matrix<double, 6, 1> pose_sigma_floor;
  double fix_reset_radius, fix_reset_mass, fix_reset_ratio, expand_ratio;
  int fix_reset_count, expand_count;
  Eigen::Matrix3d imu_rotation;
  int imu_decimation;
  EskfNoise noise;
  double lateral_sigma;
  double wheel_speed_sigma, wheel_speed_gate, wheel_max_speed, wheel_timeout;
  double gnss_origin_lat, gnss_origin_lon, gnss_time_offset, gnss_max_variance, gnss_sigma_xy, gnss_sigma_z, gnss_gate;
  double particle_fix_sigma_xy, particle_fix_sigma_z, particle_fix_outlier;
  double particle_float_sigma_xy, particle_float_sigma_z, particle_float_outlier;
  Eigen::Vector3d gnss_offset, gnss_lever;
  double output_rate_d, output_rate_a, output_grade;
  std::array<double, 36> output_pose_covariance;
  double transform_tolerance, tf_timeout, imu_wait;
  bool publish_tf;
};

class Rsf3dLocalization : public rclcpp::Node
{
public:
  explicit Rsf3dLocalization(const rclcpp::NodeOptions& options);

private:
  Params params_;
  LikelihoodField field_;
  std::unordered_map<int64_t, float, VoxelKeyHash> ground_;
  ParticleFilter particles_;
  Eskf eskf_;

  std::deque<StampedCloud> clouds_;
  std::deque<ImuSample> imu_queue_, imu_recent_;
  std::deque<GnssSample> gnss_queue_;
  std::vector<GnssSample> particle_gnss_;
  std::deque<StampedPosition> position_history_;
  std::deque<PendingScan> scan_queue_;
  std::deque<double> wheel_speeds_;
  double wheel_received_ = -1e18;
  double wheel_stamp_ = -1.0;
  double wheel_distance_ = 0.0;  // 前回の粒子の更新からの車輪の走行距離
  std::optional<Pose6> pending_pose_;
  bool initialized_ = false;
  bool searching_ = false;
  Transform search_odom_;
  double search_time_ = 0.0;
  int search_updates_ = 0;
  double filter_time_ = 0.0;
  Eigen::Vector3d last_gyro_ = Eigen::Vector3d::Zero(), last_acc_ = Eigen::Vector3d::Zero();
  int rejects_ = 0, low_streak_ = 0, gnss_rejects_ = 0, fix_misses_ = 0, wheel_rejects_ = 0;
  double last_update_time_ = 0.0;
  double gnss_n_ = 0.0, gnss_m_ = 0.0;

  Transform shadow_, particle_shadow_, output_shadow_, output_;
  double output_time_ = 0.0;
  bool has_output_ = false;

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map_publisher_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr particle_publisher_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_subscriber_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscriber_;
  rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr gnss_subscriber_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr wheel_subscriber_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr initial_pose_subscriber_;

  void init_param();
  void init_map_param();
  void init_particle_param();
  void init_filter_param();
  void init_gnss_param();
  void init_interfaces();
  void load_map();
  void publish_map(const std::vector<Point3f>& points);
  void build_ground(const std::vector<Point3f>& points);
  std::optional<double> ground_height(double x, double y) const;

  void on_cloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  void on_imu(const sensor_msgs::msg::Imu::SharedPtr msg);
  void on_gnss(const sensor_msgs::msg::NavSatFix::SharedPtr msg);
  void on_wheel(const nav_msgs::msg::Odometry::SharedPtr msg);
  void on_initial_pose(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg);

  bool lookup_transforms(const sensor_msgs::msg::PointCloud2& msg, const rclcpp::Time& stamp, Transform& base_sensor,
                         Transform& odom_base);
  void process_scans();
  std::vector<Point3f> read_points(const sensor_msgs::msg::PointCloud2& msg) const;
  void accumulate_cloud(const PendingScan& scan, const Transform& frame_base);
  std::optional<double> initial_speed();
  bool initialize(double t);
  std::optional<Eigen::Vector3d> mean_acc() const;
  void start_search(double t, const Transform& odom_base);
  void step_search(double t, const Transform& odom_base);
  bool update_due(const Pose6& moved, double dt) const;
  void advance(double t);
  void predict_step(const Eigen::Vector3d& gyro, const Eigen::Vector3d& acc, double t, double dt);
  void update_wheel();
  bool wheel_fresh() const;
  void update_gnss(double t);
  void update_particles(double t);
  void weight_gnss();
  void update_from_particles(const ParticleEstimate& estimate, const ParticleEstimate& prior, double alpha);
  std::vector<Point3f> build_scan(const Transform& frame_base) const;
  void publish_particles(const rclcpp::Time& stamp);
  void update_output(double t);
  void publish(const rclcpp::Time& stamp, const Transform& odom_base);
};

}  // namespace rsf_3d_localization

#endif  // RSF_3D_LOCALIZATION_NODE_HPP_
