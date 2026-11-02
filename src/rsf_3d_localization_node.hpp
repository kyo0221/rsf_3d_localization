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
#include "rsf_3d_localization/likelihood_grid.hpp"
#include "rsf_3d_localization/ndt_map.hpp"
#include "rsf_3d_localization/pose_math.hpp"

#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2_ros/transform_listener.h"

#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/nav_sat_fix.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

#include <array>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace rsf_3d_localization
{

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

struct StampedTransform
{
  double stamp;
  Transform transform;
};

struct Params
{
  std::string map_path, map_frame, odom_frame, base_frame;
  double map_resolution, ndt_coarse_resolution, map_eigen_ratio;
  int map_min_points, ndt_max_iterations, max_points;
  double ndt_epsilon;
  GridParams grid;
  Pose6 initial_pose;
  double range_min, range_max, scan_accumulate_s, scan_voxel_size;
  double update_min_d, update_min_a, update_max_dt, alpha_threshold, ndt_gate;
  Eigen::Matrix<double, 6, 1> ndt_sigma;
  double search_radius, search_yaw, search_step, search_yaw_step, search_ratio;
  Eigen::Matrix3d imu_rotation;
  int imu_decimation;
  EskfNoise noise;
  double lateral_sigma;
  double odom_speed_period, odom_speed_window, odom_speed_sigma, odom_speed_gate, odom_max_speed,
      odom_max_lateral_speed;
  double restart_stop_s, restart_hold_s;
  double gnss_origin_lat, gnss_origin_lon, gnss_time_offset, gnss_max_variance, gnss_sigma_xy, gnss_sigma_z,
      gnss_gate;
  Eigen::Vector3d gnss_offset, gnss_lever;
  double output_rate_d, output_rate_a, output_grade;
  std::array<double, 36> output_pose_covariance;
  double transform_tolerance, tf_timeout, imu_wait;
};

class Rsf3dLocalization : public rclcpp::Node
{
public:
  explicit Rsf3dLocalization(const rclcpp::NodeOptions& options);

private:
  Params params_;
  NdtMap map_, coarse_map_;
  LikelihoodGrid grid_;
  Eskf eskf_;

  std::deque<StampedCloud> clouds_;
  std::deque<ImuSample> imu_queue_, imu_recent_;
  std::deque<StampedPosition> gnss_queue_, position_history_;
  std::deque<StampedTransform> odom_history_, scan_queue_;
  std::optional<Pose6> pending_pose_;
  bool initialized_ = false;
  long imu_count_ = 0;
  double filter_time_ = 0.0;
  Eigen::Vector3d last_gyro_ = Eigen::Vector3d::Zero(), last_acc_ = Eigen::Vector3d::Zero();

  double last_speed_update_ = -1e18;
  double still_time_ = 0.0, moving_time_ = 0.0, restart_until_ = -1e18;
  int rejects_ = 0, low_streak_ = 0, gnss_rejects_ = 0;
  double last_update_time_ = 0.0;
  Transform last_update_pose_;
  double gnss_n_ = 0.0, gnss_m_ = 0.0;

  Transform shadow_, output_shadow_, output_;
  double output_time_ = 0.0;
  bool has_output_ = false;

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_publisher_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_subscriber_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscriber_;
  rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr gnss_subscriber_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr initial_pose_subscriber_;

  void init_param();
  void init_map_param();
  void init_filter_param();
  void init_gnss_param();
  void init_interfaces();
  void load_map();

  void on_cloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  void on_imu(const sensor_msgs::msg::Imu::SharedPtr msg);
  void on_gnss(const sensor_msgs::msg::NavSatFix::SharedPtr msg);
  void on_initial_pose(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg);

  bool lookup_transforms(const sensor_msgs::msg::PointCloud2& msg, const rclcpp::Time& stamp, Transform& base_sensor,
                         Transform& odom_base);
  void process_scans();
  void accumulate_cloud(const sensor_msgs::msg::PointCloud2& msg, const Transform& odom_sensor, double t);
  bool initialize(double t);
  void advance(double t);
  void predict_step(const Eigen::Vector3d& gyro, const Eigen::Vector3d& acc, double t, double dt);
  void track_restart(double dt);
  std::optional<Eigen::Vector3d> odom_velocity(double t, double& stamp) const;
  void update_speed(double t);
  void update_gnss(double t);
  void update_scan(double t, const Transform& odom_base);
  std::vector<Point3f> build_scan(const Transform& odom_base) const;
  double search(const std::vector<Point3f>& scan, const Transform& current, Transform& estimate, double alpha) const;
  void update_output(double t);
  void publish(const rclcpp::Time& stamp, const Transform& odom_base);
};

}  // namespace rsf_3d_localization

#endif  // RSF_3D_LOCALIZATION_NODE_HPP_
