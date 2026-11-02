# rsf_3d_localization

IMU・オドメトリ・RTK GNSS と 3D 点群地図 (PCD) への NDT マッチングを誤差状態カルマンフィルタで統合し、`map → odom` を推定する。

```bash
colcon build --packages-select rsf_3d_localization
ros2 launch rsf_3d_localization rsf_3d_localization.launch.xml map_path:=/path/to/map.pcd
```
