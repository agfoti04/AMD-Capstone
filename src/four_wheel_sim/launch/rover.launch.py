import os
os.environ.setdefault('SIM_WORLD', 'survey')   # before any world import

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('four_wheel_sim')
    rtab = get_package_share_directory('rtabmap_launch')

    def include(path, args=None):
        return IncludeLaunchDescription(PythonLaunchDescriptionSource(path),
                                        launch_arguments=(args or {}).items())

    gazebo = include(os.path.join(pkg, 'launch', 'gazebo.launch.py'))

    rtabmap = TimerAction(period=10.0, actions=[
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(rtab, 'launch', 'rtabmap.launch.py')),
            launch_arguments={
                'rtabmap_args': '--delete_db_on_start '
                                '--Vis/MinInliers 25 --RGBD/OptimizeMaxError 1.0 '
                                '--Grid/FromDepth true --Grid/RayTracing true '
                                '--Grid/RangeMax 6.0 --Grid/MaxGroundHeight 0.05 '
                                '--Grid/MaxObstacleHeight 1.0 --Grid/CellSize 0.05',
                'rgb_topic': '/oak/image_raw',
                'depth_topic': '/oak/depth/image_raw',
                'camera_info_topic': '/oak/camera_info',
                'frame_id': 'base_link',
                'odom_topic': '/odom',
                'visual_odometry': 'false',
                'approx_sync': 'true',
                'use_sim_time': 'true',
                'rviz': 'false',
                'rtabmap_viz': LaunchConfiguration('viz'),
            }.items())])

    mission = TimerAction(period=20.0, actions=[Node(
        package='four_wheel_sim', executable='rover_mission', output='screen',
        parameters=[{
            'use_sim_time': True,
            'target_x': 1.5, 'target_y': -1.0,    # grid bottom-left corner
            'grid_w': 2.5, 'grid_h': 2.0,         # lanes at y = -1, 0, 1
            'lane_spacing': 1.0, 'speed': 0.2, 'sweep': True,
        }])])

    return LaunchDescription([
        DeclareLaunchArgument('viz', default_value='true'),
        gazebo, rtabmap, mission,
    ])