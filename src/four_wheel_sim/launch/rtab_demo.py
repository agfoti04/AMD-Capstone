import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('four_wheel_sim')
    rtab_dir = get_package_share_directory('rtabmap_launch')

    # 1. Gazebo, robot, controllers, ground-truth odom
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg, 'launch', 'gazebo.launch.py')))

    # 2. RTAB-Map, delayed so the robot and camera are publishing first
    rtabmap = TimerAction(period=15.0, actions=[
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(rtab_dir, 'launch', 'rtabmap.launch.py')),
            launch_arguments={
                'rtabmap_args': '--delete_db_on_start --Vis/MinInliers 25 '
                                '--RGBD/OptimizeMaxError 1.0',
                'rgb_topic': '/oak/image_raw',
                'depth_topic': '/oak/depth/image_raw',
                'camera_info_topic': '/oak/camera_info',
                'frame_id': 'base_link',
                'odom_topic': '/odom',
                'visual_odometry': 'false',
                'approx_sync': 'true',
                'use_sim_time': 'true',
                'rviz': 'false',
                'rtabmap_viz': 'true',
            }.items())])

    # 3. RViz (rviz:=false to skip)
    rviz = TimerAction(period=10.0, actions=[
        Node(package='rviz2', executable='rviz2',
             parameters=[{'use_sim_time': True}],
             arguments=['-d', os.path.join(pkg, 'config', 'demo.rviz')],
             condition=IfCondition(LaunchConfiguration('rviz')))])

    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='true'),
        gazebo, rtabmap, rviz,
    ])