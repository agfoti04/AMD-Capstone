import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('four_wheel_sim')
    rtab_dir = get_package_share_directory('rtabmap_launch')
    nav2_dir = get_package_share_directory('nav2_bringup')

    # 1. Gazebo, robot, controllers, ground-truth odom
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg, 'launch', 'gazebo.launch.py')))

    # 2. RTAB-Map: publishes /map and map->odom
    rtabmap = TimerAction(period=20.0, actions=[
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(rtab_dir, 'launch', 'rtabmap.launch.py')),
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

    # 3. Nav2, after RTAB-Map has had time to publish the first map
    nav2 = TimerAction(period=15.0, actions=[
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(nav2_dir, 'launch', 'navigation_launch.py')),
            launch_arguments={
                'use_sim_time': 'true',
                'params_file': os.path.join(pkg, 'config', 'nav2_params.yaml'),
                'autostart': 'true',
            }.items())])

    # 4. RViz, optional
    # rviz = TimerAction(period=30.0, actions=[
    #     Node(package='rviz2', executable='rviz2',
    #          parameters=[{'use_sim_time': True}],
    #          condition=IfCondition(LaunchConfiguration('rviz')))])

    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='false'),
        DeclareLaunchArgument('viz', default_value='true'),
        gazebo, rtabmap, nav2,
    ])