#!/usr/bin/env python3
"""
Custom RTAB-Map launch for depthai_ros_driver_v3 that avoids the onboard
Basalt VIO node (which is early-access/unstable and was crashing with
an IMU timestamp-ordering assertion under WSL2's USB timing jitter).

Instead of consuming /oak/vio/odometry, this launch runs RTAB-Map's own
rgbd_odometry node on the host, computed purely from RGB + depth images.

Usage:
    ros2 launch rtabmap_host_odom.launch.py use_rviz:=true \
        params_file:=/root/oak_configs/rtabmap_host_odom.yaml
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    name = LaunchConfiguration('name')
    params_file = LaunchConfiguration('params_file')
    use_rviz = LaunchConfiguration('use_rviz')
    frame_id = LaunchConfiguration('frame_id')

    declare_name = DeclareLaunchArgument('name', default_value='oak')
    declare_params_file = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(
            get_package_share_directory('depthai_ros_driver_v3'),
            'config', 'rgbd.yaml'
        ),
        description='YAML with driver params. i_enable_vio MUST be false in this file.'
    )
    declare_use_rviz = DeclareLaunchArgument('use_rviz', default_value='true')
    declare_frame_id = DeclareLaunchArgument(
        'frame_id', default_value='oak_parent_frame',
        description='Must match the frame the driver publishes as its base link.'
    )

    # --- Reuse the driver's own driver.launch.py for the camera pipeline ---
    # This gives us the container, the depthai_ros_driver composable node,
    # robot_state_publisher, and URDF publishing, all handled the same way
    # that already worked for pointcloud.launch.py.
    driver_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('depthai_ros_driver_v3'),
                'launch', 'driver.launch.py'
            )
        ),
        launch_arguments={
            'name': name,
            'params_file': params_file,
        }.items()
    )

    # Common remappings from RTAB-Map's expected generic topic names
    # to the actual topics the depthai_ros_driver publishes.
    rgbd_remappings = [
        ('rgb/image', '/oak/rgb/image_raw'),
        ('rgb/camera_info', '/oak/rgb/camera_info'),
        ('depth/image', '/oak/stereo/image_raw'),
    ]

    # --- Host-side visual odometry from RGB + depth (no IMU/VIO involved) ---
    rgbd_odometry_node = Node(
        package='rtabmap_odom',
        executable='rgbd_odometry',
        name='rgbd_odometry',
        output='screen',
        parameters=[{
            'frame_id': frame_id,
            'approx_sync': True,
            'subscribe_rgbd': False,
            'qos_image': 0,
            'qos_camera_info': 0,
            'Odom/ResetCountdown': '0',
        }],
        remappings=rgbd_remappings,
    )

    # --- RTAB-Map SLAM node, now subscribing to /odom instead of onboard VIO ---
    rtabmap_node = Node(
        package='rtabmap_slam',
        executable='rtabmap',
        name='rtabmap',
        output='screen',
        parameters=[{
            'frame_id': frame_id,
            'subscribe_depth': True,
            'subscribe_rgb': True,
            'subscribe_odom': True,
            'approx_sync': True,
            'qos_image': 0,
            'qos_camera_info': 0,
        }],
        remappings=rgbd_remappings,
        arguments=['-d'],  # delete previous database on each run; remove if you want persistence
    )

    rtabmap_viz_node = Node(
        package='rtabmap_viz',
        executable='rtabmap_viz',
        name='rtabmap_viz',
        output='screen',
        parameters=[{
            'frame_id': frame_id,
            'subscribe_depth': True,
            'subscribe_rgb': True,
            'subscribe_odom': True,
            'approx_sync': True,
        }],
        remappings=rgbd_remappings,
        condition=None,  # always launch; drop this node from the list below if undesired
    )

    return LaunchDescription([
        declare_name,
        declare_params_file,
        declare_use_rviz,
        declare_frame_id,
        driver_launch,
        rgbd_odometry_node,
        rtabmap_node,
        rtabmap_viz_node,
    ])