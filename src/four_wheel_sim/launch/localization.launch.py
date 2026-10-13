import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('four_wheel_sim')

    vo = Node(
        package='rtabmap_odom', executable='rgbd_odometry', name='rgbd_odometry',
        output='screen',
        parameters=[{
            'frame_id': 'base_link',
            'odom_frame_id': 'vo',
            'guess_frame_id': '',
            'publish_tf': False,
            'approx_sync': False,
            'wait_for_transform': 0.5,
            'use_sim_time': True,
            'Reg/Force3DoF': 'true',
            'Odom/ResetCountdown': '2',
            'Vis/MinInliers': '12',
            'Vis/MaxFeatures': '1000',
        }],
        remappings=[('rgb/image', '/oak/image_raw'),
                    ('depth/image', '/oak/depth/image_raw'),
                    ('rgb/camera_info', '/oak/camera_info'),
                    ('odom', '/vo')])

    ekf = Node(
        package='robot_localization', executable='ekf_node', name='ekf_filter_node',
        output='screen',
        parameters=[os.path.join(pkg, 'config', 'ekf.yaml')],
        remappings=[('odometry/filtered', '/odom_ekf')])

    return LaunchDescription([vo, ekf])