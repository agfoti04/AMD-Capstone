import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('four_wheel_sim')
    with open(os.path.join(pkg, 'urdf', 'robot.urdf')) as f:
        robot_desc = f.read()

    return LaunchDescription([
        Node(package='robot_state_publisher',
             executable='robot_state_publisher',
             parameters=[{'robot_description': robot_desc}]),
        Node(package='four_wheel_sim',
             executable='kinematic_sim'),
        Node(package='rviz2', executable='rviz2'),
    ])