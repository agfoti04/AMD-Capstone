"""Start the rock detector on either the simulator camera or the OAK-D."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("rover_perception")
    return LaunchDescription([
        DeclareLaunchArgument(
            "params_file",
            default_value=os.path.join(share, "config", "sim.yaml"),
            description="Use config/oak.yaml for the physical Luxonis camera.",
        ),
        Node(
            package="rover_perception",
            executable="rock_detector",
            output="screen",
            parameters=[LaunchConfiguration("params_file")],
        ),
    ])
