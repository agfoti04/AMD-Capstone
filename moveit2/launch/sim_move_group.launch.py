import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    arm_share = get_package_share_directory('arm_description')
    moveit_share = get_package_share_directory('arm_moveit_config')
    urdf = os.path.join(arm_share, 'urdf', 'arm.urdf.xacro')
    moveit_rviz = os.path.join(moveit_share, 'config', 'moveit.rviz')

    moveit_config = (
        MoveItConfigsBuilder('arm_description', package_name='arm_moveit_config')
        .robot_description(file_path=urdf)
        .to_moveit_configs()
    )
    sim_time = {'use_sim_time': True}

    move_group = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[moveit_config.to_dict(), sim_time],
    )
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        output='log',
        arguments=['-d', moveit_rviz],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            sim_time,
        ],
    )
    return LaunchDescription([move_group, rviz])
