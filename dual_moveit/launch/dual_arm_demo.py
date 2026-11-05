import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

MOVEIT_DELAY = 15.0


def generate_launch_description():
    arm_pkg = get_package_share_directory('arm_description')
    moveit_pkg = get_package_share_directory('dual_moveit')
    urdf = os.path.join(arm_pkg, 'urdf', 'dual_arm.urdf.xacro')
    moveit_rviz = os.path.join(moveit_pkg, 'config', 'moveit.rviz')

    moveit_config = (
        MoveItConfigsBuilder('dual_arm', package_name='dual_moveit')
        .robot_description(file_path=urdf)
        .trajectory_execution(file_path='config/moveit_controllers.yaml')
        .to_moveit_configs()
    )
    sim_time = {'use_sim_time': True}

    gazebo = IncludeLaunchDescription(PythonLaunchDescriptionSource(
        os.path.join(arm_pkg, 'launch', 'dual_gz_sim.launch.py')))

    moveit = TimerAction(period=MOVEIT_DELAY, actions=[
        Node(package='moveit_ros_move_group', executable='move_group',
             output='screen',
             parameters=[moveit_config.to_dict(), sim_time]),
        Node(package='rviz2', executable='rviz2', output='log',
             arguments=['-d', moveit_rviz],
             parameters=[moveit_config.robot_description,
                         moveit_config.robot_description_semantic,
                         moveit_config.robot_description_kinematics,
                         moveit_config.planning_pipelines,
                         sim_time]),
    ])

    return LaunchDescription([gazebo, moveit])
