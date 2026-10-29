import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

# Wall-clock delays (s) from launch start. Raise them if your machine is slow to start Gazebo.
MOVEIT_DELAY = 8.0  # after Gazebo starts, so arm_controller is active
GOTO_DELAY = 15.0   # goto_xyz also waits for the MoveIt services itself


def generate_launch_description():
    arm_pkg = get_package_share_directory('arm_description')
    moveit_pkg = get_package_share_directory('arm_moveit_config')
    urdf = os.path.join(arm_pkg, 'urdf', 'new_arm.urdf.xacro')
    moveit_rviz = os.path.join(moveit_pkg, 'config', 'moveit.rviz')

    moveit_config = (
        MoveItConfigsBuilder('arm_description', package_name='arm_moveit_config')
        .robot_description(file_path=urdf)
        .to_moveit_configs()
    )
    sim_time = {'use_sim_time': True}

    gazebo = IncludeLaunchDescription(PythonLaunchDescriptionSource(
        os.path.join(arm_pkg, 'launch', 'gz_sim.launch.py')))

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
                         sim_time])])

    goto = TimerAction(period=GOTO_DELAY, actions=[
        Node(package='arm_description', executable='goto_xyz',
             output='screen', parameters=[sim_time])])

    return LaunchDescription([gazebo, moveit, goto])