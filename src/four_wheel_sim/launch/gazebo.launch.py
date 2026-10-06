import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from four_wheel_sim.maze import WALLS
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit


def make_world(path='/tmp/four_wheel_maze.world'):
    walls = ''
    for i, (cx, cy, sx, sy) in enumerate(WALLS):
        walls += f"""
    <model name="wall_{i}">
      <static>true</static>
      <pose>{cx} {cy} 0.15 0 0 0</pose>
      <link name="link">
        <collision name="c"><geometry><box><size>{sx} {sy} 0.3</size></box></geometry></collision>
        <visual name="v">
          <geometry><box><size>{sx} {sy} 0.3</size></box></geometry>
          <material><ambient>0.7 0.7 0.7 1</ambient><diffuse>0.7 0.7 0.7 1</diffuse></material>
        </visual>
      </link>
    </model>"""
    sdf = f"""<?xml version="1.0"?>
<sdf version="1.6">
  <world name="maze">
    <include><uri>model://sun</uri></include>
    <include><uri>model://ground_plane</uri></include>{walls}
  </world>
</sdf>"""
    with open(path, 'w') as f:
        f.write(sdf)
    return path


def generate_launch_description():
    pkg = get_package_share_directory('four_wheel_sim')
    robot_desc = xacro.process_file(
        os.path.join(pkg, 'urdf', 'robot.urdf.xacro')).toxml()

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')),
        launch_arguments={'world': make_world()}.items())

    rsp = Node(package='robot_state_publisher', executable='robot_state_publisher',
               parameters=[{'robot_description': robot_desc, 'use_sim_time': True}])
    spawn = Node(package='gazebo_ros', executable='spawn_entity.py', output='screen',
                 arguments=['-topic', 'robot_description', '-entity', 'four_wheel_bot',
                            '-x', '0', '-y', '0', '-z', '0.09'])
    jsb = Node(package='controller_manager', executable='spawner',
               arguments=['joint_state_broadcaster'])
    ddc = Node(package='controller_manager', executable='spawner',
               arguments=['diff_drive_controller'])

    # The Humble diff_drive_controller uses its own topic names; relay them to the standard ones
    cmd_relay = Node(package='topic_tools', executable='relay', name='cmd_vel_relay',
                     arguments=['/cmd_vel', '/diff_drive_controller/cmd_vel_unstamped'])
    odom_relay = Node(package='four_wheel_sim', executable='odom_gt',
                   parameters=[{'use_sim_time': True}])

    return LaunchDescription([
        gazebo, rsp, spawn,
        RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=[jsb])),
        RegisterEventHandler(OnProcessExit(target_action=jsb, on_exit=[ddc])),
        cmd_relay, odom_relay,
    ])