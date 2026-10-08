"""Gazebo Harmonic (ROS 2 Jazzy) version of gazebo.launch.py.

Same maze, rover, controllers, and ROS topic names as the Classic launch.
Run with gui:=false for a headless server.
"""
import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction,
                            RegisterEventHandler)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from four_wheel_sim.maze import WALLS

# A world that names any system replaces Gazebo's default set, so all are listed.
SYSTEMS = """
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-contact-system" name="gz::sim::systems::Contact"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>"""


def make_world(path='/tmp/four_wheel_maze_harmonic.sdf'):
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
          <material>
            <ambient>0.55 0.25 0.18 1</ambient><diffuse>0.62 0.30 0.22 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
      </link>
    </model>"""
    sdf = f"""<?xml version="1.0"?>
<sdf version="1.9">
  <world name="maze">{SYSTEMS}
    <gravity>0 0 -9.81</gravity>
    <scene><ambient>0.4 0.4 0.4 1</ambient><background>0.7 0.7 0.7 1</background><shadows>true</shadows></scene>
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse><specular>0.2 0.2 0.2 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
          <surface><friction><ode><mu>100</mu><mu2>50</mu2></ode></friction></surface>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient><diffuse>0.8 0.8 0.8 1</diffuse>
            <specular>0.8 0.8 0.8 1</specular>
          </material>
        </visual>
      </link>
    </model>{walls}
  </world>
</sdf>"""
    with open(path, 'w') as f:
        f.write(sdf)
    return path


def _gazebo(context):
    world = make_world()
    gui = LaunchConfiguration('gui').perform(context).lower() in ('1', 'true', 'yes')
    args = f'-r {world}' if gui else f'-r -s --headless-rendering {world}'
    return [IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': args, 'on_exit_shutdown': 'true'}.items())]


def generate_launch_description():
    pkg = get_package_share_directory('four_wheel_sim')
    robot_desc = xacro.process_file(
        os.path.join(pkg, 'urdf', 'robot.urdf.xacro'),
        mappings={'sim': 'harmonic'}).toxml()

    rsp = Node(package='robot_state_publisher', executable='robot_state_publisher',
               parameters=[{'robot_description': robot_desc, 'use_sim_time': True}])
    spawn = Node(package='ros_gz_sim', executable='create', output='screen',
                 arguments=['-topic', 'robot_description', '-name', 'four_wheel_bot',
                            '-x', '0', '-y', '0', '-z', '0.09'])
    bridge = Node(package='ros_gz_bridge', executable='parameter_bridge', output='screen',
                  parameters=[{'config_file': os.path.join(pkg, 'config', 'gz_bridge.yaml'),
                               'use_sim_time': True}])
    jsb = Node(package='controller_manager', executable='spawner',
               arguments=['joint_state_broadcaster'])
    ddc = Node(package='controller_manager', executable='spawner',
               arguments=['diff_drive_controller'])

    stamper = Node(package='four_wheel_sim', executable='twist_stamper',
                   parameters=[{'use_sim_time': True}])
    odom_relay = Node(package='four_wheel_sim', executable='odom_gt',
                      parameters=[{'use_sim_time': True}])

    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='true'),
        OpaqueFunction(function=_gazebo),
        bridge, rsp, spawn,
        RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=[jsb])),
        RegisterEventHandler(OnProcessExit(target_action=jsb, on_exit=[ddc])),
        stamper, odom_relay,
    ])
