import os

from ament_index_python.packages import get_package_share_directory
from four_wheel_sim.worlds import LANDMARKS, OBSTACLES, PATCHES, TARGETS, WALLS
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import xacro


def _material(name):
    colors = {
        'Gazebo/Blue': '0.1 0.25 0.8 1',
        'Gazebo/Bricks': '0.55 0.22 0.14 1',
        'Gazebo/Grey': '0.5 0.5 0.5 1',
        'Gazebo/Red': '0.8 0.08 0.08 1',
        'Gazebo/Green': '0.08 0.65 0.12 1',
        'Gazebo/Turquoise': '0.05 0.65 0.65 1',
        'Gazebo/Purple': '0.45 0.12 0.6 1',
        'Gazebo/Yellow': '0.9 0.8 0.05 1',
        'Gazebo/Orange': '0.95 0.35 0.05 1',
        'Gazebo/Gold': '0.8 0.6 0.05 1',
        'Gazebo/Grass': '0.2 0.5 0.16 1',
        'Gazebo/Wood': '0.42 0.25 0.12 1',
        'Gazebo/WoodFloor': '0.55 0.35 0.18 1',
    }
    color = colors.get(name, '0.5 0.5 0.5 1')
    return f'<material><ambient>{color}</ambient><diffuse>{color}</diffuse></material>'


def _box_model(name, x, y, z, sx, sy, sz, mat, collide=True):
    col = (f'<collision name="c"><geometry><box><size>{sx} {sy} {sz}</size></box>'
           '</geometry></collision>') if collide else ''
    return f"""
    <model name="{name}"><static>true</static>
      <pose>{x} {y} {z} 0 0 0</pose>
      <link name="link">{col}
        <visual name="v"><geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
          {_material(mat)}</visual>
      </link></model>"""


def _round_model(name, x, y, z, geom, mat):
    return f"""
    <model name="{name}"><static>true</static>
      <pose>{x} {y} {z} 0 0 0</pose>
      <link name="link">
        <collision name="c"><geometry>{geom}</geometry></collision>
        <visual name="v"><geometry>{geom}</geometry>{_material(mat)}</visual>
      </link></model>"""


def make_world(path='/tmp/four_wheel_sim.sdf'):
    m = ''
    for i, (px, py, sx, sy, mat) in enumerate(PATCHES):       # floor texture, no collision
        m += _box_model(f'patch_{i}', px, py, 0.005, sx, sy, 0.01, mat, collide=False)
    for i, (cx, cy, sx, sy) in enumerate(WALLS):
        m += _box_model(f'wall_{i}', cx, cy, 0.15, sx, sy, 0.3, 'Gazebo/Bricks')
    for i, (kind, x, y, a, b, mat) in enumerate(OBSTACLES):
        if kind == 'box':
            m += _box_model(f'obstacle_{i}', x, y, 0.25, a, b, 0.5, mat)
        else:
            g = f'<cylinder><radius>{a}</radius><length>0.5</length></cylinder>'
            m += _round_model(f'obstacle_{i}', x, y, 0.25, g, mat)
    for i, (x, y, mat) in enumerate(LANDMARKS):
        g = '<cylinder><radius>0.1</radius><length>1.5</length></cylinder>'
        m += _round_model(f'landmark_{i}', x, y, 0.75, g, mat)
    for i, (kind, x, y, s, mat) in enumerate(TARGETS):
        if kind == 'box':
            m += _box_model(f'sample_{i}', x, y, s / 2, s, s, s, mat)
        else:
            g = f'<sphere><radius>{s / 2}</radius></sphere>'
            m += _round_model(f'sample_{i}', x, y, s / 2, g, mat)
    sdf = f"""<?xml version="1.0"?>
<sdf version="1.10">
  <world name="maze">
    <scene>
      <ambient>0.8 0.8 0.8 1</ambient>
      <background>0.7 0.7 0.7 1</background>
      <shadows>false</shadows>
    </scene>
    <light type="directional" name="world_light">
      <pose>0 0 10 0 0 0</pose>
      <cast_shadows>false</cast_shadows>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>
    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
          <material>
            <ambient>0.35 0.35 0.35 1</ambient>
            <diffuse>0.35 0.35 0.35 1</diffuse>
          </material>
        </visual>
      </link>
    </model>{m}
  </world>
</sdf>"""
    with open(path, 'w') as f:
        f.write(sdf)
    return path


def generate_launch_description():
    pkg = get_package_share_directory('four_wheel_sim')
    robot_desc = xacro.process_file(
        os.path.join(pkg, 'urdf', 'robot.urdf.xacro')).toxml()

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': f'-r -v 4 {make_world()}'}.items())

    rsp = Node(package='robot_state_publisher', executable='robot_state_publisher',
               parameters=[{'robot_description': robot_desc, 'use_sim_time': True}])
    spawn = Node(package='ros_gz_sim', executable='create', output='screen',
                 arguments=['-name', 'four_wheel_bot', '-topic', 'robot_description',
                            '-x', '0', '-y', '0', '-z', '0.09'])
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{'config_file': os.path.join(pkg, 'config', 'ros_gz_bridge.yaml')}],
        output='screen')
    jsb = Node(package='controller_manager', executable='spawner',
               arguments=['joint_state_broadcaster', '--controller-manager',
                          '/controller_manager'])
    ddc = Node(package='controller_manager', executable='spawner',
               arguments=['diff_drive_controller', '--controller-manager',
                          '/controller_manager'])
    head = Node(package='controller_manager', executable='spawner',
                arguments=['head_controller', '--controller-manager',
                           '/controller_manager'])

    cmd_relay = Node(
        package='four_wheel_sim', executable='cmd_filter',
        parameters=[{'use_sim_time': True}])
    odom_relay = Node(
        package='four_wheel_sim', executable='odom_gt',
        parameters=[{'use_sim_time': True}])

    return LaunchDescription([
        gz_sim, rsp, bridge, spawn,
        RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=[jsb])),
        RegisterEventHandler(OnProcessExit(target_action=jsb, on_exit=[ddc])),
        RegisterEventHandler(OnProcessExit(target_action=ddc, on_exit=[head])),
        cmd_relay, odom_relay,
    ])
