import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

from four_wheel_sim.worlds import WALLS, OBSTACLES, TARGETS, LANDMARKS, PATCHES

from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit

def _material(name):
    return f"""<material><script>
            <uri>file://media/materials/scripts/gazebo.material</uri>
            <name>{name}</name></script></material>"""

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


def make_world(path='/tmp/four_wheel_maze.world'):
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
<sdf version="1.6">
  <world name="maze">
    <include><uri>model://sun</uri></include>
    <include><uri>model://ground_plane</uri></include>{m}
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
    head = Node(package='controller_manager', executable='spawner',
                arguments=['head_controller'])

    # The Humble diff_drive_controller uses its own topic names; relay them to the standard ones
    cmd_relay = Node(package='four_wheel_sim', executable='cmd_filter', parameters=[{'use_sim_time': True}])
    odom_relay = Node(package='four_wheel_sim', executable='odom_gt',
                   parameters=[{'use_sim_time': True}])

    return LaunchDescription([
        gazebo, rsp, spawn,
        RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=[jsb])),
        RegisterEventHandler(OnProcessExit(target_action=jsb, on_exit=[ddc])),
        RegisterEventHandler(OnProcessExit(target_action=ddc, on_exit=[head])),
        cmd_relay, odom_relay,
    ])