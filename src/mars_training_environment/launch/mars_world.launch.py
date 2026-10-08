"""Existing rover and controllers in an additive, pre-generated Mars world."""
import json
import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, RegisterEventHandler, Shutdown
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from mars_training_environment.generate import write_world
from mars_training_environment.robot import robot_description


def setup(context):
    path = Path(LaunchConfiguration('scene').perform(context)).resolve()
    scene = json.loads(path.read_text())
    bundle = path.parents[2]
    assets = json.loads((bundle/'assets/catalog.json').read_text())
    # Re-resolve absolute file URIs when a generated bundle moves between hosts.
    write_world(bundle,scene,assets)
    robot = robot_description(scene['config'])
    (path.parent/'robot_capture.urdf').write_text(robot)
    gazebo_share = get_package_share_directory('gazebo_ros')
    server = IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(gazebo_share,'launch','gzserver.launch.py')),
                                     launch_arguments={'world':str(path.parent/'world.sdf'), 'verbose':'true', 'server_required':'true'}.items())
    gui = IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(gazebo_share,'launch','gzclient.launch.py')),
                                  condition=IfCondition(LaunchConfiguration('gui')))
    rsp = Node(package='robot_state_publisher',executable='robot_state_publisher',parameters=[{'robot_description':robot,'use_sim_time':True}])
    spawn = Node(package='gazebo_ros',executable='spawn_entity.py',output='screen',
                 arguments=['-topic','robot_description','-entity','four_wheel_bot','-x','0','-y','0','-z','0.09'])
    jsb = Node(package='controller_manager',executable='spawner',arguments=['joint_state_broadcaster'])
    ddc = Node(package='controller_manager',executable='spawner',arguments=['diff_drive_controller'])
    relay = Node(package='topic_tools',executable='relay',name='cmd_vel_relay',arguments=['/cmd_vel','/diff_drive_controller/cmd_vel_unstamped'])
    odom = Node(package='four_wheel_sim',executable='odom_gt',parameters=[{'use_sim_time':True}])
    actions = [server,gui,rsp,spawn,RegisterEventHandler(OnProcessExit(target_action=spawn,on_exit=[jsb])),
               RegisterEventHandler(OnProcessExit(target_action=jsb,on_exit=[ddc])),relay,odom]
    if LaunchConfiguration('capture').perform(context).lower() == 'true':
        capture = Node(package='mars_training_environment',executable='mars_capture',output='screen',
                       parameters=[{'use_sim_time':True,'scene':str(path),
                                    'output':LaunchConfiguration('output').perform(context)}])
        actions += [capture,RegisterEventHandler(OnProcessExit(target_action=capture,on_exit=[Shutdown()]))]
    return actions


def generate_launch_description():
    return LaunchDescription([DeclareLaunchArgument('scene',description='Absolute scene.json path from mars_generate'),
                              DeclareLaunchArgument('gui',default_value='true'),
                              DeclareLaunchArgument('capture',default_value='false'),
                              DeclareLaunchArgument('output',default_value='dataset/mars_sample'),
                              OpaqueFunction(function=setup)])
