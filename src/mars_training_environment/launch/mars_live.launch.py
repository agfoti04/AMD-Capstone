"""Interactive Mars world plus local camera/control bridge, without capture."""
import json
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from rover_perception.drive_mode import control_mode, gazebo_bridge_argument


def setup(context):
    mode = control_mode(LaunchConfiguration('control_mode').perform(context))
    scene = LaunchConfiguration('scene').perform(context)
    with open(scene, encoding='utf-8') as source:
        camera = json.load(source)['config']['camera']
    share = get_package_share_directory('mars_training_environment')
    world = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(share, 'launch', 'mars_world.launch.py')),
        launch_arguments={'scene': scene, 'gui': LaunchConfiguration('gui'), 'capture': 'false',
                          'live_camera': 'true', 'live_width': LaunchConfiguration('live_width')}.items())
    bridge = Node(package='rover_perception', executable='live_bridge', output='screen',
                  parameters=[{'image_topic': camera['image_topic'],
                               'port': int(LaunchConfiguration('bridge_port').perform(context)),
                               'camera_timeout_sec': float(LaunchConfiguration('camera_timeout').perform(context)),
                               'control_mode': mode,
                               'use_sim_time': False}])
    actions = [world, bridge]
    if mode == 'gazebo':
        actions += [
            Node(package='ros_gz_bridge', executable='parameter_bridge', name='gazebo_drive_bridge',
                 arguments=[gazebo_bridge_argument()], output='screen'),
            Node(package='rover_perception', executable='gazebo_drive', output='screen',
                 parameters=[{'use_sim_time': False}])]
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('scene', description='scene.json from a separate live-evaluation bundle'),
        DeclareLaunchArgument('gui', default_value='false'),
        DeclareLaunchArgument('live_width', default_value='960'),
        DeclareLaunchArgument('bridge_port', default_value='8770'),
        DeclareLaunchArgument('camera_timeout', default_value='3.0'),
        DeclareLaunchArgument('control_mode', default_value='browser', choices=['browser', 'gazebo']),
        OpaqueFunction(function=setup)])
