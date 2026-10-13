"""Interactive Mars world plus local camera/control bridge, without capture."""
import json
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def setup(context):
    scene = LaunchConfiguration('scene').perform(context)
    with open(scene, encoding='utf-8') as source:
        camera = json.load(source)['config']['camera']
    share = get_package_share_directory('mars_training_environment')
    world = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(share, 'launch', 'mars_world.launch.py')),
        launch_arguments={'scene': scene, 'gui': LaunchConfiguration('gui'), 'capture': 'false'}.items())
    bridge = Node(package='rover_perception', executable='live_bridge', output='screen',
                  parameters=[{'image_topic': camera['image_topic'],
                               'port': int(LaunchConfiguration('bridge_port').perform(context)),
                               'use_sim_time': False}])
    return [world, bridge]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('scene', description='scene.json from a separate live-evaluation bundle'),
        DeclareLaunchArgument('gui', default_value='true'),
        DeclareLaunchArgument('bridge_port', default_value='8770'),
        OpaqueFunction(function=setup)])
