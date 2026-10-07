"""Launch one rock scene and exit after its viewpoints are saved."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    Shutdown,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_setup(context, *args, **kwargs):
    training_share = get_package_share_directory("rover_training")
    scene_id = LaunchConfiguration("scene_id").perform(context)
    seed = int(LaunchConfiguration("seed").perform(context))
    random_scenes = int(LaunchConfiguration("random_scenes").perform(context))
    output_dir = LaunchConfiguration("output_dir").perform(context)
    gui = LaunchConfiguration("gui").perform(context)
    world = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(training_share, "launch", "rocks_world.launch.py")),
        launch_arguments={
            "gui": gui,
            "scene_id": scene_id,
            "seed": str(seed),
            "random_scenes": str(random_scenes),
        }.items(),
    )
    capture = Node(
        package="rover_training",
        executable="capture_dataset",
        output="screen",
        parameters=[{
            "use_sim_time": True,
            "scene_id": scene_id,
            "seed": seed,
            "random_scenes": random_scenes,
            "output_dir": output_dir,
            "plan_path": "/tmp/rover_training_plan.json",
        }],
    )
    return [
        world,
        capture,
        RegisterEventHandler(OnProcessExit(target_action=capture, on_exit=[Shutdown()])),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("gui", default_value="false"),
        DeclareLaunchArgument("seed", default_value="7"),
        DeclareLaunchArgument("random_scenes", default_value="2"),
        DeclareLaunchArgument("scene_id", default_value="scene_three"),
        DeclareLaunchArgument("output_dir", default_value="dataset/rocks"),
        OpaqueFunction(function=_launch_setup),
    ])
