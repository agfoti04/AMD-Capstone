"""Rock world that reuses the rover description and controllers.

four_wheel_sim/launch/gazebo.launch.py always builds the maze and does not
declare a world argument, so this file starts gzserver itself. The maze launch
is unchanged.
"""

import json
import os
from pathlib import Path

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from rover_training.assets import load_assets
from rover_training.randomize import build_plan, scene_by_id
from rover_training.world_builder import write_world


def _launch_setup(context, *args, **kwargs):
    training_share = Path(get_package_share_directory("rover_training"))
    rover_share = get_package_share_directory("four_wheel_sim")
    gazebo_share = get_package_share_directory("gazebo_ros")
    seed = int(LaunchConfiguration("seed").perform(context))
    scene_id = LaunchConfiguration("scene_id").perform(context)
    random_scenes = int(LaunchConfiguration("random_scenes").perform(context))
    xacro_path = Path(rover_share) / "urdf" / "robot.urdf.xacro"
    assets = load_assets(training_share / "models" / "assets.json")
    plan = build_plan(
        seed, assets, xacro_path.read_text(encoding="utf-8"), random_scenes)
    scene = scene_by_id(plan, scene_id)
    world_path = Path("/tmp") / f"rover_training_{scene_id}.world"
    plan_path = Path("/tmp") / "rover_training_plan.json"
    write_world(world_path, scene, assets)
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    model_path = str(training_share / "models")
    previous = os.environ.get("GAZEBO_MODEL_PATH", "")
    combined = model_path + (os.pathsep + previous if previous else "")
    os.environ["GAZEBO_MODEL_PATH"] = combined

    robot_desc = xacro.process_file(str(xacro_path)).toxml()
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gazebo_share, "launch", "gzserver.launch.py")),
        launch_arguments={"world": str(world_path)}.items(),
    )
    gui = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gazebo_share, "launch", "gzclient.launch.py")),
        condition=IfCondition(LaunchConfiguration("gui")),
    )
    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": robot_desc, "use_sim_time": True}],
    )
    spawn = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        output="screen",
        arguments=["-topic", "robot_description", "-entity", "four_wheel_bot",
                   "-x", "0", "-y", "0", "-z", "0.09"],
    )
    jsb = Node(package="controller_manager", executable="spawner",
               arguments=["joint_state_broadcaster"])
    ddc = Node(package="controller_manager", executable="spawner",
               arguments=["diff_drive_controller"])
    cmd_relay = Node(
        package="topic_tools",
        executable="relay",
        name="cmd_vel_relay",
        arguments=["/cmd_vel", "/diff_drive_controller/cmd_vel_unstamped"],
    )
    odom_relay = Node(
        package="four_wheel_sim",
        executable="odom_gt",
        parameters=[{"use_sim_time": True}],
    )
    return [
        SetEnvironmentVariable("GAZEBO_MODEL_PATH", combined),
        gazebo,
        gui,
        rsp,
        spawn,
        RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=[jsb])),
        RegisterEventHandler(OnProcessExit(target_action=jsb, on_exit=[ddc])),
        cmd_relay,
        odom_relay,
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("gui", default_value="true"),
        DeclareLaunchArgument("seed", default_value="7"),
        DeclareLaunchArgument("random_scenes", default_value="2"),
        DeclareLaunchArgument("scene_id", default_value="scene_three",
                              description="scene_three, scene_empty, scene_clip, scene_aligned, scene_holdout, or scene_random_00"),
        OpaqueFunction(function=_launch_setup),
    ])
