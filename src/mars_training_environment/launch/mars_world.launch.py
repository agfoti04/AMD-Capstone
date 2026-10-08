"""Existing rover and controllers in a pre-generated Mars world (Gazebo Harmonic)."""
import json
import os
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, RegisterEventHandler, Shutdown
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from mars_training_environment.generate import write_world
from mars_training_environment.robot import BOXES_GZ_TOPIC, robot_description


def bridge_config(camera):
    def gz_to_ros(ros, gz, ros_type, gz_type):
        return {'ros_topic_name': ros, 'gz_topic_name': gz, 'ros_type_name': ros_type,
                'gz_type_name': gz_type, 'direction': 'GZ_TO_ROS'}
    return [gz_to_ros('/clock', '/clock', 'rosgraph_msgs/msg/Clock', 'gz.msgs.Clock'),
            gz_to_ros('/odom_gt', '/odom_gt', 'nav_msgs/msg/Odometry', 'gz.msgs.Odometry'),
            gz_to_ros(camera['image_topic'], '/oak/image', 'sensor_msgs/msg/Image', 'gz.msgs.Image'),
            gz_to_ros(camera['depth_topic'], '/oak/depth_image', 'sensor_msgs/msg/Image', 'gz.msgs.Image'),
            gz_to_ros(camera['camera_info_topic'], '/oak/camera_info', 'sensor_msgs/msg/CameraInfo',
                      'gz.msgs.CameraInfo'),
            gz_to_ros(camera['boxes_topic'], '/'+BOXES_GZ_TOPIC, 'vision_msgs/msg/Detection2DArray',
                      'gz.msgs.AnnotatedAxisAligned2DBox_V')]


def setup(context):
    path = Path(LaunchConfiguration('scene').perform(context)).resolve()
    scene = json.loads(path.read_text())
    bundle = path.parents[2]
    assets = json.loads((bundle/'assets/catalog.json').read_text())
    # Re-resolve absolute file paths when a generated bundle moves between hosts.
    write_world(bundle,scene,assets)
    robot = robot_description(scene['config'])
    (path.parent/'robot_capture.urdf').write_text(robot)
    bridge_file = path.parent/'bridge.yaml'
    bridge_file.write_text(yaml.safe_dump(bridge_config(scene['config']['camera'])))
    world = path.parent/'world.sdf'
    gui = LaunchConfiguration('gui').perform(context).lower() in ('1','true','yes')
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(get_package_share_directory('ros_gz_sim'),'launch','gz_sim.launch.py')),
        launch_arguments={'gz_args': f'-r {world}' if gui else f'-r -s --headless-rendering {world}',
                          'on_exit_shutdown':'true'}.items())
    bridge = Node(package='ros_gz_bridge',executable='parameter_bridge',output='screen',
                  parameters=[{'config_file':str(bridge_file),'use_sim_time':True}])
    rsp = Node(package='robot_state_publisher',executable='robot_state_publisher',
               parameters=[{'robot_description':robot,'use_sim_time':True}])
    spawn = Node(package='ros_gz_sim',executable='create',output='screen',
                 arguments=['-topic','robot_description','-name','four_wheel_bot','-x','0','-y','0','-z','0.09'])
    jsb = Node(package='controller_manager',executable='spawner',arguments=['joint_state_broadcaster'])
    ddc = Node(package='controller_manager',executable='spawner',arguments=['diff_drive_controller'])
    stamper = Node(package='four_wheel_sim',executable='twist_stamper',parameters=[{'use_sim_time':True}])
    odom = Node(package='four_wheel_sim',executable='odom_gt',parameters=[{'use_sim_time':True}])
    actions = [gazebo,bridge,rsp,spawn,RegisterEventHandler(OnProcessExit(target_action=spawn,on_exit=[jsb])),
               RegisterEventHandler(OnProcessExit(target_action=jsb,on_exit=[ddc])),stamper,odom]
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
