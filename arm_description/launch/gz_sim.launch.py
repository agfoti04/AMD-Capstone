import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('arm_description')
    world = os.path.join(pkg, 'worlds', 'arm_world.sdf')
    robot_desc = xacro.process_file(
        os.path.join(pkg, 'urdf', 'arm.urdf.xacro')).toxml()

    # Lets Gazebo resolve package://arm_description/meshes/...
    resource_path = os.pathsep.join(
        p for p in (os.path.dirname(pkg), os.environ.get('GZ_SIM_RESOURCE_PATH', ''))
        if p)

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': f'-r {world}'}.items())

    rsp = Node(package='robot_state_publisher', executable='robot_state_publisher',
               parameters=[{'robot_description': robot_desc, 'use_sim_time': True}])

    spawn = Node(package='ros_gz_sim', executable='create', output='screen',
                 arguments=['-topic', 'robot_description', '-name', 'six_servo_arm'])

    rod = Node(package='ros_gz_sim', executable='create', output='screen',
               arguments=['-file', os.path.join(pkg, 'models', 'rod.sdf'),
                          '-name', 'rod'])

    clock = Node(package='ros_gz_bridge', executable='parameter_bridge', output='screen',
                 arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'])

    def spawner(name):
        return Node(package='controller_manager', executable='spawner', output='screen',
                    arguments=[name, '--controller-manager', '/controller_manager',
                               '--controller-manager-timeout', '60'])

    jsb, arm, grip = (spawner('joint_state_broadcaster'),
                      spawner('arm_controller'), spawner('gripper_controller'))

    return LaunchDescription([
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', resource_path),
        gazebo, clock, rsp,
        RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=[rod])),
        RegisterEventHandler(OnProcessExit(target_action=rod, on_exit=[jsb])),
        RegisterEventHandler(OnProcessExit(target_action=jsb, on_exit=[arm])),
        RegisterEventHandler(OnProcessExit(target_action=arm, on_exit=[grip])),
        spawn,
    ])