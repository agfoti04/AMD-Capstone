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
    robot_desc = xacro.process_file(
        os.path.join(pkg, 'urdf', 'arm.urdf.xacro')).toxml()
    gazebo_model_path = os.pathsep.join(
        path for path in (os.path.dirname(pkg), os.environ.get('GAZEBO_MODEL_PATH', ''))
        if path
    )

    gazebo = IncludeLaunchDescription(PythonLaunchDescriptionSource(
        os.path.join(get_package_share_directory('gazebo_ros'),
                     'launch', 'gazebo.launch.py')))
    rsp = Node(package='robot_state_publisher', executable='robot_state_publisher',
               parameters=[{'robot_description': robot_desc, 'use_sim_time': True}])
    spawn = Node(package='gazebo_ros', executable='spawn_entity.py', output='screen',
                 arguments=['-topic', 'robot_description', '-entity', 'six_servo_arm'])

    rod = Node(package='gazebo_ros', executable='spawn_entity.py', output='screen',
           arguments=['-file', os.path.join(pkg, 'models', 'rod.sdf'),
                      '-entity', 'rod'])

    

    def spawner(name):
        return Node(
            package='controller_manager',
            executable='spawner',
            output='screen',
            arguments=[
                name,
                '--controller-manager', '/controller_manager',
                '--controller-manager-timeout', '60',
            ],
        )

    jsb, arm, grip = (spawner('joint_state_broadcaster'),
                      spawner('arm_controller'), spawner('gripper_controller'))

    return LaunchDescription([
        SetEnvironmentVariable('GAZEBO_MODEL_PATH', gazebo_model_path),
        gazebo, rsp, spawn, rod,
        RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=[jsb])),
        RegisterEventHandler(OnProcessExit(target_action=jsb, on_exit=[arm])),
        RegisterEventHandler(OnProcessExit(target_action=arm, on_exit=[grip])),
    ])