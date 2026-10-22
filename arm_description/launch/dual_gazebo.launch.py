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

# Spawned one after another, in this order (each starts when the previous one exits).
CONTROLLERS = [
    'joint_state_broadcaster',
    'left_arm_controller',
    'right_arm_controller',
    'left_gripper_controller',
    'right_gripper_controller',
]


def generate_launch_description():
    pkg = get_package_share_directory('arm_description')
    robot_desc = xacro.process_file(
        os.path.join(pkg, 'urdf', 'dual_arm.urdf.xacro')).toxml()
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
                 arguments=['-topic', 'robot_description', '-entity', 'dual_arm'])
    # Same obstacle as before. Delete this node (and `rod` in the list below) to remove it.
    rod = Node(package='gazebo_ros', executable='spawn_entity.py', output='screen',
               arguments=['-file', os.path.join(pkg, 'models', 'rod.sdf'),
                          '-entity', 'rod'])

    def spawner(name):
        return Node(
            package='controller_manager',
            executable='spawner',
            output='screen',
            arguments=[name,
                       '--controller-manager', '/controller_manager',
                       '--controller-manager-timeout', '60'],
        )

    nodes = [spawner(n) for n in CONTROLLERS]
    chain = [RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=[nodes[0]]))]
    for prev, nxt in zip(nodes, nodes[1:]):
        chain.append(RegisterEventHandler(OnProcessExit(target_action=prev, on_exit=[nxt])))

    return LaunchDescription([
        SetEnvironmentVariable('GAZEBO_MODEL_PATH', gazebo_model_path),
        gazebo, rsp, spawn, rod, *chain,
    ])