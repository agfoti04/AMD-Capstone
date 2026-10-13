#!/usr/bin/env python3

import math
import select
import sys
import termios
import time
import tty

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


ARM_JOINTS = (
    'Joint_1',
    'Joint_2',
    'Joint_3',
    'Joint_4',
    'Joint_6',
)
CONTROLLED_JOINTS = ARM_JOINTS + ('Joint_Gripper',)
# Bounded joints only; continuous Joint_1 intentionally has no entry.
JOINT_LIMITS = {
    'Joint_2': (-1.57, 1.57),
    'Joint_3': (-1.57, 1.57),
    'Joint_4': (-1.57, 1.57),
    'Joint_6': (-3.14, 3.14),
    'Joint_Gripper': (-0.5, 0.18),
}
KEY_BINDINGS = {
    'q': ('Joint_1', 1),
    'a': ('Joint_1', -1),
    'w': ('Joint_2', 1),
    's': ('Joint_2', -1),
    'e': ('Joint_3', 1),
    'd': ('Joint_3', -1),
    'r': ('Joint_4', 1),
    'f': ('Joint_4', -1),
    'y': ('Joint_6', 1),
    'h': ('Joint_6', -1),
    'u': ('Joint_Gripper', 1),
    'j': ('Joint_Gripper', -1),
}
STEP_RADIANS = 0.02
MOVE_DURATION_SECONDS = 0.3
STATE_TIMEOUT_SECONDS = 10.0
CONTROLLER_TIMEOUT_SECONDS = 15.0


class ArmTeleop(Node):
    def __init__(self):
        super().__init__('arm_teleop')
        self.joint_positions = {}
        self.arm_targets = None
        self.state_subscriber = self.create_subscription(
            JointState, '/joint_states', self._on_joint_state, 10
        )
        self.arm_publisher = self.create_publisher(
            JointTrajectory, '/arm_controller/joint_trajectory', 10
        )
        self.gripper_publisher = self.create_publisher(
            Float64MultiArray, '/gripper_controller/commands', 10
        )

    def _on_joint_state(self, message):
        for name, position in zip(message.name, message.position):
            if name not in CONTROLLED_JOINTS:
                continue

            limits = JOINT_LIMITS.get(name)
            if limits is None:
                self.joint_positions[name] = position
            else:
                lower, upper = limits
                self.joint_positions[name] = min(max(position, lower), upper)

    def wait_for_robot(self):
        deadline = time.monotonic() + STATE_TIMEOUT_SECONDS
        required_joints = set(CONTROLLED_JOINTS)
        while not required_joints.issubset(self.joint_positions):
            if time.monotonic() >= deadline:
                missing = sorted(required_joints - self.joint_positions.keys())
                raise RuntimeError(
                    'No joint state received for: ' + ', '.join(missing)
                    + '. Start Gazebo and the joint_state_broadcaster first.'
                )
            rclpy.spin_once(self, timeout_sec=0.1)

        self.arm_targets = [self.joint_positions[name] for name in ARM_JOINTS]

        deadline = time.monotonic() + CONTROLLER_TIMEOUT_SECONDS
        while (
            self.arm_publisher.get_subscription_count() == 0
            or self.gripper_publisher.get_subscription_count() == 0
        ):
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    'Controller command topics are not available. '
                    'Check that arm_controller and gripper_controller are active.'
                )
            rclpy.spin_once(self, timeout_sec=0.1)

    def jog(self, joint_name, direction):
        publisher = (
            self.gripper_publisher
            if joint_name == 'Joint_Gripper'
            else self.arm_publisher
        )
        if publisher.get_subscription_count() == 0:
            raise RuntimeError(
                f'No active controller is subscribed for {joint_name}.'
            )

        requested = self.joint_positions[joint_name] + direction * STEP_RADIANS
        limits = JOINT_LIMITS.get(joint_name)
        if limits is None:
            target = requested
        else:
            lower, upper = limits
            target = min(max(requested, lower), upper)
        self.joint_positions[joint_name] = target

        if joint_name == 'Joint_Gripper':
            command = Float64MultiArray()
            command.data = [target]
            self.gripper_publisher.publish(command)
        else:
            joint_index = ARM_JOINTS.index(joint_name)
            self.arm_targets[joint_index] = target
            trajectory = JointTrajectory()
            trajectory.joint_names = list(ARM_JOINTS)
            point = JointTrajectoryPoint()
            point.positions = list(self.arm_targets)
            point.time_from_start.sec = int(MOVE_DURATION_SECONDS)
            point.time_from_start.nanosec = int(
                (MOVE_DURATION_SECONDS % 1) * 1_000_000_000
            )
            trajectory.points = [point]
            self.arm_publisher.publish(trajectory)

        angle = math.degrees(target)
        reached_limit = limits is not None and target != requested
        status = ' (limit reached)' if reached_limit else ''
        print(f'\r{joint_name}: {angle:+.1f} deg{status}     ', end='', flush=True)


def print_controls():
    print('Arm keyboard teleop (each key jogs 0.02 rad, about 1.15 deg)')
    print('Positive / negative joint jogs:')
    print('  Joint_1: q / a     Joint_2: w / s     Joint_3: e / d')
    print('  Joint_4: r / f     Joint_6: y / h')
    print('  Gripper: u / j')
    print('Press x or Ctrl+C to quit. Joint targets are clamped to URDF limits.')


def main():
    if not sys.stdin.isatty():
        print('arm_teleop requires an interactive terminal.', file=sys.stderr)
        return 1

    rclpy.init()
    node = ArmTeleop()
    exit_code = 0
    try:
        node.wait_for_robot()
        print_controls()
        print(
            'Current joint positions received; waiting for jog input.',
            flush=True,
        )
        previous_terminal_settings = termios.tcgetattr(sys.stdin.fileno())
        try:
            tty.setraw(sys.stdin.fileno())
            while rclpy.ok():
                readable, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not readable:
                    rclpy.spin_once(node, timeout_sec=0.0)
                    continue

                key = sys.stdin.read(1)
                if key in ('x', '\x03'):
                    break
                if key in KEY_BINDINGS:
                    node.jog(*KEY_BINDINGS[key])
        finally:
            termios.tcsetattr(
                sys.stdin.fileno(), termios.TCSADRAIN, previous_terminal_settings
            )
            print()
    except KeyboardInterrupt:
        pass
    except RuntimeError as error:
        exit_code = 1
        if str(error):
            print(f'\nTeleop stopped: {error}', file=sys.stderr)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
