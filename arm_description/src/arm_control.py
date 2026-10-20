#!/usr/bin/env python3
import math
import time
from collections import deque

import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

ARM_JOINTS = ('Joint_1', 'Joint_2', 'Joint_3', 'Joint_4', 'Joint_6')
GRIPPER = 'Joint_Gripper'
GRIPPER_EFFORT_LIMIT = 1.0
# Keep aligned with the limits in urdf/new_arm.urdf.xacro
LIMITS = {
    'Joint_1': (-1.57, 1.57),
    'Joint_2': (-1.57, 1.57),
    'Joint_3': (-1.57, 1.57),
    'Joint_4': (-1.57, 1.57),
    'Joint_6': (-1.57, 1.57),
}
SPEED = 0.5            # rad/s used to size each move
MIN_MOVE_S = 0.5
TOL = 0.03             # rad, "arrived" tolerance
SETTLE_TIMEOUT = 3.0   # s, wait for joint_states to reach the target


class ArmCommander(Node):
    def __init__(self):
        super().__init__('arm_commander')
        self.pos = {}
        self.arm_targets = None
        self.queue = deque()
        self.create_subscription(JointState, '/joint_states', self._on_state, 10)
        self.create_subscription(JointState, '/arm_goal', self._on_goal, 10)
        self.create_subscription(
            Float64MultiArray, '/gripper_effort', self._on_gripper_effort, 10)
        self.arm_client = ActionClient(
            self, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')
        self.grip_pub = self.create_publisher(
            Float64MultiArray, '/gripper_controller/commands', 10)

    # ---------- callbacks ----------
    def _on_state(self, msg):
        for n, p in zip(msg.name, msg.position):
            self.pos[n] = p

    def _on_goal(self, msg):
        if len(msg.name) != len(msg.position):
            self.get_logger().error('name and position lengths differ, ignoring')
            return
        for n, p in zip(msg.name, msg.position):
            if n == GRIPPER:
                self.get_logger().error(
                    'Gripper position goals are no longer supported; '
                    'publish effort to /gripper_effort')
                continue
            if n not in LIMITS:
                self.get_logger().error(f'Unknown joint "{n}", skipping')
                continue
            self.queue.append((n, float(p)))
        self.get_logger().info(f'Queued {len(self.queue)} move(s)')

    def _on_gripper_effort(self, msg):
        if len(msg.data) != 1 or not math.isfinite(msg.data[0]):
            self.get_logger().error(
                '/gripper_effort must contain exactly one finite effort value')
            return
        effort = min(max(msg.data[0], -GRIPPER_EFFORT_LIMIT), GRIPPER_EFFORT_LIMIT)
        if effort != msg.data[0]:
            self.get_logger().warn(
                f'Gripper effort {msg.data[0]:.2f} outside '
                f'[-{GRIPPER_EFFORT_LIMIT:.1f}, {GRIPPER_EFFORT_LIMIT:.1f}], '
                f'using {effort:.2f}')
        self.grip_pub.publish(Float64MultiArray(data=[effort]))

    # ---------- helpers ----------
    def wait_ready(self):
        self.get_logger().info('Waiting for joint states and controllers...')
        while rclpy.ok() and not set(ARM_JOINTS).issubset(self.pos):
            rclpy.spin_once(self, timeout_sec=0.1)
        self.arm_client.wait_for_server()
        while rclpy.ok() and self.grip_pub.get_subscription_count() == 0:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.arm_targets = [self.pos[j] for j in ARM_JOINTS]
        self.get_logger().info(
            'Ready. Publish arm positions to /arm_goal and one signed effort '
            'value to /gripper_effort (positive closes, negative opens; zero stops).')

    def settle(self, name, target, timeout):
        end = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            if abs(self.pos[name] - target) < TOL:
                return True
        return False

    # ---------- motion ----------
    def move_joint(self, name, target):
        lower, upper = LIMITS[name]
        clamped = min(max(target, lower), upper)
        if clamped != target:
            self.get_logger().warn(
                f'{name}: {target:.2f} outside [{lower}, {upper}], using {clamped:.2f}')
        target = clamped
        self.get_logger().info(f'{name}: {self.pos[name]:+.2f} -> {target:+.2f} rad')

        idx = ARM_JOINTS.index(name)
        duration = max(MIN_MOVE_S, abs(target - self.pos[name]) / SPEED)
        self.arm_targets[idx] = target

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory(joint_names=list(ARM_JOINTS))
        pt = JointTrajectoryPoint(positions=list(self.arm_targets))
        pt.time_from_start.sec = int(duration)
        pt.time_from_start.nanosec = int((duration % 1) * 1e9)
        goal.trajectory.points = [pt]

        send = self.arm_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send, timeout_sec=5.0)
        handle = send.result()
        if handle is None or not handle.accepted:
            self.get_logger().error(f'{name}: goal rejected')
            self.arm_targets[idx] = self.pos[name]
            return
        res = handle.get_result_async()
        rclpy.spin_until_future_complete(self, res, timeout_sec=duration + 10.0)
        if res.result() is None or res.result().result.error_code != 0:
            self.get_logger().error(f'{name}: trajectory failed')
            return
        if not self.settle(name, target, SETTLE_TIMEOUT):
            self.get_logger().warn(f'{name}: finished but off target ({self.pos[name]:+.2f})')

    def run(self):
        self.wait_ready()
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.queue:
                name, target = self.queue.popleft()
                self.move_joint(name, target)
                if not self.queue:
                    self.get_logger().info('Idle')


def main():
    rclpy.init()
    node = ArmCommander()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        if node.grip_pub.get_subscription_count() > 0:
            node.grip_pub.publish(Float64MultiArray(data=[0.0]))
            rclpy.spin_once(node, timeout_sec=0.05)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()