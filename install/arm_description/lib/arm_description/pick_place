#!/usr/bin/env python3
import math
import time

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from moveit_msgs.srv import GetPositionIK
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

ARM_JOINTS = ('Joint_1', 'Joint_2', 'Joint_3', 'Joint_4', 'Joint_6')
GRIPPER = 'Joint_Gripper'
GROUP, TIP, FRAME = 'arm', 'EE_Gripper_Link', 'base_link'

GRIPPER_EFFORT_LIMIT = 1.0
GRIP_OPEN_EFFORT = -GRIPPER_EFFORT_LIMIT
GRIP_CLOSE_EFFORT = GRIPPER_EFFORT_LIMIT
GRIPPER_SETTLE_SECONDS = 0.5

# ---- Task tuning ----
PRE_HEIGHT = 0.06       # m above the grasp point for approach and lift
SPEED = 0.4             # rad/s used to time moves
HOME = [0.0, 0.0, 0.0, 0.0, 0.0]
ROD_PICK = (0.25, 0.0, 0.06)
ROD_PLACE = (0.25, 0.15, 0.06)
ROD_PREGRASP_DISTANCE = 0.06
ROD_LIFT_HEIGHT = 0.06


class PickPlace(Node):
    def __init__(self):
        super().__init__('pick_place')
        self.pos = {}
        self.request = None
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.create_subscription(JointState, '/joint_states', self._on_state, 10)
        self.create_subscription(
            Float64MultiArray, '/pick_place', self._on_request, 10)
        self.client = ActionClient(
            self, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')
        self.grip_pub = self.create_publisher(
            Float64MultiArray, '/gripper_controller/commands', 10)
        self.grip_cmd = 0.0
        self.create_timer(0.05, self._grip_tick)


    def _grip_tick(self):
        self.grip_pub.publish(Float64MultiArray(data=[self.grip_cmd]))

    def _on_state(self, msg):
        for n, p in zip(msg.name, msg.position):
            self.pos[n] = p

    def _on_request(self, msg):
        if len(msg.data) not in (0, 6):
            self.get_logger().error(
                'Send an empty request for the default rod transfer, or '
                '[pick_x, pick_y, pick_z, place_x, place_y, place_z] for a custom transfer')
            return
        self.request = list(msg.data)

    # ---------- kinematics ----------
    def ik(self, x, y, z, previous):
        """Solve for the current URDF using the configured position-only IK."""
        request = GetPositionIK.Request()
        ik_request = request.ik_request
        ik_request.group_name = GROUP
        ik_request.ik_link_name = TIP
        ik_request.avoid_collisions = True
        ik_request.timeout.sec = 0
        ik_request.timeout.nanosec = 200_000_000
        ik_request.pose_stamped.header.frame_id = FRAME
        ik_request.pose_stamped.pose.position.x = x
        ik_request.pose_stamped.pose.position.y = y
        ik_request.pose_stamped.pose.position.z = z
        ik_request.pose_stamped.pose.orientation.w = 1.0
        ik_request.robot_state.is_diff = True
        ik_request.robot_state.joint_state.name = list(ARM_JOINTS)
        ik_request.robot_state.joint_state.position = previous

        future = self.ik_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        response = future.result()
        if response is None or response.error_code.val != 1:
            return None
        solution = dict(zip(response.solution.joint_state.name,
                            response.solution.joint_state.position))
        if any(joint not in solution for joint in ARM_JOINTS):
            self.get_logger().error('IK solution omitted one or more arm joints')
            return None
        return [float(solution[joint]) for joint in ARM_JOINTS]

    def plan_vertical(self, x, y, z_low, z_high, prev):
        """Plan position waypoints, seeding each IK solve from the previous one."""
        waypoints, previous = [], prev
        for z in np.linspace(z_high, z_low, 4):
            solution = self.ik(x, y, float(z), previous)
            if solution is None:
                return None
            waypoints.append(solution)
            previous = solution
        return waypoints

    def plan_line(self, start, end, prev):
        """Plan Cartesian position waypoints with seeded IK."""
        waypoints, previous = [], prev
        for fraction in np.linspace(0.0, 1.0, 4):
            point = [
                start[i] + float(fraction) * (end[i] - start[i])
                for i in range(3)
            ]
            solution = self.ik(point[0], point[1], point[2], previous)
            if solution is None:
                return None
            waypoints.append(solution)
            previous = solution
        return waypoints

    # ---------- motion ----------
    def arm_now(self):
        return [self.pos[j] for j in ARM_JOINTS]

    def send(self, waypoints):
        pts, t, prev = [], 0.0, self.arm_now()
        for wp in waypoints:
            t += max(0.5, max(abs(a - b) for a, b in zip(wp, prev)) / SPEED)
            pt = JointTrajectoryPoint(positions=[float(v) for v in wp])
            pt.time_from_start.sec = int(t)
            pt.time_from_start.nanosec = int((t % 1) * 1e9)
            pts.append(pt)
            prev = wp
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory(joint_names=list(ARM_JOINTS), points=pts)
        fut = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        handle = fut.result()
        if handle is None or not handle.accepted:
            self.get_logger().error('Trajectory rejected')
            return False
        res = handle.get_result_async()
        rclpy.spin_until_future_complete(self, res, timeout_sec=t + 10.0)
        if res.result() is None or res.result().result.error_code != 0:
            self.get_logger().error('Trajectory failed')
            return False
        self.spin_for(0.3)
        return True

    def spin_for(self, sec):
        end = time.monotonic() + sec
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def grip(self, effort):
        self.grip_cmd = effort
        self.grip_pub.publish(Float64MultiArray(data=[effort]))
        self.spin_for(GRIPPER_SETTLE_SECONDS)
        if effort == GRIP_OPEN_EFFORT:
            self.grip_cmd = 0.0
            self.grip_pub.publish(Float64MultiArray(data=[0.0]))

    # ---------- task ----------
    def execute(self, d):
        if not d:
            self.execute_rod_transfer()
            return

        px, py, pz, qx, qy, qz = d
        cur = self.arm_now()
        pick = self.plan_vertical(px, py, pz, pz + PRE_HEIGHT, cur)
        if pick is None:
            self.get_logger().error('Pick point unreachable. Move it closer to the base.')
            return
        place = self.plan_vertical(qx, qy, qz, qz + PRE_HEIGHT, pick[0])
        if place is None:
            self.get_logger().error('Place point unreachable. Move it closer to the base.')
            return
        pick_down = pick                  # pre-grasp -> grasp
        pick_up = pick[::-1]
        place_down = place
        place_up = place[::-1]

        self.get_logger().info('Opening gripper')
        self.grip(GRIP_OPEN_EFFORT)
        self.get_logger().info('Approach and descend')
        if not self.send(pick_down):
            return
        self.get_logger().info('Closing gripper')
        self.grip(GRIP_CLOSE_EFFORT)
        self.get_logger().info('Lift')
        if not self.send(pick_up):
            return
        self.get_logger().info('Carry to place point and lower')
        if not self.send(place_down):
            return
        self.get_logger().info('Releasing')
        self.grip(GRIP_OPEN_EFFORT)
        self.get_logger().info('Retreat and home')
        if not self.send(place_up):
            return
        self.send([HOME])
        self.get_logger().info('Done')

    def execute_rod_transfer(self):
        px, py, pz = ROD_PICK
        qx, qy, qz = ROD_PLACE
        radius = math.hypot(px, py)
        if radius <= ROD_PREGRASP_DISTANCE:
            self.get_logger().error('Rod pre-grasp distance reaches behind the robot base')
            return

        pregrasp = (
            px * (radius - ROD_PREGRASP_DISTANCE) / radius,
            py * (radius - ROD_PREGRASP_DISTANCE) / radius,
            pz,
        )
        pick = (px, py, pz)
        lift = (px, py, pz + ROD_LIFT_HEIGHT)
        place = (qx, qy, qz)
        place_lift = (qx, qy, qz + ROD_LIFT_HEIGHT)

        approach_path = self.plan_line(pregrasp, pick, self.arm_now())
        if approach_path is None:
            self.get_logger().error('Rod midpoint approach is unreachable')
            return
        lift_path = self.plan_line(pick, lift, approach_path[-1])
        if lift_path is None:
            self.get_logger().error('Rod lift path is unreachable')
            return
        carry_path = self.plan_line(lift, place_lift, lift_path[-1])
        if carry_path is None:
            self.get_logger().error('Rod carry path is unreachable')
            return
        lower_path = self.plan_line(place_lift, place, carry_path[-1])
        if lower_path is None:
            self.get_logger().error('Rod set-down path is unreachable')
            return
        retreat_path = self.plan_line(place, place_lift, lower_path[-1])
        if retreat_path is None:
            self.get_logger().error('Rod retreat path is unreachable')
            return

        self.get_logger().info(
            f'Grasping rod midpoint at ({px:.2f}, {py:.2f}, {pz:.2f})')
        self.grip(GRIP_OPEN_EFFORT)
        if not self.send(approach_path):
            return
        self.grip(GRIP_CLOSE_EFFORT)
        if not self.send(lift_path):
            return
        if not self.send(carry_path):
            return
        if not self.send(lower_path):
            return
        self.grip(GRIP_OPEN_EFFORT)
        if not self.send(retreat_path):
            return
        if self.send([HOME]):
            self.get_logger().info('Rod transfer complete')

    def run(self):
        self.get_logger().info('Waiting for joint states and controller...')
        while rclpy.ok() and not set(ARM_JOINTS + (GRIPPER,)).issubset(self.pos):
            rclpy.spin_once(self, timeout_sec=0.1)
        if not self.ik_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error('MoveIt /compute_ik service is not available')
            return
        self.client.wait_for_server()
        while rclpy.ok() and self.grip_pub.get_subscription_count() == 0:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info(
            'Ready. Publish an empty Float64MultiArray to /pick_place for the '
            'rod transfer, or six coordinates for a custom transfer')
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.request is not None:
                req, self.request = self.request, None
                self.execute(req)


def main():
    rclpy.init()
    node = PickPlace()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.grip_cmd = 0.0
        if node.grip_pub.get_subscription_count() > 0:
            node.grip_pub.publish(Float64MultiArray(data=[0.0]))
            rclpy.spin_once(node, timeout_sec=0.05)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()