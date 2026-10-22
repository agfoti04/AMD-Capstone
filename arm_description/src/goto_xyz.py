#!/usr/bin/env python3
import rclpy
from geometry_msgs.msg import Point, PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint
from moveit_msgs.srv import GetPositionIK
from rclpy.action import ActionClient
from rclpy.node import Node

GROUP, TIP, FRAME = 'arm', 'hand_link', 'base_link'
ARM_JOINTS = ('Joint_1', 'Joint_2', 'Joint_3', 'Joint_4', 'Joint_6')


class GotoXYZ(Node):
    def __init__(self):
        super().__init__('goto_xyz')
        self.request = None
        self.ik = self.create_client(GetPositionIK, '/compute_ik')
        self.mg = ActionClient(self, MoveGroup, '/move_action')
        self.create_subscription(Point, '/goto_xyz', self._on_req, 1)

    def _on_req(self, msg):
        self.request = msg

    def solve_ik(self, p):
        req = GetPositionIK.Request()
        r = req.ik_request
        r.group_name, r.ik_link_name, r.avoid_collisions = GROUP, TIP, True
        r.timeout.sec, r.timeout.nanosec = 1, 0
        r.pose_stamped = PoseStamped()
        r.pose_stamped.header.frame_id = FRAME
        r.pose_stamped.pose.position = p
        r.pose_stamped.pose.orientation.w = 1.0   # ignored by position_only_ik
        fut = self.ik.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        res = fut.result()
        if res is None or res.error_code.val != 1:
            code = None if res is None else res.error_code.val
            self.get_logger().warn(f'IK failed, code={code}')
            return None
        sol = dict(zip(res.solution.joint_state.name,
                       res.solution.joint_state.position))
        return {j: sol[j] for j in ARM_JOINTS}

    def move(self, joints):
        goal = MoveGroup.Goal()
        mpr = goal.request
        mpr.group_name = GROUP
        mpr.num_planning_attempts = 5
        mpr.allowed_planning_time = 5.0
        mpr.max_velocity_scaling_factor = 0.3
        mpr.max_acceleration_scaling_factor = 0.3
        c = Constraints()
        for name, pos in joints.items():
            c.joint_constraints.append(JointConstraint(
                joint_name=name, position=pos,
                tolerance_above=0.01, tolerance_below=0.01, weight=1.0))
        mpr.goal_constraints = [c]
        goal.planning_options.plan_only = False
        fut = self.mg.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
        handle = fut.result()
        if handle is None or not handle.accepted:
            self.get_logger().error('MoveGroup rejected the goal')
            return
        res = handle.get_result_async()
        rclpy.spin_until_future_complete(self, res)
        code = res.result().result.error_code.val
        self.get_logger().info(
            'Done' if code == 1 else f'Planning/execution failed (code {code})')

    def run(self):
        self.ik.wait_for_service()
        self.mg.wait_for_server()
        self.get_logger().info('Ready. Publish a geometry_msgs/Point to /goto_xyz')
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.request:
                p, self.request = self.request, None
                self.get_logger().info(
                    f'Target ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})')
                joints = self.solve_ik(p)
                if joints is None:
                    self.get_logger().warn('Out of reach: no IK solution')
                else:
                    self.move(joints)


def main():
    rclpy.init()
    node = GotoXYZ()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()