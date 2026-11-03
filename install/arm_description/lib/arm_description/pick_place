#!/usr/bin/env python3
import math
import time

import numpy as np
import rclpy
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint
from rclpy.action import ActionClient
from rclpy.node import Node
from moveit_msgs.srv import GetPositionFK, GetPositionIK
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray

ARM_JOINTS = ('Joint_1', 'Joint_2', 'Joint_3', 'Joint_4', 'Joint_6')
ARM_STATE_TOLERANCE = 0.03
ARM_STATE_TIMEOUT_SECONDS = 5.0
GRIPPER = 'Joint_Gripper'
GROUP, TIP, FRAME = 'arm', 'grasp_point', 'base_link'

GRIP_OPEN_POSITION = 0.0
GRIP_CLOSE_POSITION = -1.57
GRIPPER_CLOSE_TIMEOUT_SECONDS = 4.0
GRIPPER_CONTACT_SETTLE_SECONDS = 0.4
GRIPPER_OPEN_TIMEOUT_SECONDS = 3.0
GRIPPER_POSITION_TOLERANCE = 0.04

# ---- Task tuning ----
PRE_HEIGHT = 0.06       # m above the grasp point for approach and lift
HOME = [0.0, 0.0, 0.0, 0.0, 0.0]
# grasp_point is 2 cm below the top of the 16 cm rod.
ROD_PICK = (0.12, 0.0, 0.14)
ROD_PLACE = (0.12, 0.05, 0.14)
ROD_APPROACH_HEIGHT = 0.08
ROD_LIFT_HEIGHT = 0.01
TOOL_AXIS_TOLERANCE = math.radians(50.0)


class PickPlace(Node):
    def __init__(self):
        super().__init__('pick_place')
        self.pos = {}
        self.request = None
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.fk_client = self.create_client(GetPositionFK, '/compute_fk')
        self.create_subscription(JointState, '/joint_states', self._on_state, 10)
        self.create_subscription(
            Float64MultiArray, '/pick_place', self._on_request, 10)
        self.move_group = ActionClient(self, MoveGroup, '/move_action')
        self.grip_pub = self.create_publisher(
            Float64, '/gripper_cmd', 10)

    def _on_state(self, msg):
        for n, p in zip(msg.name, msg.position):
            self.pos[n] = p

    def _on_request(self, msg):
        if len(msg.data) not in (0, 6) or any(
                not math.isfinite(value) for value in msg.data):
            self.get_logger().error(
                'Send an empty request for the default rod transfer, or six finite '
                '[pick_x, pick_y, pick_z, place_x, place_y, place_z] for a custom transfer')
            return
        self.request = list(msg.data)
        request_type = 'default rod transfer' if not msg.data else 'custom transfer'
        self.get_logger().info(f'Received {request_type} request')

    # ---------- kinematics ----------
    def tool_axis_tilt(self, joints):
        request = GetPositionFK.Request()
        request.header.frame_id = FRAME
        request.fk_link_names = [TIP]
        request.robot_state.is_diff = True
        request.robot_state.joint_state.name = list(ARM_JOINTS)
        request.robot_state.joint_state.position = joints
        future = self.fk_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if not future.done():
            self.get_logger().error('Timed out checking the gripper FK orientation')
            return None
        response = future.result()
        if (response is None or response.error_code.val != 1
                or not response.pose_stamped):
            code = None if response is None else response.error_code.val
            self.get_logger().error(
                f'Could not compute gripper FK orientation (MoveIt code {code})')
            return None
        q = response.pose_stamped[0].pose.orientation
        axis_z = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        axis_x = 2.0 * (q.x * q.z + q.w * q.y)
        axis_y = 2.0 * (q.y * q.z - q.w * q.x)
        return math.atan2(math.hypot(axis_x, axis_y), -axis_z)

    def ik(self, x, y, z, previous, require_grasp_orientation=False):
        """Solve the target position and optionally limit tool-axis tilt."""
        request = GetPositionIK.Request()
        ik_request = request.ik_request
        ik_request.group_name = GROUP
        ik_request.ik_link_name = TIP
        ik_request.timeout.sec = 1
        ik_request.timeout.nanosec = 0
        ik_request.pose_stamped.header.frame_id = FRAME
        ik_request.pose_stamped.pose.position.x = x
        ik_request.pose_stamped.pose.position.y = y
        ik_request.pose_stamped.pose.position.z = z
        ik_request.pose_stamped.pose.orientation.w = 1.0
        ik_request.robot_state.is_diff = True
        ik_request.robot_state.joint_state.name = list(ARM_JOINTS)

        seeds = []
        for candidate in (
                previous,
                [0.0, 0.0, -1.4, -1.5, 0.0],
                self.arm_now(),
                HOME):
            if not any(np.allclose(candidate, seed) for seed in seeds):
                seeds.append(candidate)

        error_codes = {True: None, False: None}
        best_tilt = None
        for avoid_collisions in (True, False):
            ik_request.avoid_collisions = avoid_collisions
            for seed in seeds:
                ik_request.robot_state.joint_state.position = seed
                future = self.ik_client.call_async(request)
                rclpy.spin_until_future_complete(
                    self, future, timeout_sec=5.0)
                if not future.done():
                    error_codes[avoid_collisions] = 'timeout'
                    continue
                response = future.result()
                code = None if response is None else response.error_code.val
                error_codes[avoid_collisions] = code
                if response is None or code != 1:
                    continue

                solution = dict(zip(response.solution.joint_state.name,
                                    response.solution.joint_state.position))
                if any(joint not in solution for joint in ARM_JOINTS):
                    self.get_logger().error(
                        'IK solution omitted one or more arm joints')
                    return None
                joints = [float(solution[joint]) for joint in ARM_JOINTS]
                if require_grasp_orientation:
                    tilt = self.tool_axis_tilt(joints)
                    if tilt is None:
                        continue
                    best_tilt = tilt if best_tilt is None else min(best_tilt, tilt)
                    if tilt > TOOL_AXIS_TOLERANCE:
                        continue
                if not avoid_collisions:
                    self.get_logger().warn(
                        f'IK found a candidate at ({x:.3f}, {y:.3f}, {z:.3f}) '
                        'only without collision filtering; MoveIt will '
                        'collision-check the planned arm motion')
                return joints

        self.get_logger().error(
            f'IK failed at ({x:.3f}, {y:.3f}, {z:.3f}) in {FRAME}; '
            f'collision-checked code {error_codes[True]}, '
            f'collision-unchecked code {error_codes[False]} after {len(seeds)} '
            f'joint seeds; best tool-axis tilt from down was '
            f'{None if best_tilt is None else f"{math.degrees(best_tilt):.1f} deg"}')
        return None

    def plan_vertical(
            self, x, y, z_low, z_high, prev,
            require_grasp_orientation=False):
        """Plan position waypoints, seeding each IK solve from the previous one."""
        waypoints, previous = [], prev
        for z in np.linspace(z_high, z_low, 4):
            solution = self.ik(
                x, y, float(z), previous,
                require_grasp_orientation=require_grasp_orientation)
            if solution is None:
                return None
            waypoints.append(solution)
            previous = solution
        return waypoints

    def plan_line(
            self, start, end, prev, require_grasp_orientation=False):
        """Plan Cartesian position waypoints with seeded IK."""
        waypoints, previous = [], prev
        for fraction in np.linspace(0.0, 1.0, 4):
            point = [
                start[i] + float(fraction) * (end[i] - start[i])
                for i in range(3)
            ]
            solution = self.ik(
                point[0], point[1], point[2], previous,
                require_grasp_orientation=require_grasp_orientation)
            if solution is None:
                return None
            waypoints.append(solution)
            previous = solution
        return waypoints

    # ---------- motion ----------
    def arm_now(self):
        return [self.pos[j] for j in ARM_JOINTS]

    def send_goal_constraints(self, constraints):
        goal = MoveGroup.Goal()
        request = goal.request
        request.group_name = GROUP
        request.num_planning_attempts = 5
        request.allowed_planning_time = 5.0
        request.max_velocity_scaling_factor = 0.3
        request.max_acceleration_scaling_factor = 0.3
        request.goal_constraints = [constraints]
        goal.planning_options.plan_only = False

        send_future = self.move_group.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=10.0)
        if not send_future.done():
            self.get_logger().error('Timed out sending an arm waypoint to MoveIt')
            return False
        handle = send_future.result()
        if handle is None or not handle.accepted:
            self.get_logger().error('MoveIt rejected an arm waypoint')
            return False

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=30.0)
        if not result_future.done():
            self.get_logger().error('MoveIt timed out planning or executing a waypoint')
            cancel_future = handle.cancel_goal_async()
            rclpy.spin_until_future_complete(
                self, cancel_future, timeout_sec=2.0)
            return False
        wrapped_result = result_future.result()
        if wrapped_result is None or wrapped_result.result.error_code.val != 1:
            code = (
                None if wrapped_result is None
                else wrapped_result.result.error_code.val
            )
            self.get_logger().error(
                f'MoveIt failed to plan or execute a waypoint (code {code})')
            return False
        self.spin_for(0.2)
        return True

    def send(self, waypoints, verify_state=True):
        for waypoint in waypoints:
            constraints = Constraints()
            for name, position in zip(ARM_JOINTS, waypoint):
                constraints.joint_constraints.append(JointConstraint(
                    joint_name=name,
                    position=float(position),
                    tolerance_above=0.01,
                    tolerance_below=0.01,
                    weight=1.0,
                ))
            if not self.send_goal_constraints(constraints):
                return False
            if not verify_state:
                errors = {
                    name: (
                        None if name not in self.pos
                        else round(self.pos[name] - position, 4)
                    )
                    for name, position in zip(ARM_JOINTS, waypoint)
                }
                self.get_logger().warn(
                    f'MoveIt reported success; measured waypoint errors '
                    f'(rad): {errors}')
            if verify_state and not self.wait_for_arm_state(waypoint):
                return False
        return True

    def wait_for_arm_state(self, target):
        deadline = time.monotonic() + ARM_STATE_TIMEOUT_SECONDS
        while rclpy.ok() and time.monotonic() < deadline:
            if all(
                    name in self.pos
                    and abs(self.pos[name] - value) <= ARM_STATE_TOLERANCE
                    for name, value in zip(ARM_JOINTS, target)):
                return True
            rclpy.spin_once(self, timeout_sec=0.05)

        errors = {
            name: (
                None if name not in self.pos
                else round(self.pos[name] - value, 4)
            )
            for name, value in zip(ARM_JOINTS, target)
        }
        self.get_logger().error(
            f'Arm did not settle at waypoint within '
            f'{ARM_STATE_TIMEOUT_SECONDS:.1f} s; joint errors (rad): {errors}')
        return False

    def spin_for(self, sec):
        end = time.monotonic() + sec
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def grip(self, position, wait_until_open=False):
        start_position = self.pos.get(GRIPPER)
        if start_position is None:
            self.get_logger().error(
                f'Cannot command gripper: no joint state for {GRIPPER}')
            return False

        self.grip_pub.publish(Float64(data=position))
        self.get_logger().info(
            f'Published gripper command {position:.2f} rad '
            f'(current {start_position:.2f} rad)')

        timeout = (
            GRIPPER_OPEN_TIMEOUT_SECONDS
            if wait_until_open else GRIPPER_CLOSE_TIMEOUT_SECONDS
        )
        deadline = time.monotonic() + timeout
        last_position = start_position
        last_motion_time = time.monotonic()
        observed_motion = False
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            current = self.pos.get(GRIPPER)
            if current is None:
                continue
            if abs(current - position) <= GRIPPER_POSITION_TOLERANCE:
                self.get_logger().info(
                    f'Gripper reached {current:.2f} rad')
                return True

            if wait_until_open:
                continue

            if abs(current - last_position) > 0.01:
                observed_motion = True
                last_motion_time = time.monotonic()
            elif (observed_motion
                  and time.monotonic() - last_motion_time
                  >= GRIPPER_CONTACT_SETTLE_SECONDS):
                self.get_logger().info(
                    f'Gripper stopped at {current:.2f} rad after closing '
                    'motion (likely contact with the rod)')
                return True
            last_position = current

        if wait_until_open:
            self.get_logger().error('Gripper did not reach the open position')
        elif observed_motion:
            self.get_logger().error(
                f'Gripper moved while closing but did not settle; '
                f'Joint_Gripper={self.pos.get(GRIPPER):.2f} rad')
        else:
            self.get_logger().error(
                'Gripper close command produced no measured joint motion')
        return False

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
        if not self.grip(GRIP_OPEN_POSITION, wait_until_open=True):
            return
        self.get_logger().info('Approach and descend')
        if not self.send(pick_down):
            return
        self.get_logger().info('Closing gripper')
        self.grip(GRIP_CLOSE_POSITION)
        self.get_logger().info('Lift')
        if not self.send(pick_up):
            return
        self.get_logger().info('Carry to place point and lower')
        if not self.send(place_down):
            return
        self.get_logger().info('Releasing')
        if not self.grip(GRIP_OPEN_POSITION, wait_until_open=True):
            return
        self.get_logger().info('Retreat and home')
        if not self.send(place_up):
            return
        self.send([HOME])
        self.get_logger().info('Done')

    def execute_rod_transfer(self):
        self.get_logger().info('Planning rod transfer waypoints')
        px, py, pz = ROD_PICK
        qx, qy, qz = ROD_PLACE

        pregrasp = (px, py, pz + ROD_APPROACH_HEIGHT)
        pick = (px, py, pz)
        lift = (px, py, pz + ROD_LIFT_HEIGHT)
        place = (qx, qy, qz)
        place_lift = (qx, qy, qz + ROD_LIFT_HEIGHT)

        self.get_logger().info(
            f'Moving gripper grasp point near rod top '
            f'({px:.2f}, {py:.2f}, {pz:.2f})')
        pregrasp_joints = self.ik(
            *pregrasp, self.arm_now(), require_grasp_orientation=False)
        if pregrasp_joints is None:
            self.get_logger().error('Rod pre-grasp position is unreachable')
            return
        if not self.grip(GRIP_OPEN_POSITION, wait_until_open=True):
            return
        if not self.send([pregrasp_joints], verify_state=False):
            self.get_logger().error('MoveIt could not reach the rod top grasp point')
            return

        measured_pregrasp = self.arm_now()
        self.get_logger().info(
            f'Resolving pick from measured arm state; Joint_3='
            f'{measured_pregrasp[ARM_JOINTS.index("Joint_3")]:.3f} rad')
        pick_joints = self.ik(
            *pick, measured_pregrasp, require_grasp_orientation=True)
        if pick_joints is None:
            self.get_logger().error(
                'Could not solve the pick from the measured pre-grasp state')
            return
        if not self.send([pick_joints], verify_state=False):
            self.get_logger().error('MoveIt could not execute the rod grasp approach')
            return
        if not self.grip(GRIP_CLOSE_POSITION):
            return
        if not self.wait_for_arm_state(pick_joints):
            self.get_logger().error(
                'Gripper was commanded closed, but the arm missed the grasp '
                'waypoint; stopping before lift')
            return

        lift_path = self.plan_line(
            pick, lift, pick_joints, require_grasp_orientation=True)
        if lift_path is None:
            self.get_logger().error('Rod lift path is unreachable from measured grasp')
            return
        carry_path = self.plan_line(
            lift, place_lift, lift_path[-1],
            require_grasp_orientation=True)
        if carry_path is None:
            self.get_logger().error('Rod carry path is unreachable from measured grasp')
            return
        lower_path = self.plan_line(
            place_lift, place, carry_path[-1],
            require_grasp_orientation=True)
        if lower_path is None:
            self.get_logger().error('Rod set-down path is unreachable from measured grasp')
            return
        retreat_joints = self.ik(
            qx, qy, qz + ROD_APPROACH_HEIGHT,
            lower_path[-1], require_grasp_orientation=False)
        if retreat_joints is None:
            self.get_logger().error('Rod retreat position is unreachable')
            return
        if not self.send(lift_path[1:]):
            self.get_logger().error('MoveIt could not lift the grasped rod')
            return
        if not self.send(carry_path[1:]):
            self.get_logger().error('MoveIt could not carry the rod to the drop point')
            return
        if not self.send(lower_path[1:]):
            self.get_logger().error('MoveIt could not lower the rod to the drop point')
            return
        if not self.grip(GRIP_OPEN_POSITION, wait_until_open=True):
            return
        if not self.send([retreat_joints]):
            self.get_logger().error('MoveIt could not retreat from the drop point')
            return
        if self.send([HOME]):
            self.get_logger().info('Rod transfer complete')

    def run(self):
        self.get_logger().info('Waiting for joint states and controller...')
        state_deadline = time.monotonic() + 20.0
        required_joints = set(ARM_JOINTS + (GRIPPER,))
        while rclpy.ok() and not required_joints.issubset(self.pos):
            if time.monotonic() >= state_deadline:
                missing = sorted(required_joints - self.pos.keys())
                self.get_logger().error(
                    'Timed out waiting for joint states: ' + ', '.join(missing))
                return
            rclpy.spin_once(self, timeout_sec=0.1)
        if not self.ik_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error('MoveIt /compute_ik service is not available')
            return
        if not self.fk_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error('MoveIt /compute_fk service is not available')
            return
        if not self.move_group.wait_for_server(timeout_sec=15.0):
            self.get_logger().error('MoveIt /move_action is not available')
            return
        fan_deadline = time.monotonic() + 10.0
        while rclpy.ok() and self.grip_pub.get_subscription_count() == 0:
            if time.monotonic() >= fan_deadline:
                self.get_logger().error(
                    'No gripper_fan subscriber on /gripper_cmd')
                return
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info(
            'Ready. Publish an empty Float64MultiArray to /pick_place for the '
            'rod transfer, or six coordinates for a custom transfer')
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.request is not None:
                req, self.request = self.request, None
                self.get_logger().info('Starting pick-and-place request')
                self.execute(req)


def main():
    rclpy.init()
    node = PickPlace()
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