#!/usr/bin/env python3
"""
arm_bridge: tap the ROS 2 arm_controller and mirror its joint positions into a struct.

For now the struct is just printed to the terminal. Later, `on_update()` is the one
place to hook in the Ethernet / Raspberry Pi / servo-driver translation.

What it listens to
------------------
The JointTrajectoryController publishes its controller state every update cycle.
That state holds the *reference* (the position the controller is commanding right
now, i.e. what a position servo should be told to go to) and the *feedback*
(where the simulated joint actually is). This works no matter who sent the
trajectory (MoveIt action, topic publisher, arm_teleop, ...).

  Jazzy / Iron / newer : /arm_controller/controller_state  (reference, feedback)
  Humble               : /arm_controller/state             (desired, actual)

The node subscribes to both topic names, so it works on any distro.

Run (in its own terminal, with the sim already running):
  source /opt/ros/<distro>/setup.bash
  python3 arm_bridge.py --ros-args -p use_sim_time:=true
"""

import sys
from dataclasses import dataclass, field

import rclpy
from control_msgs.msg import JointTrajectoryControllerState
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

# Order matters: this is the order of the servos / the struct fields.
ARM_JOINTS = ('Joint_1', 'Joint_2', 'Joint_3', 'Joint_4', 'Joint_6')

STATE_TOPICS = (
    '/arm_controller/controller_state',  # Iron and newer
    '/arm_controller/state',             # Humble
)
STALE_AFTER_SECONDS = 2.0


@dataclass
class ArmState:
    """Latest joint positions, one entry per joint in ARM_JOINTS order (radians)."""
    joint_names: tuple = ARM_JOINTS
    commanded: list = field(default_factory=lambda: [0.0] * len(ARM_JOINTS))  # setpoint -> servos
    actual: list = field(default_factory=lambda: [0.0] * len(ARM_JOINTS))     # sim feedback
    stamp: float = 0.0   # node-clock seconds of the last update
    valid: bool = False  # becomes True after the first message


class ArmBridge(Node):
    def __init__(self):
        super().__init__('arm_bridge')
        self.state = ArmState()

        # Best-effort so we match both reliable and best-effort publishers.
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._subs = [
            self.create_subscription(
                JointTrajectoryControllerState, topic, self._on_controller_state, qos
            )
            for topic in STATE_TOPICS
        ]

    # ------------------------------------------------------------------ input
    def _on_controller_state(self, msg):
        # Field names differ between ROS distros.
        ref = msg.reference if hasattr(msg, 'reference') else msg.desired
        fb = msg.feedback if hasattr(msg, 'feedback') else msg.actual

        # Map by joint name so the struct order never depends on message order.
        changed = not self.state.valid
        names = list(msg.joint_names)
        for i, joint in enumerate(ARM_JOINTS):
            if joint not in names:
                continue
            idx = names.index(joint)
            if idx < len(ref.positions):
                changed |= self.state.commanded[i] != ref.positions[idx]
                self.state.commanded[i] = ref.positions[idx]
            if idx < len(fb.positions):
                changed |= self.state.actual[i] != fb.positions[idx]
                self.state.actual[i] = fb.positions[idx]

        self.state.stamp = self.get_clock().now().nanoseconds * 1e-9
        self.state.valid = True
        self.on_update(self.state)
        if changed:
            self._print_state()

    def on_update(self, state: ArmState):
        """Called on every controller update. Future: send `state.commanded` over
        Ethernet to the Pi / convert to servo driver commands here."""
        pass

    # ----------------------------------------------------------------- output
    def _print_state(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        s = self.state

        lines = ['\033[H\033[J' + 'arm_bridge  (Ctrl+C to quit)', '']
        if not s.valid:
            lines.append('Waiting for arm_controller state on:')
            lines += [f'  {t}' for t in STATE_TOPICS]
            lines.append('Is the sim running and arm_controller active?')
        else:
            age = now - s.stamp
            lines.append(f'{"joint":<10}{"cmd [rad]":>12}{"cmd [deg]":>12}')
            lines.append('-' * 34)
            import math
            for name, cmd in zip(s.joint_names, s.commanded):
                lines.append(
                    f'{name:<10}{cmd:>+12.4f}{math.degrees(cmd):>+12.2f}'
                )
            lines.append('')
            if age > STALE_AFTER_SECONDS:
                lines.append(f'WARNING: no update for {age:.1f} s')
            else:
                lines.append(f'last update {age * 1000:.0f} ms ago')
        sys.stdout.write('\n'.join(lines) + '\n')
        sys.stdout.flush()


def main():
    rclpy.init()
    node = ArmBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()