#!/usr/bin/env python3
"""Wheel command bridge (runs on the main computer / Strix Halo).

Subscribes to the diff drive controller's velocity command, reproduces the
controller's wheel math, and emits a WheelCommands struct (one value per
wheel) every time the command changes.

Proof of concept: the struct is printed to the terminal. Later, swap
PrintSink for a sink that serialises the struct and sends it over Ethernet
to the Raspberry Pi.

Run it directly with:

    ros2 run four_wheel_sim whl_to_term

It is also started by the package's Gazebo launch file.
"""
from dataclasses import dataclass
import math

from geometry_msgs.msg import Twist, TwistStamped
import rclpy
from rclpy.node import Node


# ----------------------------------------------------------------------
# The struct that will eventually go over Ethernet
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class WheelCommands:
    """Angular velocity command per wheel, rad/s.

    Positive = wheel rolls the robot forward. The sign is the direction;
    the Pi decides how to turn magnitude into PWM.
    """
    front_left: float = 0.0
    rear_left: float = 0.0
    front_right: float = 0.0
    rear_right: float = 0.0


# ----------------------------------------------------------------------
# Sinks: where a WheelCommands struct goes
# ----------------------------------------------------------------------
class PrintSink:
    def __init__(self, wheel_radius):
        self.r = wheel_radius
        self.count = 0

    def send(self, c: WheelCommands, reason: str):
        self.count += 1
        rows = [('front_left', c.front_left), ('rear_left', c.rear_left),
                ('front_right', c.front_right), ('rear_right', c.rear_right)]
        print(f'--- WheelCommands #{self.count} ({reason}) ---', flush=True)
        for name, w in rows:
            direction = 'FWD' if w > 0 else 'REV' if w < 0 else 'STOP'
            print(f'  {name:<12} {w:+8.3f} rad/s  {w * self.r:+7.3f} m/s  '
                  f'{direction}', flush=True)


# class UdpSink:   # future
#     def __init__(self, host, port): ...
#     def send(self, c: WheelCommands, reason: str): ...


# ----------------------------------------------------------------------
# Node
# ----------------------------------------------------------------------
def clamp(x, limit):
    return max(-limit, min(limit, x))


class WheelCommandBridge(Node):
    def __init__(self):
        super().__init__('wheel_command_bridge')

        # Defaults match controllers.yaml
        self.declare_parameter('cmd_topic', '/diff_drive_controller/cmd_vel')
        self.declare_parameter('stamped', True)
        self.declare_parameter('wheel_separation', 0.24)
        self.declare_parameter('wheel_separation_multiplier', 1.5)
        self.declare_parameter('wheel_radius', 0.05)
        self.declare_parameter('max_linear_velocity', 0.6)
        self.declare_parameter('max_angular_velocity', 1.5)
        self.declare_parameter('cmd_vel_timeout', 0.5)

        g = lambda n: self.get_parameter(n).value
        self.sep = g('wheel_separation') * g('wheel_separation_multiplier')
        self.radius = g('wheel_radius')
        self.max_v = g('max_linear_velocity')
        self.max_w = g('max_angular_velocity')
        self.timeout = g('cmd_vel_timeout')

        self.sink = PrintSink(self.radius)
        self.last_sent = None
        self.last_cmd_time = None

        topic = g('cmd_topic')
        if g('stamped'):
            self.create_subscription(
                TwistStamped, topic, lambda m: self.cmd_cb(m.twist), 10)
        else:
            self.create_subscription(Twist, topic, self.cmd_cb, 10)

        self.create_timer(0.05, self.watchdog)
        self.get_logger().info(
            f'Listening on {topic} '
            f'({"TwistStamped" if g("stamped") else "Twist"}). '
            f'Printing a struct on each change.')

    def cmd_cb(self, twist):
        self.last_cmd_time = self.get_clock().now()

        # Same limiting and kinematics as ros2_control's diff_drive_controller
        v = clamp(twist.linear.x, self.max_v)
        w = clamp(twist.angular.z, self.max_w)
        left = (v - w * self.sep / 2.0) / self.radius
        right = (v + w * self.sep / 2.0) / self.radius

        # Rounding keeps float noise from counting as an "update"
        left, right = round(left, 3), round(right, 3)
        self.emit(WheelCommands(left, left, right, right), 'new command')

    def watchdog(self):
        # Mirrors cmd_vel_timeout: no command -> wheels go to zero
        if self.last_cmd_time is None:
            return
        age = (self.get_clock().now() - self.last_cmd_time).nanoseconds / 1e9
        if age > self.timeout:
            self.emit(WheelCommands(), 'timeout')

    def emit(self, cmds: WheelCommands, reason: str):
        if cmds == self.last_sent:
            return
        self.last_sent = cmds
        self.sink.send(cmds, reason)


def main():
    rclpy.init()
    node = WheelCommandBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()