import select
import sys
import termios
import time
import tty

from geometry_msgs.msg import TwistStamped
import rclpy
from rclpy.node import Node

PUBLISH_PERIOD = 0.05
INPUT_TIMEOUT = 0.75
KEY_BINDINGS = {
    'i': (0.3, 0.0),
    ',': (-0.3, 0.0),
    'j': (0.0, 0.7),
    'l': (0.0, -0.7),
}


class Teleop(Node):
    def __init__(self):
        super().__init__(
            'my_teleop',
            parameter_overrides=[
                rclpy.parameter.Parameter('use_sim_time', value=True)])
        self.publisher = self.create_publisher(
            TwistStamped, '/diff_drive_controller/cmd_vel', 10)


def main():
    if not sys.stdin.isatty():
        raise RuntimeError('Keyboard teleop requires an interactive terminal.')

    terminal_settings = termios.tcgetattr(sys.stdin)
    rclpy.init()
    node = Teleop()
    linear_x = 0.0
    angular_z = 0.0
    last_key_time = 0.0

    try:
        tty.setcbreak(sys.stdin.fileno())
        node.get_logger().info(
            'Teleop ready: i forward, , reverse, j/l turn, k stop, q quit.')
        while rclpy.ok():
            readable, _, _ = select.select(
                [sys.stdin], [], [], PUBLISH_PERIOD)
            if readable:
                key = sys.stdin.read(1)
                if key == 'q':
                    break
                if key == 'k':
                    linear_x = angular_z = 0.0
                    last_key_time = 0.0
                elif key in KEY_BINDINGS:
                    linear_x, angular_z = KEY_BINDINGS[key]
                    last_key_time = time.monotonic()
                else:
                    linear_x = angular_z = 0.0
                    last_key_time = 0.0

            if (last_key_time and
                    time.monotonic() - last_key_time > INPUT_TIMEOUT):
                linear_x = angular_z = 0.0
                last_key_time = 0.0

            command = TwistStamped()
            command.header.stamp = node.get_clock().now().to_msg()
            command.header.frame_id = 'base_link'
            command.twist.linear.x = linear_x
            command.twist.angular.z = angular_z
            node.publisher.publish(command)
            rclpy.spin_once(node, timeout_sec=0.0)
    except KeyboardInterrupt:
        pass
    finally:
        stop = TwistStamped()
        stop.header.stamp = node.get_clock().now().to_msg()
        stop.header.frame_id = 'base_link'
        node.publisher.publish(stop)
        node.destroy_node()
        rclpy.shutdown()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, terminal_settings)


if __name__ == '__main__':
    main()
