from geometry_msgs.msg import Twist
from rclpy.node import Node
import rclpy
from sensor_msgs.msg import JointState


WHEEL_NAMES = (
    'front_left_wheel_joint',
    'rear_left_wheel_joint',
    'front_right_wheel_joint',
    'rear_right_wheel_joint',
)


class BaseController(Node):
    def __init__(self):
        super().__init__('base_controller')

        self.declare_parameter('wheel_radius', 0.05)
        self.declare_parameter('wheel_separation', 0.24)
        self.declare_parameter('command_timeout', 0.5)
        self.wheel_radius = self.get_parameter('wheel_radius').value
        self.wheel_separation = self.get_parameter('wheel_separation').value
        self.command_timeout = self.get_parameter('command_timeout').value

        if (self.wheel_radius <= 0.0 or self.wheel_separation <= 0.0
                or self.command_timeout <= 0.0):
            raise ValueError('wheel dimensions and command_timeout must be positive')

        self.wheel_speeds = [0.0, 0.0, 0.0, 0.0]
        self.last_command_time = self.get_clock().now()

        self.create_subscription(Twist, 'cmd_vel', self.cmd_cb, 10)
        self.wheel_pub = self.create_publisher(JointState, 'wheel_commands', 10)
        self.create_timer(0.02, self.publish_wheel_commands)

    def cmd_cb(self, msg):
        left_speed = (
            msg.linear.x - msg.angular.z * self.wheel_separation / 2.0
        ) / self.wheel_radius
        right_speed = (
            msg.linear.x + msg.angular.z * self.wheel_separation / 2.0
        ) / self.wheel_radius

        self.wheel_speeds = [left_speed, left_speed, right_speed, right_speed]
        self.last_command_time = self.get_clock().now()

    def publish_wheel_commands(self):
        age = (self.get_clock().now() - self.last_command_time).nanoseconds / 1e9
        speeds = self.wheel_speeds if age <= self.command_timeout else [0.0] * 4
        command = JointState()
        command.header.stamp = self.get_clock().now().to_msg()
        command.name = list(WHEEL_NAMES)
        command.velocity = speeds
        self.wheel_pub.publish(command)


def main():
    rclpy.init()
    node = BaseController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop_command = JointState()
        stop_command.name = list(WHEEL_NAMES)
        stop_command.velocity = [0.0] * 4
        node.wheel_pub.publish(stop_command)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()