"""Forward native Gazebo Teleop commands through the existing ROS rover controls."""
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

from .drive_mode import GAZEBO_COMMAND_TIMEOUT, GAZEBO_DRIVE_TOPIC
from .live_core import DriveLease


class GazeboDrive(Node):
    def __init__(self):
        super().__init__('rover_gazebo_drive')
        # Native Teleop publishes key events, including keyboard auto-repeat.
        # Allow the initial repeat delay, then expire if events stop arriving.
        self.lease = DriveLease(timeout=GAZEBO_COMMAND_TIMEOUT, max_linear=0.3, max_angular=0.8)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Twist, GAZEBO_DRIVE_TOPIC, self.on_command, 10)
        # Default wall clock: expiry still works if Gazebo is paused.
        self.create_timer(0.05, self.drive)
        self.get_logger().info(f'Gazebo keyboard controls: {GAZEBO_DRIVE_TOPIC}; command timeout 1 s')

    def on_command(self, msg):
        try:
            self.lease.set(msg.linear.x, msg.angular.z)
        except (ValueError, TypeError):
            self.lease.set(0, 0)

    def drive(self):
        msg = Twist()
        msg.linear.x, msg.angular.z = self.lease.get()
        self.pub.publish(msg)

    def close(self):
        self.lease.set(0, 0)
        self.drive()


def main():
    rclpy.init()
    node = GazeboDrive()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
