import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray

RATE = 0.5            # rad/s head speed
DRIVE_SPAN = 0.9      # +/- rad while driving (about 50 deg)
IDLE_SPAN = 2.9       # +/- rad when stopped (about 165 deg)


class HeadScanner(Node):
    def __init__(self):
        super().__init__('head_scanner')
        self.pub = self.create_publisher(Float64MultiArray, '/head_controller/commands', 10)
        self.create_subscription(Twist, '/cmd_vel', self.cmd_cb, 10)
        self.driving = False
        self.last_cmd = 0.0
        self.angle, self.dir = 0.0, 1.0
        self.dt = 0.05
        self.create_timer(self.dt, self.tick)

    def cmd_cb(self, m):
        self.driving = abs(m.linear.x) > 0.03
        self.last_cmd = self.get_clock().now().nanoseconds / 1e9

    def tick(self):
        now = self.get_clock().now().nanoseconds / 1e9
        if now - self.last_cmd > 1.0:
            self.driving = False
        span = DRIVE_SPAN if self.driving else IDLE_SPAN
        self.angle += self.dir * RATE * self.dt
        if abs(self.angle) >= span:
            self.angle = math.copysign(span, self.angle)
            self.dir = -self.dir
        self.pub.publish(Float64MultiArray(data=[self.angle]))


def main():
    rclpy.init()
    rclpy.spin(HeadScanner())


if __name__ == '__main__':
    main()