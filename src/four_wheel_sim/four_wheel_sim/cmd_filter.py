import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

MIN_W, MIN_V = 0.7, 0.10   # smallest commands that actually move the robot
DEAD_W, DEAD_V = 0.03, 0.02  # below this, treat as "stop"


class CmdFilter(Node):
    def __init__(self):
        super().__init__('cmd_filter')
        self.pub = self.create_publisher(
            Twist, '/diff_drive_controller/cmd_vel_unstamped', 10)
        self.create_subscription(Twist, '/cmd_vel', self.cb, 10)

    def cb(self, m):
        w, v = m.angular.z, m.linear.x
        if abs(w) < DEAD_W:
            m.angular.z = 0.0
        elif abs(w) < MIN_W:
            m.angular.z = MIN_W if w > 0 else -MIN_W
        if abs(v) < DEAD_V:
            m.linear.x = 0.0
        elif abs(v) < MIN_V:
            m.linear.x = MIN_V if v > 0 else -MIN_V
        self.pub.publish(m)


def main():
    rclpy.init()
    rclpy.spin(CmdFilter())


if __name__ == '__main__':
    main()