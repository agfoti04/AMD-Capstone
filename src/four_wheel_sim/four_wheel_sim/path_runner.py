import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


def forward(dist, speed=0.3):
    return (math.copysign(speed, dist), 0.0, abs(dist) / speed)


def turn(angle, rate=0.5):
    return (0.0, math.copysign(rate, angle), abs(angle) / rate)


PATH = [
    forward(2.0),
    turn(math.pi / 2),    # left 90°
    forward(2.0),
    turn(-math.pi / 2),   # right 90°
    forward(2.0),
]


class PathRunner(Node):
    def __init__(self):
        super().__init__('path_runner')
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.t0 = self.get_clock().now()
        self.seg_start = None
        self.i = 0
        self.start_delay = 3.0  # seconds, so RViz is ready
        self.create_timer(0.05, self.tick)  # 20 Hz

    def elapsed(self, since):
        return (self.get_clock().now() - since).nanoseconds / 1e9

    def tick(self):
        if self.elapsed(self.t0) < self.start_delay:
            return
        if self.i >= len(PATH):
            self.pub.publish(Twist())  # hold stopped
            return
        if self.seg_start is None:
            self.seg_start = self.get_clock().now()

        v, w, dur = PATH[self.i]
        if self.elapsed(self.seg_start) >= dur:
            self.i += 1
            self.seg_start = self.get_clock().now()
            if self.i >= len(PATH):
                self.get_logger().info('Path complete.')
                self.pub.publish(Twist())
                return
            v, w, dur = PATH[self.i]
            self.get_logger().info(f'Segment {self.i}: v={v:.2f} w={w:.2f} for {dur:.1f}s')

        msg = Twist()
        msg.linear.x = v
        msg.angular.z = w
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = PathRunner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()