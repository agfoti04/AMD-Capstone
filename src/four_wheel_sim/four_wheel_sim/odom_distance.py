import math

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node


def valid_position(msg):
    position = msg.pose.pose.position
    orientation = msg.pose.pose.orientation
    values = (position.x, position.y, position.z, orientation.x,
              orientation.y, orientation.z, orientation.w)
    if not msg.header.frame_id or not all(math.isfinite(value) for value in values):
        return None
    if math.sqrt(sum(value * value for value in values[3:])) < 1e-6:
        return None
    return position.x, position.y, position.z


class OdomDistance(Node):
    def __init__(self):
        super().__init__('odom_distance')
        self.origin = None
        self.latest_position = None
        self.previous_position = None
        self.frame_id = None
        self.path_length = 0.0
        self.invalid_pose_warning_shown = False
        self.frame_warning_shown = False
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_timer(1.0, self.report_distance)
        self.get_logger().info('Waiting for valid RTAB-Map odometry on /odom')

    def odom_callback(self, msg):
        position = valid_position(msg)
        if position is None:
            if not self.invalid_pose_warning_shown:
                self.get_logger().warning(
                    'Ignoring invalid odometry pose until tracking is valid')
                self.invalid_pose_warning_shown = True
            return

        self.invalid_pose_warning_shown = False
        if self.origin is None:
            self.origin = position
            self.latest_position = position
            self.previous_position = position
            self.frame_id = msg.header.frame_id
            self.get_logger().info(
                f'Measurement started at first valid pose in {self.frame_id}')
            return

        if msg.header.frame_id != self.frame_id:
            if not self.frame_warning_shown:
                self.get_logger().error(
                    f'Ignoring odometry in {msg.header.frame_id}; expected '
                    f'{self.frame_id}')
                self.frame_warning_shown = True
            return

        self.frame_warning_shown = False
        self.path_length += math.dist(self.previous_position, position)
        self.previous_position = position
        self.latest_position = position

    def report_distance(self):
        if self.origin is None:
            return

        displacement = tuple(
            current - start
            for current, start in zip(self.latest_position, self.origin)
        )
        straight_line_distance = math.dist(self.origin, self.latest_position)
        self.get_logger().info(
            'Displacement since start: '
            f'x={displacement[0]:.3f} m, y={displacement[1]:.3f} m, '
            f'z={displacement[2]:.3f} m; '
            f'straight-line={straight_line_distance:.3f} m, '
            f'accumulated path={self.path_length:.3f} m')


def main(args=None):
    rclpy.init(args=args)
    node = OdomDistance()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.report_distance()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
