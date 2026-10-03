import rclpy
from rclpy.node import Node
from std_msgs.msg import String

COMMANDS = [
    'forward 0.3 25 1.0',
    #'turn 90',
    'forward 0.3 -25 1.0',
    #'turn -90',
    'forward 0.3 0 1.0',
]


class PathRunner(Node):
    def __init__(self):
        super().__init__('path_runner')
        self.pub = self.create_publisher(String, 'drive_cmd', 10)
        self.timer = self.create_timer(2.0, self.send)  # wait for discovery

    def send(self):
        self.timer.cancel()
        for c in COMMANDS:
            self.pub.publish(String(data=c))
        self.get_logger().info(f'Sent {len(COMMANDS)} commands.')


def main():
    rclpy.init()
    node = PathRunner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()