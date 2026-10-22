#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float64MultiArray

LO, HI = -1.57, 0.0   # Joint_Gripper limits


class GripperFanout(Node):
    def __init__(self):
        super().__init__('gripper_fanout')
        self.pub = self.create_publisher(
            Float64MultiArray, '/gripper_controller/commands', 10)
        self.create_subscription(Float64, '/gripper_cmd', self.on_cmd, 10)

    def on_cmd(self, msg):
        q = min(max(msg.data, LO), HI)
        # order: Joint_Gripper, grip_right, tendon_left, tendon_right, finger_left, finger_right
        self.pub.publish(Float64MultiArray(data=[q, -q, q, -q, -q, q]))


def main():
    rclpy.init()
    node = GripperFanout()
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