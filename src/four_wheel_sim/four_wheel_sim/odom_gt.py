import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


class OdomGT(Node):
    def __init__(self):
        super().__init__('odom_gt')
        self.tf = TransformBroadcaster(self)
        self.pub = self.create_publisher(Odometry, 'odom', 10)
        self.create_subscription(Odometry, 'odom_gt', self.cb, 10)

    def cb(self, msg):
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_link'
        self.pub.publish(msg)

        t = TransformStamped()
        t.header = msg.header
        t.child_frame_id = 'base_link'
        p = msg.pose.pose
        t.transform.translation.x = p.position.x
        t.transform.translation.y = p.position.y
        t.transform.translation.z = p.position.z
        t.transform.rotation = p.orientation
        self.tf.sendTransform(t)


def main():
    rclpy.init()
    node = OdomGT()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()