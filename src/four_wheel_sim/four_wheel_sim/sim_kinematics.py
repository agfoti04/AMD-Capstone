import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster


class KinematicSim(Node):
    def __init__(self):
        super().__init__('kinematic_sim')

        #Parameters
        self.declare_parameter('wheel_radius', 0.05)
        self.declare_parameter('wheel_separation', 0.24)
        self.r = self.get_parameter('wheel_radius').value
        self.sep = self.get_parameter('wheel_separation').value

        self.x = self.y = self.th = 0.0
        self.v = self.w = 0.0
        self.left_angle = self.right_angle = 0.0

        # Subscribers and publishers
        self.create_subscription(Twist, 'cmd_vel', self.cmd_cb, 10)
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.odom_pub = self.create_publisher(Odometry, 'odom', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.last = self.get_clock().now()
        self.create_timer(0.02, self.update)  # 50 Hz


    def cmd_cb(self, msg):
        self.v = msg.linear.x
        self.w = msg.angular.z

    
    def update(self):
        now = self.get_clock().now()
        dt = (now - self.last).nanoseconds / 1e9
        self.last = now

        # Integrate pose
        self.x += self.v * math.cos(self.th) * dt
        self.y += self.v * math.sin(self.th) * dt
        self.th += self.w * dt

        # Wheel angles (skid-steer: both wheels on a side spin together)
        left_w = (self.v - self.w * self.sep / 2.0) / self.r
        right_w = (self.v + self.w * self.sep / 2.0) / self.r
        self.left_angle += left_w * dt
        self.right_angle += right_w * dt

        stamp = now.to_msg()
        qz, qw = math.sin(self.th / 2.0), math.cos(self.th / 2.0)

        # TF: odom -> base_link
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(t)

        # Odometry
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = self.v
        odom.twist.twist.angular.z = self.w
        self.odom_pub.publish(odom)

        # Joint states so the wheels spin
        js = JointState()
        js.header.stamp = stamp
        js.name = ['front_left_wheel_joint', 'rear_left_wheel_joint',
                   'front_right_wheel_joint', 'rear_right_wheel_joint']
        js.position = [self.left_angle, self.left_angle,
                       self.right_angle, self.right_angle]
        self.joint_pub.publish(js)


def main():
    rclpy.init()
    node = KinematicSim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()