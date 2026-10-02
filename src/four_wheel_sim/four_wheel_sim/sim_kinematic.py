import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster


WHEEL_NAMES = (
    'front_left_wheel_joint',
    'rear_left_wheel_joint',
    'front_right_wheel_joint',
    'rear_right_wheel_joint',
)


class KinematicSim(Node):
    def __init__(self):
        super().__init__('kinematic_sim')

        #Parameters
        self.declare_parameter('wheel_radius', 0.05)
        self.declare_parameter('wheel_separation', 0.24)
        self.r = self.get_parameter('wheel_radius').value
        self.sep = self.get_parameter('wheel_separation').value

        self.x = self.y = self.th = 0.0
        self.wheel_speeds = [0.0, 0.0, 0.0, 0.0]
        self.wheel_angles = [0.0, 0.0, 0.0, 0.0]

        # Subscribers and publishers
        self.create_subscription(
            JointState, 'wheel_commands', self.wheel_cb, 10)
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.odom_pub = self.create_publisher(Odometry, 'odom', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.last = self.get_clock().now()
        self.create_timer(0.02, self.update)  # 50 Hz


    def wheel_cb(self, msg):
        if len(msg.name) != len(msg.velocity):
            self.get_logger().warning('Wheel command names and velocities differ in length')
            return
        speeds_by_name = dict(zip(msg.name, msg.velocity))
        if not all(name in speeds_by_name for name in WHEEL_NAMES):
            self.get_logger().warning('Wheel command is missing one or more wheel joints')
            return
        self.wheel_speeds = [speeds_by_name[name] for name in WHEEL_NAMES]

    
    def update(self):
        now = self.get_clock().now()
        dt = (now - self.last).nanoseconds / 1e9
        self.last = now

        left_w = (self.wheel_speeds[0] + self.wheel_speeds[1]) / 2.0
        right_w = (self.wheel_speeds[2] + self.wheel_speeds[3]) / 2.0
        v = self.r * (left_w + right_w) / 2.0
        w = self.r * (right_w - left_w) / self.sep

        self.x += v * math.cos(self.th) * dt
        self.y += v * math.sin(self.th) * dt
        self.th += w * dt
        self.wheel_angles = [
            angle + speed * dt
            for angle, speed in zip(self.wheel_angles, self.wheel_speeds)
        ]

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
        odom.twist.twist.linear.x = v
        odom.twist.twist.angular.z = w
        self.odom_pub.publish(odom)

        # Joint states so the wheels spin
        js = JointState()
        js.header.stamp = stamp
        js.name = ['front_left_wheel_joint', 'rear_left_wheel_joint',
                   'front_right_wheel_joint', 'rear_right_wheel_joint']
        js.position = self.wheel_angles
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