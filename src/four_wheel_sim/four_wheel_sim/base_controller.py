import math
from collections import deque

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class DriveController(Node):
    def __init__(self):
        super().__init__('drive_controller')
        self.declare_parameter('wheel_separation', 0.24)
        self.declare_parameter('heading_kp', 1.5)
        self.declare_parameter('max_turn_rate_deg', 60.0)
        self.declare_parameter('min_turn_rate_deg', 15.0)
        self.declare_parameter('angle_tolerance_deg', 2.0)
        self.sep = self.get_parameter('wheel_separation').value
        self.kp = self.get_parameter('heading_kp').value
        self.max_w = math.radians(self.get_parameter('max_turn_rate_deg').value)
        self.min_w = math.radians(self.get_parameter('min_turn_rate_deg').value)
        self.tol = math.radians(self.get_parameter('angle_tolerance_deg').value)

        self.wz = 0.0
        self.settle = 0 

        self.goal = 0.0

        self.x = self.y = self.yaw = 0.0
        self.have_odom = False
        self.queue = deque()
        self.active = None

        self.create_subscription(String, 'drive_cmd', self.cmd_cb, 10)
        self.create_subscription(Odometry, 'odom', self.odom_cb, 10)
        self.out = self.create_publisher(Twist, 'cmd_vel', 10)
        self.create_timer(0.02, self.tick)  # 50 Hz

    # ---------- input ----------
    def odom_cb(self, msg):
        p = msg.pose.pose
        self.x, self.y = p.position.x, p.position.y
        q = p.orientation
        self.yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                              1 - 2 * (q.y * q.y + q.z * q.z))
        if not self.have_odom:
            self.goal = self.yaw
        self.have_odom = True
        self.wz = msg.twist.twist.angular.z

    def cmd_cb(self, msg):
        parts = msg.data.lower().split()
        if not parts:
            return
        try:
            name = parts[0]
            a = [float(p) for p in parts[1:]]
            if name in ('forward', 'backward'):
                v = abs(a[0]) * (1.0 if name == 'forward' else -1.0)
                cmd = {'kind': 'drive', 'v': v, 'angle': math.radians(a[1]),
                       'dist': a[2] if len(a) > 2 else None}
            elif name == 'turn':
                rate = math.radians(a[1]) if len(a) > 1 else self.max_w
                cmd = {'kind': 'turn', 'angle': math.radians(a[0]),
                       'rate': min(abs(rate), self.max_w)}
            elif name == 'stop':
                self.queue.clear()
                self.active = None
                self.send_sides(0.0, 0.0)
                self.goal = self.yaw
                return
            else:
                raise ValueError(name)
        except (ValueError, IndexError):
            self.get_logger().error(f'Bad command: "{msg.data}"')
            return
        cmd['text'] = msg.data
        self.queue.append(cmd)

    # ---------- control ----------
    def tick(self):
        if not self.have_odom:
            return
        #Settle Phase
        if self.settle > 0:
            self.settle += 1
            self.send_sides(0.0, 0.0)
            stopped = self.settle > 10 and abs(self.wz) < 0.02
            if stopped or self.settle > 150:   # 3 s timeout
                self.settle = 0
            return
        
         # Pop the next command
        if self.active is None:
            if not self.queue:
                return
            self.active = self.queue.popleft()
            c = self.active
            c['start'] = (self.x, self.y)
            c['t0'] = self.get_clock().now()
            if c['kind'] == 'turn':
                self.goal = wrap(self.goal + c['angle'])
                c['heading'] = self.goal
            else:
                c['heading'] = wrap(self.goal + c['angle'])
                self.goal = c['heading']
                c['path'] = 0.0
                c['last'] = (self.x, self.y)
            self.get_logger().info(f'Running: {c["text"]}')

        c = self.active
        err = wrap(c['heading'] - self.yaw)

        # Watchdog and debug log
        elapsed = (self.get_clock().now() - c['t0']).nanoseconds / 1e9
        if elapsed > 30.0:
            self.get_logger().warn(f'Timeout on: {c["text"]}')
            return self.finish()
        self.get_logger().info(
            f'err={math.degrees(err):.1f}deg yaw={math.degrees(self.yaw):.1f}deg '
            f'vx={self.x:.3f} wz={self.wz:.3f}', throttle_duration_sec=1.0)

        if c['kind'] == 'turn':
            # Pivot in place
            if abs(err) < self.tol:
                return self.finish()
            mag = min(max(abs(self.kp * err), self.min_w), c['rate'])
            v, w = 0.0, math.copysign(mag, err)
        else:
            # Curved drive: finish the heading change over the path length
            c['path'] += math.hypot(self.x - c['last'][0], self.y - c['last'][1])
            c['last'] = (self.x, self.y)
            v = c['v']
            if c['dist'] is not None:
                remaining = c['dist'] - c['path']
                if remaining <= 0.0:
                    return self.finish()
                if remaining > 0.1:
                    w = abs(v) * err / remaining
                else:
                    w = self.kp * err      # final stretch: clean up the error
            else:
                w = self.kp * err          # no distance: curve toward heading
            w = max(-self.max_w, min(self.max_w, w))

        # Left/right side speeds (m/s)
        left = v - w * self.sep / 2.0
        right = v + w * self.sep / 2.0
        self.send_sides(left, right)

    def finish(self):
        self.get_logger().info('Done.')
        self.active = None
        self.settle = 1
        self.send_sides(0.0, 0.0)

    def send_sides(self, left, right):
        # The Gazebo plugin only accepts Twist, so convert side speeds back.
        # On real hardware you'd send left/right straight to the motor drivers.
        msg = Twist()
        msg.linear.x = (left + right) / 2.0
        msg.angular.z = (right - left) / self.sep
        self.out.publish(msg)


def main():
    rclpy.init()
    node = DriveController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.send_sides(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()