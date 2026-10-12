import math

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image
from tf2_ros import Buffer, TransformListener

STOP_DIST = 0.6               # m, obstacle distance that triggers avoidance
TURN_MIN, TURN_MAX = 0.6, 1.0  # rad/s, spin speeds that actually move the robot


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class Rover(Node):
    def __init__(self):
        super().__init__('rover_mission')
        defaults = dict(target_x=2.0, target_y=-3.0, grid_w=8.0, grid_h=6.0,
                lane_spacing=2.0, speed=0.2, sweep=True, frame='odom')
        for k, v in defaults.items():
            self.declare_parameter(k, v)
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.create_subscription(Image, '/oak/depth/image_raw', self.depth_cb, 1)
        self.buf = Buffer()
        TransformListener(self.buf, self)
        self.frame = self.par('frame')
        self.clear = {'L': 9.0, 'C': 9.0, 'R': 9.0}   # nearest obstacle per sector
        self.side, self.avoid_n, self.last_avoid = 1, 0, -100.0

    # ---------- helpers ----------
    def par(self, k):
        return self.get_parameter(k).value

    def now(self):
        return self.get_clock().now().nanoseconds / 1e9

    def tick(self, dt=0.05):
        rclpy.spin_once(self, timeout_sec=dt)

    def send(self, v=0.0, w=0.0):
        m = Twist()
        m.linear.x, m.angular.z = float(v), float(w)
        self.pub.publish(m)

    def hold(self, sec):
        t0 = self.now()
        while rclpy.ok() and self.now() - t0 < sec:
            self.send()
            self.tick()

    def pose(self):
        try:
            t = self.buf.lookup_transform(self.frame, 'base_link', Time())
        except Exception:
            return None
        q, p = t.transform.rotation, t.transform.translation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return p.x, p.y, yaw

    def depth_cb(self, m):
        h, w = m.height, m.width
        if m.encoding == '32FC1':
            d = np.frombuffer(bytes(m.data), np.float32).reshape(h, w)
        else:
            d = np.frombuffer(bytes(m.data), np.uint16).reshape(h, w) / 1000.0
        band = d[int(0.1 * h):int(0.4 * h)]     # upper half, so the floor isn't seen
        for key, cols in zip('LCR', (slice(0, w // 3), slice(w // 3, 2 * w // 3),
                                     slice(2 * w // 3, w))):
            s = band[:, cols]
            s = s[np.isfinite(s) & (s > 0.3)]
            self.clear[key] = float(np.percentile(s, 5)) if s.size > 50 else 9.0

    def blocked(self):
        c = self.clear
        return c['C'] < STOP_DIST or min(c['L'], c['R']) < STOP_DIST * 0.7

    # ---------- motion primitives ----------
    def rotate_to(self, heading, tol=0.07):
        for _ in range(3):
            t0 = self.now()
            while rclpy.ok():
                self.tick()
                p = self.pose()
                if p is None:
                    continue
                err = wrap(heading - p[2])
                if int(self.now() * 2) != getattr(self, '_rlog', -1):
                    self._rlog = int(self.now() * 2)
                    self.get_logger().info(f'rotate: yaw={p[2]:.2f} target={heading:.2f} err={err:.2f}')
                if abs(err) < tol or self.now() - t0 > 20.0:
                    break
                w = min(TURN_MAX, max(TURN_MIN, 1.5 * abs(err)))
                self.send(w=math.copysign(w, err))
            self.hold(0.6)
            p = self.pose()
            if p and abs(wrap(heading - p[2])) < 2 * tol:
                return
            

    def move(self, v, dist, watch=False):
        p0 = self.pose()
        t0 = self.now()
        while rclpy.ok():
            self.tick()
            p = self.pose()
            if p is None:
                continue
            if math.hypot(p[0] - p0[0], p[1] - p0[1]) >= dist:
                break
            if self.now() - t0 > dist / abs(v) * 3 + 2:
                break
            if watch and v > 0 and self.blocked():
                break
            self.send(v=v)
        self.hold(0.4)

    def avoid(self):
        c, t = self.clear, self.now()
        if t - self.last_avoid < 20.0:
            self.avoid_n += 1                    # same obstacle again: same side, farther
        else:
            self.avoid_n = 0
            self.side = 1 if c['L'] >= c['R'] else -1
        self.last_avoid = t
        self.get_logger().info(f'Obstacle at {c["C"]:.2f} m: avoiding '
                               f'({"left" if self.side > 0 else "right"})')
        self.move(-0.15, 0.25)
        yaw = self.pose()[2]
        self.rotate_to(wrap(yaw + self.side * math.pi / 2))
        self.move(0.2, min(0.8 + 0.4 * self.avoid_n, 2.4), watch=True)

    def drive_to(self, x, y, tol=0.25, timeout=150.0):
        t0 = self.now()
        while rclpy.ok() and self.now() - t0 < timeout:
            self.tick()
            p = self.pose()
            if p is None:
                continue
            dx, dy = x - p[0], y - p[1]
            dist = math.hypot(dx, dy)
            if dist < tol:
                break
            bearing = math.atan2(dy, dx)
            err = wrap(bearing - p[2])
            if abs(err) > 0.2:
                self.hold(0.3)
                self.rotate_to(bearing)
                continue
            if self.blocked():
                self.hold(0.3)
                self.avoid()
                continue
            v = min(self.par('speed'), max(0.12, dist))
            self.send(v=v, w=max(-0.4, min(0.4, 1.5 * err)))
        else:
            self.get_logger().warn(f'Timeout driving to ({x:.1f}, {y:.1f}), moving on')
        self.hold(0.5)

    def sweep(self):
        base = self.pose()[2]
        for off in (0.7, -0.7, 0.0):             # look left, look right, back
            self.rotate_to(wrap(base + off))

    # ---------- mission ----------
    def lawnmower(self):
        x0, y0 = self.par('target_x'), self.par('target_y')
        x1, y1 = x0 + self.par('grid_w'), y0 + self.par('grid_h')
        pts, y, i = [], y0, 0
        while y <= y1 + 1e-6:
            xs = (x0, x1) if i % 2 == 0 else (x1, x0)
            pts += [(xs[0], y), (xs[1], y)]
            y, i = y + self.par('lane_spacing'), i + 1
        return pts

    def wait_for_pose(self):
        t0 = self.now()
        while rclpy.ok():
            self.tick(0.2)
            if self.pose():
                return
            if self.frame == 'map' and self.now() - t0 > 40.0:
                self.get_logger().warn('No map frame, using odom instead')
                self.frame, t0 = 'odom', self.now()

    def run(self):
        self.wait_for_pose()
        hx, hy, hyaw = self.pose()
        self.get_logger().info(f'Home = ({hx:.2f}, {hy:.2f}) in frame {self.frame}')
        self.get_logger().info('Driving to grid start')
        self.drive_to(self.par('target_x'), self.par('target_y'))
        pts = self.lawnmower()
        for n, (x, y) in enumerate(pts, 1):
            self.get_logger().info(f'Lawnmower {n}/{len(pts)}: ({x:.1f}, {y:.1f})')
            self.drive_to(x, y)
            if self.par('sweep'):
                self.sweep()
        self.get_logger().info('Grid done, returning home')
        self.drive_to(hx, hy)
        self.rotate_to(hyaw)
        self.get_logger().info('Home. Mission complete.')


def main():
    rclpy.init()
    node = Rover()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.send()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()