"""Local PNG camera/drive bridge; GPU inference stays in the host Conda environment."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from .live_core import DriveLease, image_to_bgr


class LiveBridge(Node):
    def __init__(self):
        super().__init__('rover_live_bridge')
        self.declare_parameter('image_topic', '/oak/image_raw')
        self.declare_parameter('port', 8770)
        self.lease = DriveLease()
        self.lock = threading.Lock()
        self.frame = None
        self.sequence = 0
        self.received = 0.0
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Image, self.get_parameter('image_topic').value,
                                 self.on_image, qos_profile_sensor_data)
        # This node deliberately uses its default wall clock, not use_sim_time.
        self.create_timer(0.05, self.drive)
        self.http = ThreadingHTTPServer(('127.0.0.1', self.get_parameter('port').value), handler(self))
        self.http.daemon_threads = True
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.thread.start()
        self.get_logger().info(f'Camera bridge: http://127.0.0.1:{self.http.server_port}')

    def on_image(self, msg):
        # Encode lazily for each requested frame, rather than every 30-Hz sensor sample.
        with self.lock:
            self.frame = msg
            self.sequence += 1
            self.received = time.monotonic()

    def snapshot(self):
        with self.lock:
            return self.frame, self.sequence, self.received

    def drive(self):
        _, _, received = self.snapshot()
        linear, angular = self.lease.get()
        if not received or time.monotonic() - received > 1.0:
            linear, angular = 0.0, 0.0
        msg = Twist()
        msg.linear.x, msg.angular.z = linear, angular
        self.pub.publish(msg)

    def close(self):
        self.lease.set(0, 0)
        self.drive()
        self.http.shutdown()
        self.http.server_close()


def handler(node):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, body, kind='application/json', headers=None):
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            for key, value in (headers or {}).items():
                self.send_header(key, str(value))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            frame, sequence, received = node.snapshot()
            age = time.monotonic() - received if received else None
            if self.path == '/status':
                self.reply(200, json.dumps({'sequence': sequence, 'camera_age_seconds': age,
                                           'command': node.lease.get()}).encode())
            elif self.path == '/frame.png':
                if frame is None or age > 1.0:
                    self.reply(503, b'{"error":"Waiting for a fresh camera image"}')
                    return
                try:
                    ok, encoded = cv2.imencode('.png', image_to_bgr(frame), [cv2.IMWRITE_PNG_COMPRESSION, 1])
                    if not ok:
                        raise ValueError('PNG encoding failed')
                    stamp = frame.header.stamp
                    self.reply(200, encoded.tobytes(), 'image/png',
                               {'X-Frame-Sequence': sequence,
                                'X-Frame-Age': time.monotonic() - received,
                                'X-Sim-Time': stamp.sec + stamp.nanosec / 1e9})
                except ValueError as error:
                    self.reply(422, json.dumps({'error': str(error)}).encode())
            else:
                self.reply(404, b'{}')

        def do_POST(self):
            try:
                if self.path != '/command' or self.headers.get('Content-Type') != 'application/json':
                    raise ValueError('Expected JSON /command')
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 1024 or self.headers.get('Origin'):
                    raise ValueError('Invalid command request')
                command = json.loads(self.rfile.read(size))
                node.lease.set(command['linear'], command['angular'])
                self.reply(200, b'{"ok":true}')
            except (ValueError, TypeError, KeyError) as error:
                node.lease.set(0, 0)
                self.reply(400, json.dumps({'error': str(error)}).encode())
    return Handler


def main():
    rclpy.init()
    node = LiveBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
