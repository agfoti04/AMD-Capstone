"""Local PNG camera/drive bridge; GPU inference stays in the host Conda environment."""
import json
import socket
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from .live_core import DEFAULT_CAMERA_TIMEOUT, DriveLease, camera_timeout, image_to_bgr
from .drive_mode import control_mode


class LiveBridge(Node):
    def __init__(self):
        super().__init__('rover_live_bridge')
        self.declare_parameter('image_topic', '/oak/image_raw')
        self.declare_parameter('port', 8770)
        self.declare_parameter('camera_timeout_sec', DEFAULT_CAMERA_TIMEOUT)
        self.declare_parameter('control_mode', 'browser')
        self.control_mode = control_mode(self.get_parameter('control_mode').value)
        self.camera_timeout = camera_timeout(self.get_parameter('camera_timeout_sec').value)
        self.stream_id = uuid.uuid4().hex
        self.lease = DriveLease()
        self.lock = threading.Lock()
        self.frame = None
        self.sequence = 0
        self.received = 0.0
        self.pub = None
        if self.control_mode == 'browser':
            self.pub = self.create_publisher(Twist, '/cmd_vel', 1)
            # This node deliberately uses its default wall clock, not use_sim_time.
            self.create_timer(0.02, self.drive)
        self.create_subscription(Image, self.get_parameter('image_topic').value,
                                 self.on_image, QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
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
        if self.control_mode != 'browser':
            return
        _, _, received = self.snapshot()
        linear, angular = self.lease.get()
        if not received or time.monotonic() - received >= self.camera_timeout:
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
        protocol_version = 'HTTP/1.1'

        def setup(self):
            super().setup()
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.connection.settimeout(3)

        def log_message(self, *args):
            pass

        def reply(self, status, body, kind='application/json', headers=None):
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Control-Mode', node.control_mode)
            for key, value in (headers or {}).items():
                self.send_header(key, str(value))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            frame, sequence, received = node.snapshot()
            age = time.monotonic() - received if received else None
            if self.path == '/status':
                self.reply(200, json.dumps({'sequence': sequence, 'camera_age_seconds': age,
                                           'camera_timeout_seconds': node.camera_timeout,
                                           'camera_ready': frame is not None and age < node.camera_timeout,
                                           'control_mode': node.control_mode,
                                           'controls_enabled': node.control_mode == 'browser',
                                           'command': node.lease.get() if node.control_mode == 'browser' else None}).encode())
            elif self.path == '/frame.png':
                if frame is None or age >= node.camera_timeout:
                    message = ('Waiting for the simulator camera.' if frame is None else
                               'Camera stopped updating. Check Gazebo is playing.' +
                               (' Browser driving is paused.' if node.control_mode == 'browser' else ''))
                    self.reply(503, json.dumps({'error': message, 'camera_age_seconds': age,
                                                'camera_timeout_seconds': node.camera_timeout}).encode())
                    return
                etag = f'"{node.stream_id}:{sequence}"'
                if self.headers.get('If-None-Match') == etag:
                    # The camera is still within its freshness limit. Let the
                    # consumer wait for a new sample without encoding the same PNG.
                    self.reply(304, b'', headers={'ETag': etag})
                    return
                try:
                    ok, encoded = cv2.imencode('.png', image_to_bgr(frame), [cv2.IMWRITE_PNG_COMPRESSION, 1])
                    if not ok:
                        raise ValueError('PNG encoding failed')
                    stamp = frame.header.stamp
                    self.reply(200, encoded.tobytes(), 'image/png',
                               {'X-Frame-Sequence': sequence,
                                'X-Frame-Age': time.monotonic() - received,
                                'ETag': etag,
                                'X-Camera-Timeout': node.camera_timeout,
                                'X-Sim-Time': stamp.sec + stamp.nanosec / 1e9})
                except ValueError as error:
                    self.reply(422, json.dumps({'error': str(error)}).encode())
            else:
                self.reply(404, b'{}')

        def do_POST(self):
            if node.control_mode != 'browser':
                self.close_connection = True
                self.reply(403, b'{"error":"Driving is controlled in Gazebo; this bridge is a camera viewer."}')
                return
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
                self.close_connection = True
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
