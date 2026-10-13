"""Live HTTP pipeline tests; ROS interfaces/model output are controlled doubles."""
import json
import sys
import tempfile
import threading
import time
import types
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from rover_perception.live_core import DriveLease, image_to_bgr
from rover_perception.live_dashboard import LiveSession, dashboard_handler


def twist():
    return types.SimpleNamespace(linear=types.SimpleNamespace(x=0.0), angular=types.SimpleNamespace(z=0.0))


# Load the ROS adapter without requiring ROS on a developer's Windows machine.
ros = {name: types.ModuleType(name) for name in ('rclpy', 'rclpy.node', 'rclpy.qos',
                                               'geometry_msgs', 'geometry_msgs.msg', 'sensor_msgs', 'sensor_msgs.msg')}
ros['rclpy.node'].Node = object
ros['rclpy.qos'].qos_profile_sensor_data = object()
ros['geometry_msgs.msg'].Twist = twist
ros['sensor_msgs.msg'].Image = object
with patch.dict(sys.modules, ros):
    from rover_perception.live_bridge import LiveBridge, handler as bridge_handler


def image_message(rgb=None, encoding='rgb8', padding=2):
    rgb = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [180, 130, 10]]], dtype=np.uint8) if rgb is None else rgb
    height, width = rgb.shape[:2]
    data = b''.join(row.tobytes() + b'\xff' * padding for row in rgb)
    return types.SimpleNamespace(width=width, height=height, step=width * 3 + padding, encoding=encoding,
                                 data=data, header=types.SimpleNamespace(stamp=types.SimpleNamespace(sec=42, nanosec=500)))


class FakeModel:
    def __init__(self):
        self.calls = []
        self.fail = False
        self.names = {0: 'rock', 1: 'pale_soil_patch', 2: 'yellow_crystal_rock'}

    def predict(self, image, **options):
        if self.fail:
            raise RuntimeError('Controlled inference failure')
        self.calls.append((image.copy(), options))
        boxes = [types.SimpleNamespace(cls=[index], conf=[0.8 + index * 0.05],
                                        xyxy=np.array([[0, 0, 1, 1]])) for index in range(3)]
        return [types.SimpleNamespace(boxes=boxes, names=self.names, plot=lambda: image.copy())]


def run_server(handler):
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f'http://127.0.0.1:{server.server_port}'


def post(url, payload, **headers):
    request = urllib.request.Request(url, json.dumps(payload).encode(),
                                     dict({'Content-Type': 'application/json'}, **headers), method='POST')
    with urllib.request.urlopen(request, timeout=2) as response:
        return json.load(response)


def await_condition(predicate):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError('Condition did not become true')


class CoreTests(unittest.TestCase):
    def test_padded_rgb_and_bgr_have_same_correct_colors(self):
        rgb = image_message()
        actual = image_to_bgr(rgb)
        np.testing.assert_array_equal(actual[0, 0], [0, 0, 255])
        bgr = image_message(actual, 'bgr8', 5)
        np.testing.assert_array_equal(image_to_bgr(bgr), actual)
        self.assertTrue(actual.flags.c_contiguous)

    def test_invalid_encoding_and_buffer_are_rejected(self):
        for change in ({'encoding': 'mono8'}, {'step': 1}, {'data': b'bad'}):
            msg = image_message()
            vars(msg).update(change)
            with self.assertRaises(ValueError):
                image_to_bgr(msg)

    def test_drive_clamps_and_expires_on_wall_time(self):
        now = [10.0]
        lease = DriveLease(clock=lambda: now[0])
        lease.set(12, -5)
        self.assertEqual(lease.get(), (0.3, -0.8))
        now[0] += 0.36
        self.assertEqual(lease.get(), (0, 0))
        lease.set(0, 0)
        self.assertEqual(lease.get(), (0, 0))

    def test_invalid_drive_inputs_are_rejected(self):
        for value in (float('nan'), float('inf'), True, 'invalid'):
            with self.assertRaises(ValueError):
                DriveLease().set(value, 0)


class LiveHTTPTests(unittest.TestCase):
    def setUp(self):
        self.node = LiveBridge.__new__(LiveBridge)
        self.node.lock = threading.Lock()
        self.node.frame, self.node.sequence, self.node.received = image_message(), 1, time.monotonic()
        self.node.lease = DriveLease()
        self.published = []
        self.node.pub = types.SimpleNamespace(publish=self.published.append)
        self.bridge, self.url = run_server(bridge_handler(self.node))
        self.model = FakeModel()
        self.session = LiveSession(self.model, self.url, fps=20)
        page = Path(__file__).resolve().parents[1] / 'rover_perception' / 'live_dashboard.html'
        self.dashboard, self.web = run_server(dashboard_handler(self.session, page.read_bytes()))

    def tearDown(self):
        self.session.close()
        for server in (self.dashboard, self.bridge):
            server.shutdown()
            server.server_close()

    def process_frame(self):
        self.session.thread.start()
        await_condition(lambda: self.session.snapshot()['ready'])

    def test_actual_http_pipeline_preserves_colors_and_three_class_names(self):
        self.process_frame()
        image, settings = self.model.calls[0]
        np.testing.assert_array_equal(image, image_to_bgr(self.node.frame))
        self.assertEqual(settings, dict(imgsz=960, conf=0.25, device='0', rect=False, verbose=False))
        with urllib.request.urlopen(self.web + '/status') as response:
            status = json.load(response)
        self.assertEqual([row['class_name'] for row in status['detections']], list(self.model.names.values()))
        self.assertEqual(status['sim_time'], 42.0000005)
        self.assertEqual(len(self.model.calls), 1)

    def test_driving_without_ready_camera_is_zero_then_commands_and_release_work(self):
        post(self.web + '/command', dict(linear=0.3, angular=0))
        self.assertEqual(self.node.lease.get(), (0, 0))
        self.process_frame()
        post(self.web + '/command', dict(linear=10, angular=-10))
        self.node.drive()
        self.assertEqual((self.published[-1].linear.x, self.published[-1].angular.z), (0.3, -0.8))
        post(self.web + '/command', dict(linear=0, angular=0))
        self.node.drive()
        self.assertEqual(self.published[-1].linear.x, 0)

    def test_stale_camera_stops_rover_even_with_renewed_command(self):
        self.process_frame()
        self.node.received = time.monotonic() - 2
        self.node.lease.set(0.3, 0.8)
        self.node.drive()
        self.assertEqual((self.published[-1].linear.x, self.published[-1].angular.z), (0, 0))
        await_condition(lambda: not self.session.snapshot()['ready'])
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(self.url + '/frame.png')
        self.assertEqual(error.exception.code, 503)

    def test_inference_failure_disables_controls(self):
        self.process_frame()
        self.model.fail = True
        self.node.sequence += 1
        self.node.received = time.monotonic()
        await_condition(lambda: bool(self.session.snapshot()['error']))
        post(self.web + '/command', dict(linear=0.3, angular=0))
        self.assertEqual(self.node.lease.get(), (0, 0))

    def test_malformed_command_and_foreign_origin_are_rejected(self):
        for url, headers in ((self.web, {'Origin': 'http://example.com'}), (self.url, {})):
            self.node.lease.set(0.3, 0)
            with self.assertRaises(urllib.error.HTTPError) as error:
                post(url + '/command', dict(linear='nan', angular=0), **headers)
            self.assertEqual(error.exception.code, 400)
            self.assertEqual(self.node.lease.get(), (0, 0))

    def test_recording_contains_raw_image_and_prediction_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            self.session.record = Path(folder)
            self.process_frame()
            files = list(Path(folder).glob('*.png'))
            self.assertEqual(len(files), 1)
            np.testing.assert_array_equal(cv2.imread(str(files[0])), image_to_bgr(self.node.frame))
            prediction = json.loads((Path(folder) / 'predictions.jsonl').read_text().strip())
            self.assertEqual(len(prediction['detections']), 3)
            self.assertEqual(prediction['image'], files[0].name)
            self.assertEqual(len(list(Path(folder).glob('*_boxes.jpg'))), 1)

    def test_stream_and_dashboard_are_served(self):
        self.process_frame()
        with urllib.request.urlopen(self.web + '/') as response:
            self.assertIn(b'Rover live vision', response.read())
        with urllib.request.urlopen(self.web + '/stream.mjpg', timeout=2) as response:
            data = response.read(80)
            self.assertIn(b'Content-Type: image/jpeg', data)

    def test_same_frame_is_not_repeatedly_inferred(self):
        self.process_frame()
        time.sleep(0.15)
        self.assertEqual(len(self.model.calls), 1)
        self.node.on_image(image_message())
        await_condition(lambda: len(self.model.calls) == 2)


if __name__ == '__main__':
    unittest.main()
