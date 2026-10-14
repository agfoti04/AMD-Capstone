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

from rover_perception.live_core import DEFAULT_CAMERA_TIMEOUT, DriveLease, camera_timeout, image_to_bgr
from rover_perception.live_dashboard import LiveSession, dashboard_handler
from rover_perception.drive_mode import GAZEBO_COMMAND_TIMEOUT, GAZEBO_DRIVE_TOPIC, control_mode


def twist():
    return types.SimpleNamespace(linear=types.SimpleNamespace(x=0.0), angular=types.SimpleNamespace(z=0.0))


class FakeRosNode:
    overrides = {}

    def __init__(self, name):
        self.parameters = dict(self.overrides)
        self.publishers, self.timers, self.subscriptions = [], [], []

    def declare_parameter(self, name, value):
        self.parameters.setdefault(name, value)

    def get_parameter(self, name):
        return types.SimpleNamespace(value=self.parameters[name])

    def create_publisher(self, kind, topic, queue):
        publisher = types.SimpleNamespace(publish=lambda msg: None)
        self.publishers.append(topic)
        return publisher

    def create_timer(self, interval, callback):
        self.timers.append((interval, callback))

    def create_subscription(self, kind, topic, callback, qos):
        self.subscriptions.append(topic)

    def get_logger(self):
        return types.SimpleNamespace(info=lambda message: None)


# Load the ROS adapter without requiring ROS on a developer's Windows machine.
ros = {name: types.ModuleType(name) for name in ('rclpy', 'rclpy.node', 'rclpy.qos',
                                               'geometry_msgs', 'geometry_msgs.msg', 'sensor_msgs', 'sensor_msgs.msg')}
ros['rclpy.node'].Node = FakeRosNode
ros['rclpy.qos'].QoSProfile = lambda **kwargs: types.SimpleNamespace(**kwargs)
ros['rclpy.qos'].ReliabilityPolicy = types.SimpleNamespace(BEST_EFFORT='best_effort')
ros['geometry_msgs.msg'].Twist = twist
ros['sensor_msgs.msg'].Image = object
with patch.dict(sys.modules, ros):
    from rover_perception.live_bridge import LiveBridge, handler as bridge_handler
    from rover_perception.gazebo_drive import GazeboDrive


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
        self.assertEqual(lease.get(), (0.6, -1.5))
        now[0] += 0.36
        self.assertEqual(lease.get(), (0, 0))
        lease.set(0, 0)
        self.assertEqual(lease.get(), (0, 0))

    def test_invalid_drive_inputs_are_rejected(self):
        for value in (float('nan'), float('inf'), True, 'invalid'):
            with self.assertRaises(ValueError):
                DriveLease().set(value, 0)

    def test_camera_timeout_is_finite_bounded_and_separate_from_drive_lease(self):
        self.assertEqual(camera_timeout(3), 3.0)
        for value in (float('nan'), float('inf'), True, 0, 11, 'invalid'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                camera_timeout(value)
        now = [10.0]
        lease = DriveLease(clock=lambda: now[0])
        lease.set(0.3, 0)
        now[0] += 0.36
        self.assertEqual(lease.get(), (0, 0))


class LiveHTTPTests(unittest.TestCase):
    def setUp(self):
        self.node = LiveBridge.__new__(LiveBridge)
        self.node.lock = threading.Lock()
        self.node.frame, self.node.sequence, self.node.received = image_message(), 1, time.monotonic()
        self.node.lease = DriveLease()
        self.node.camera_timeout = DEFAULT_CAMERA_TIMEOUT
        self.node.control_mode = 'browser'
        self.node.stream_id = 'test-stream'
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
        self.assertEqual(settings, dict(imgsz=960, conf=0.25, device='0', rect=True, verbose=False))
        with urllib.request.urlopen(self.web + '/status') as response:
            status = json.load(response)
        self.assertEqual([row['class_name'] for row in status['detections']], list(self.model.names.values()))
        self.assertEqual(status['sim_time'], 42.0000005)
        self.assertEqual(len(self.model.calls), 1)

    def test_raw_camera_transport_preserves_pixels_without_png_compression(self):
        with patch('rover_perception.live_bridge.cv2.imencode') as encode:
            with urllib.request.urlopen(self.url + '/frame.bgr') as response:
                self.assertEqual(response.headers['X-Image-Width'], '2')
                self.assertEqual(response.headers['X-Image-Height'], '2')
                pixels = np.frombuffer(response.read(), dtype=np.uint8).reshape(2, 2, 3)
            np.testing.assert_array_equal(pixels, image_to_bgr(self.node.frame))
            encode.assert_not_called()
        with urllib.request.urlopen(self.url + '/status') as response:
            status = json.load(response)
        self.assertEqual(status['pipeline_version'], 2)
        self.assertEqual(status['image_size'], [2, 2])

    def test_outdated_installed_bridge_is_reported_instead_of_using_old_camera_path(self):
        class OldBridge(bridge_handler(self.node)):
            def do_GET(handler):
                if handler.path == '/frame.bgr':
                    handler.reply(404, b'{}')
                else:
                    super().do_GET()
        self.bridge.RequestHandlerClass = OldBridge
        self.session.thread.start()
        await_condition(lambda: 'outdated' in self.session.snapshot().get('error', ''))
        self.assertFalse(self.session.snapshot()['ready'])
        self.assertIn('tar --touch', self.session.snapshot()['error'])
        self.assertEqual(len(self.model.calls), 0)

    def test_driving_without_ready_camera_is_zero_then_commands_and_release_work(self):
        post(self.web + '/command', dict(linear=0.3, angular=0))
        self.assertEqual(self.node.lease.get(), (0, 0))
        self.process_frame()
        post(self.web + '/command', dict(linear=10, angular=-10))
        self.node.drive()
        self.assertEqual((self.published[-1].linear.x, self.published[-1].angular.z), (0.6, -1.5))
        post(self.web + '/command', dict(linear=0, angular=0))
        self.node.drive()
        self.assertEqual(self.published[-1].linear.x, 0)

    def test_stale_camera_stops_rover_even_with_renewed_command(self):
        self.process_frame()
        self.node.received = time.monotonic() - 4
        self.node.lease.set(0.3, 0.8)
        self.node.drive()
        self.assertEqual((self.published[-1].linear.x, self.published[-1].angular.z), (0, 0))
        await_condition(lambda: not self.session.snapshot()['ready'])
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(self.url + '/frame.png')
        self.assertEqual(error.exception.code, 503)
        await_condition(lambda: bool(self.session.snapshot()['error']))
        self.assertIn('Camera stopped updating', self.session.snapshot()['error'])
        self.assertNotIn('HTTPError', self.session.snapshot()['error'])

    def test_observed_1_41_second_gap_still_allows_inference_and_driving(self):
        self.node.received = time.monotonic() - 1.41
        self.process_frame()
        status = self.session.snapshot()
        self.assertTrue(status['camera_delayed'])
        self.assertEqual(status['camera_timeout_seconds'], 3.0)
        post(self.web + '/command', dict(linear=0.3, angular=0))
        self.node.drive()
        self.assertEqual(self.published[-1].linear.x, 0.3)
        time.sleep(0.15)
        self.assertTrue(self.session.snapshot()['ready'])
        self.assertEqual(len(self.model.calls), 1)

    def test_unchanged_frame_returns_304_without_encoding_again(self):
        with patch('rover_perception.live_bridge.cv2.imencode', wraps=cv2.imencode) as encode:
            with urllib.request.urlopen(self.url + '/frame.png') as response:
                etag = response.headers['ETag']
                response.read()
            request = urllib.request.Request(self.url + '/frame.png', headers={'If-None-Match': etag})
            with self.assertRaises(urllib.error.HTTPError) as unchanged:
                urllib.request.urlopen(request)
            self.assertEqual(unchanged.exception.code, 304)
            unchanged.exception.close()
            self.assertEqual(encode.call_count, 1)
            self.node.on_image(image_message())
            with urllib.request.urlopen(request) as response:
                self.assertNotEqual(response.headers['ETag'], etag)
                response.read()
            self.assertEqual(encode.call_count, 2)

    def test_expired_cached_frame_is_rejected_instead_of_returning_304(self):
        with urllib.request.urlopen(self.url + '/frame.png') as response:
            etag = response.headers['ETag']
            response.read()
        self.node.received = time.monotonic() - 4
        request = urllib.request.Request(self.url + '/frame.png', headers={'If-None-Match': etag})
        with self.assertRaises(urllib.error.HTTPError) as expired:
            urllib.request.urlopen(request)
        self.assertEqual(expired.exception.code, 503)
        expired.exception.close()

    def test_stricter_camera_timeout_is_honored(self):
        self.session.camera_timeout = 0.5
        self.node.received = time.monotonic() - 0.8
        self.session.thread.start()
        await_condition(lambda: bool(self.session.snapshot()['error']))
        self.assertFalse(self.session.snapshot()['ready'])

    def test_bridge_timeout_limits_dashboard_even_if_its_setting_is_longer(self):
        self.node.camera_timeout = 0.5
        self.process_frame()
        self.assertEqual(self.session.snapshot()['camera_timeout_seconds'], 0.5)
        self.session.captured = time.monotonic() - 0.7
        self.assertFalse(self.session.snapshot()['ready'])

    def test_camera_recovers_after_prolonged_gap_without_cached_error_sticking(self):
        self.process_frame()
        self.node.received = time.monotonic() - 4
        await_condition(lambda: bool(self.session.snapshot()['error']))
        self.assertFalse(self.session.snapshot()['ready'])
        self.node.on_image(image_message())
        await_condition(lambda: self.session.snapshot()['ready'])
        self.assertIsNone(self.session.snapshot()['error'])

    def test_bridge_restart_with_same_sequence_and_timestamp_still_gets_processed(self):
        self.process_frame()
        self.node.stream_id = 'restarted-stream'
        self.node.received = time.monotonic()
        await_condition(lambda: len(self.model.calls) == 2)

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

    def test_gazebo_mode_rejects_http_commands_without_publishing_idle_stops(self):
        self.node.control_mode = 'gazebo'
        self.process_frame()
        self.assertEqual(self.session.snapshot()['control_mode'], 'gazebo')
        self.assertFalse(self.session.snapshot()['controls_enabled'])
        self.assertEqual(len(self.model.calls), 1)  # predictions remain active
        with urllib.request.urlopen(self.url + '/status') as response:
            self.assertIsNone(json.load(response)['command'])  # camera bridge does not report native driving state
        for url in (self.url, self.web):
            for linear in (0.0, 0.3):
                with self.assertRaises(urllib.error.HTTPError) as error:
                    post(url + '/command', dict(linear=linear, angular=0))
                self.assertEqual(error.exception.code, 403)
                error.exception.close()
        self.node.lease.set(0.3, 0)
        self.node.drive()
        self.assertEqual(self.published, [])
        with patch.object(self.session, 'command') as command:
            self.session.close()
            command.assert_not_called()

    def test_gazebo_mode_is_discovered_even_before_camera_is_ready(self):
        self.node.control_mode = 'gazebo'
        self.node.frame = None
        self.node.received = 0
        self.session.thread.start()
        await_condition(lambda: self.session.snapshot()['control_mode'] == 'gazebo')
        self.assertFalse(self.session.snapshot()['ready'])
        self.assertFalse(self.session.snapshot()['controls_enabled'])


class GazeboDriveTests(unittest.TestCase):
    def setUp(self):
        self.now = [10.0]
        self.node = GazeboDrive.__new__(GazeboDrive)
        self.node.lease = DriveLease(timeout=GAZEBO_COMMAND_TIMEOUT, max_linear=0.3, max_angular=0.8,
                                     clock=lambda: self.now[0])
        self.published = []
        self.node.pub = types.SimpleNamespace(publish=self.published.append)

    def output(self):
        self.node.drive()
        return self.published[-1].linear.x, self.published[-1].angular.z

    def command(self, linear, angular=0):
        msg = twist()
        msg.linear.x, msg.angular.z = linear, angular
        self.node.on_command(msg)

    def test_key_repeat_delay_renewal_release_and_connection_expiry(self):
        self.command(2, -2)
        self.assertEqual(self.output(), (0.3, -0.8))
        self.now[0] += 0.6  # initial desktop auto-repeat delay
        self.assertEqual(self.output(), (0.3, -0.8))
        self.command(0.3, -0.8)  # repeated key event renews the lease
        self.now[0] += 0.7
        self.assertEqual(self.output(), (0.3, -0.8))
        self.command(0, 0)  # release is immediate, not one second later
        self.assertEqual(self.output(), (0, 0))
        self.command(0.3)
        self.now[0] += 1.01  # input disappears, even if sim time is paused
        self.assertEqual(self.output(), (0, 0))

    def test_invalid_native_command_and_shutdown_stop(self):
        self.command(0.3)
        self.command(float('nan'))
        self.assertEqual(self.output(), (0, 0))
        self.command(0.3)
        self.node.close()
        self.assertEqual((self.published[-1].linear.x, self.published[-1].angular.z), (0, 0))

    def test_native_drive_has_no_dependency_on_prediction_or_http_camera(self):
        self.command(0.3)
        self.assertEqual(self.output(), (0.3, 0))
        self.assertFalse(hasattr(self.node, 'frame'))
        self.assertFalse(hasattr(self.node, 'http'))

    def test_gazebo_mode_constructs_camera_bridge_without_drive_publisher_or_timer(self):
        with patch.object(FakeRosNode, 'overrides', {'control_mode': 'gazebo', 'port': 0}):
            node = LiveBridge()
        try:
            self.assertIsNone(node.pub)
            self.assertEqual(node.publishers, [])
            self.assertEqual(node.timers, [])
            self.assertEqual(node.subscriptions, ['/oak/image_raw'])
        finally:
            node.close()

    def test_native_adapter_subscribes_to_the_dedicated_topic_and_owns_cmd_vel(self):
        node = GazeboDrive()
        self.assertEqual(node.subscriptions, [GAZEBO_DRIVE_TOPIC])
        self.assertEqual(node.publishers, ['/cmd_vel'])
        self.assertEqual(node.timers[0][0], 0.05)
        for mode in ('browser', 'gazebo'):
            self.assertEqual(control_mode(mode), mode)
        with self.assertRaises(ValueError):
            control_mode('both')


if __name__ == '__main__':
    unittest.main()
