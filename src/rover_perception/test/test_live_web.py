"""Exercise real WebSockets, the local HTTP bridge and frame backpressure."""
import asyncio
import json
import threading
import time
import types
import unittest

import cv2
from aiohttp import ClientSession, WSMsgType, web

from rover_perception.live_core import DriveLease, image_to_bgr
from rover_perception.live_dashboard import LiveSession
from rover_perception.live_web import create_app
from test_live import FakeModel, LiveBridge, bridge_handler, image_message, run_server


class WebTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.node = LiveBridge.__new__(LiveBridge)
        self.node.lock = threading.Lock()
        self.node.frame, self.node.sequence, self.node.received = image_message(), 1, time.monotonic()
        self.node.lease, self.node.camera_timeout = DriveLease(), 3.0
        self.node.stream_id, self.node.control_mode = 'web-test', 'browser'
        self.bridge, url = run_server(bridge_handler(self.node))
        self.session = LiveSession(FakeModel(), url)
        self.session.control_mode = 'browser'
        self.session.captured = time.monotonic()
        self.session.status = {'error': None, 'detections': []}
        self.session.jpeg = cv2.imencode('.jpg', image_to_bgr(self.node.frame))[1].tobytes()
        self.session.version = 1
        self.runner = web.AppRunner(create_app(self.session, b'<html>Rover live vision</html>'))
        await self.runner.setup()
        site = web.TCPSite(self.runner, '127.0.0.1', 0)
        await site.start()
        self.url = 'http://127.0.0.1:' + str(site._server.sockets[0].getsockname()[1])
        self.client = ClientSession()

    async def asyncTearDown(self):
        await self.client.close()
        await self.runner.cleanup()
        self.session.close()
        await asyncio.to_thread(self.bridge.shutdown)
        self.bridge.server_close()

    async def drive(self, ws, seq, linear, angular=0):
        await ws.send_json(dict(seq=seq, linear=linear, angular=angular))
        reply = await ws.receive_json(timeout=1)
        self.assertEqual(reply, {'seq': seq, 'ok': True})

    async def test_held_controls_speed_clamping_release_and_single_driver(self):
        async with self.client.ws_connect(self.url + '/control') as ws:
            for seq in range(8):
                await self.drive(ws, seq, 0.6, 1.5)
                self.assertEqual(self.node.lease.get(), (0.6, 1.5))
                await asyncio.sleep(0.05)
            await self.drive(ws, 8, 5, -5)
            self.assertEqual(self.node.lease.get(), (0.6, -1.5))
            await self.drive(ws, 9, 0)
            self.assertEqual(self.node.lease.get(), (0, 0))
            async with self.client.get(self.url + '/control') as response:
                self.assertEqual(response.status, 409)

    async def test_release_can_be_sent_before_move_ack_without_reordering(self):
        async with self.client.ws_connect(self.url + '/control') as ws:
            await ws.send_json({'seq': 1, 'linear': 0.6, 'angular': 0})
            await ws.send_json({'seq': 2, 'linear': 0, 'angular': 0})
            self.assertEqual((await ws.receive_json(timeout=1))['seq'], 1)
            self.assertEqual((await ws.receive_json(timeout=1))['seq'], 2)
            self.assertEqual(self.node.lease.get(), (0, 0))

    async def test_missing_heartbeat_expires_and_closes_connection(self):
        async with self.client.ws_connect(self.url + '/control') as ws:
            await self.drive(ws, 1, 0.6)
            await asyncio.sleep(0.38)
            self.assertEqual(self.node.lease.get(), (0, 0))
            message = await ws.receive(timeout=1)
            self.assertIn(message.type, (WSMsgType.CLOSE, WSMsgType.CLOSED))

    async def test_disconnect_stops_and_new_connection_can_drive(self):
        ws = await self.client.ws_connect(self.url + '/control')
        await self.drive(ws, 1, 0.6)
        await ws.close()
        await asyncio.sleep(0.05)
        self.assertEqual(self.node.lease.get(), (0, 0))
        async with self.client.ws_connect(self.url + '/control') as new:
            await self.drive(new, 1, 0.2)
            self.assertEqual(self.node.lease.get(), (0.2, 0))

    async def test_stale_or_failed_inference_sends_zero(self):
        async with self.client.ws_connect(self.url + '/control') as ws:
            await self.drive(ws, 1, 0.6)
            self.session.captured = time.monotonic() - 4
            await self.drive(ws, 2, 0.6)
            self.assertEqual(self.node.lease.get(), (0, 0))
            self.session.captured = time.monotonic()
            self.session.status['error'] = 'Inference failed'
            await self.drive(ws, 3, 0.6)
            self.assertEqual(self.node.lease.get(), (0, 0))

    async def test_bad_sequence_stops_and_foreign_origin_is_rejected(self):
        async with self.client.get(self.url + '/control', headers={'Origin': 'http://example.com'}) as response:
            self.assertEqual(response.status, 403)
        async with self.client.ws_connect(self.url + '/control') as ws:
            await self.drive(ws, 1, 0.6)
            await ws.send_json({'seq': 1, 'linear': 0.6, 'angular': 0})
            await ws.receive(timeout=1)
            self.assertEqual(self.node.lease.get(), (0, 0))

    async def test_slow_viewer_receives_newest_frame_and_does_not_block_controls(self):
        async with self.client.ws_connect(self.url + '/view') as view:
            first = await view.receive_json(timeout=1)
            self.assertEqual(first['type'], 'frame')
            self.assertEqual(first['video_id'], 1)
            self.assertEqual(await view.receive_bytes(timeout=1), self.session.jpeg)
            # Multiple frames cross the network before any acknowledgment,
            # removing the single-round-trip frame-rate limit.
            for version in (2, 3):
                self.session.version = version
                self.session.jpeg = b'jpeg' + str(version).encode()
                self.assertEqual((await view.receive_json(timeout=1))['video_id'], version)
                self.assertEqual(await view.receive_bytes(timeout=1), self.session.jpeg)
            # The three-frame window is full. Further frames stay in the single
            # latest slot rather than accumulating in the socket.
            for version in range(4, 12):
                self.session.version = version
                self.session.jpeg = b'jpeg' + str(version).encode()
            with self.assertRaises(asyncio.TimeoutError):
                await view.receive(timeout=0.05)
            async with self.client.ws_connect(self.url + '/control') as control:
                await self.drive(control, 1, 0.6)
                self.assertEqual(self.node.lease.get(), (0.6, 0))
                await view.send_json({'ack': 3})
                self.assertEqual((await view.receive_json(timeout=1))['video_id'], 11)
                self.assertEqual(await view.receive_bytes(timeout=1), b'jpeg11')

    async def test_invalid_video_acknowledgment_closes_stream(self):
        async with self.client.ws_connect(self.url + '/view') as view:
            await view.receive_json(timeout=1)
            await view.receive_bytes(timeout=1)
            await view.send_json({'ack': 999})
            self.assertIn((await view.receive(timeout=1)).type, (WSMsgType.CLOSE, WSMsgType.CLOSED))


if __name__ == '__main__':
    unittest.main()
