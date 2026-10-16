"""Exercise backend-independent prediction delivery through the camera bridge."""
import contextlib
import io
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

from rover_perception.detectors import DetectionResult
from rover_perception.live_core import DriveLease
from rover_perception.live_dashboard import LiveSession, main
from test_live import LiveBridge, bridge_handler, image_message, run_server, await_condition


class ControlledDetector:
    names = {0: 'rock', 1: 'pale_soil_patch', 2: 'yellow_crystal_rock', 3: 'mudstone_slab'}
    metadata = {'backend': 'npu', 'precision': 'bf16', 'model': 'controlled-test-model'}

    def __init__(self):
        self.fail = False

    def detect(self, image, conf=0.25):
        if self.fail:
            raise RuntimeError('Controlled NPU execution failure')
        return DetectionResult([{'class_id': 3, 'class_name': 'mudstone_slab',
                                 'confidence': 0.9, 'xyxy': [0, 0, 1, 1]}],
                               {'preprocess_ms': 1, 'execution_ms': 2, 'postprocess_ms': 3})


class BackendDeliveryTests(unittest.TestCase):
    def test_npu_startup_refuses_provider_only_proof_without_importing_ultralytics(self):
        detector = ControlledDetector()
        detector.metadata = dict(detector.metadata, proof_state='npu_executed')
        detector.warmup = Mock()
        with tempfile.TemporaryDirectory() as folder:
            manifest = Path(folder) / 'manifest.json'
            manifest.write_text('{}', encoding='utf-8')
            errors = io.StringIO()
            with patch('rover_perception.live_dashboard.create_detector', return_value=detector), \
                    patch.dict(sys.modules, {'ultralytics': None}), \
                    contextlib.redirect_stderr(errors), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as failure:
                    main(['--backend', 'npu', '--model-manifest', str(manifest)])
            self.assertEqual(failure.exception.code, 2)
            self.assertIn('not verified', errors.getvalue())
            detector.warmup.assert_called_once()

    def test_backend_metadata_timings_and_failure_reach_drive_readiness(self):
        node = LiveBridge.__new__(LiveBridge)
        node.lock = threading.Lock()
        node.frame, node.sequence, node.received = image_message(), 1, time.monotonic()
        node.lease, node.camera_timeout = DriveLease(), 3.0
        node.stream_id, node.control_mode = 'backend-test', 'browser'
        bridge, url = run_server(bridge_handler(node))
        detector = ControlledDetector()
        session = LiveSession(detector, url)
        try:
            session.thread.start()
            await_condition(lambda: session.snapshot()['ready'])
            status = session.snapshot()
            self.assertEqual(status['detector']['backend'], 'npu')
            self.assertEqual(status['detector_timings']['execution_ms'], 2)
            self.assertGreater(status['detector_capacity_fps'], 0)
            self.assertGreaterEqual(status['annotation_ms'], 0)
            self.assertGreaterEqual(status['encoding_ms'], 0)
            self.assertEqual(status['detections'][0]['class_id'], 3)
            self.assertEqual(status['processed_frames'], 1)
            self.assertTrue(status['processing_session_id'])
            time.sleep(.1)
            self.assertEqual(session.snapshot()['processed_frames'], 1)
            self.assertGreater(session.drive_values(0.3, 0)['linear'], 0)
            detector.fail = True
            node.sequence += 1
            node.received = time.monotonic()
            await_condition(lambda: bool(session.snapshot()['error']))
            self.assertIn('Controlled NPU execution failure', session.snapshot()['error'])
            self.assertEqual(session.drive_values(0.3, 0)['linear'], 0)
            self.assertEqual(session.snapshot()['processed_frames'], 1)
        finally:
            session.close()
            bridge.shutdown()
            bridge.server_close()


if __name__ == '__main__':
    unittest.main()
