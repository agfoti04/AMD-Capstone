"""Host-side live inference and browser driving view (no ROS Python dependency)."""
import argparse
import hashlib
import json
import threading
import time
import uuid
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from http.client import HTTPException
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

from .live_core import (DEFAULT_CAMERA_TIMEOUT, LIVE_PIPELINE_VERSION, MAX_LINEAR_SPEED, MAX_ANGULAR_SPEED,
                        DriveLease, camera_timeout, rover_model_names)
from .drive_mode import control_mode
from .detectors import TorchDetector, create_detector, draw_detections


class LiveSession:
    def __init__(self, model, bridge, imgsz=960, conf=0.25, device='0', fps=30, record=None,
                 camera_timeout_sec=DEFAULT_CAMERA_TIMEOUT, rect=True):
        self.model = model if callable(getattr(model, 'detect', None)) else TorchDetector(
            model=model, device=device, imgsz=imgsz, rect=rect)
        self.bridge = bridge.rstrip('/')
        self.imgsz, self.conf, self.device, self.fps = imgsz, conf, device, fps
        self.record = record
        self.rect = rect
        self.camera_timeout = camera_timeout(camera_timeout_sec)
        self.bridge_camera_timeout = self.camera_timeout
        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.jpeg = None
        self.version = 0
        self.processed_frames = 0
        self.processing_session_id = str(uuid.uuid4())
        self.captured = 0.0
        self.control_mode = 'unknown'
        self.status = {'error': 'Waiting for the simulator camera', 'detections': []}
        self.thread = threading.Thread(target=self.run, daemon=True)

    def snapshot(self):
        with self.condition:
            age = time.monotonic() - self.captured if self.captured else None
            limit = min(self.camera_timeout, self.bridge_camera_timeout)
            return dict(self.status, frame_age_seconds=age, camera_timeout_seconds=limit,
                        processed_frames=self.processed_frames,
                        processing_session_id=self.processing_session_id,
                        class_names=getattr(self.model, 'names', {}),
                        detector=dict(getattr(self.model, 'metadata', {})),
                        pipeline_version=LIVE_PIPELINE_VERSION,
                        control_mode=self.control_mode, controls_enabled=self.control_mode == 'browser',
                        max_linear_speed=MAX_LINEAR_SPEED, max_angular_speed=MAX_ANGULAR_SPEED,
                        camera_delayed=age is not None and 1.0 < age < limit,
                        ready=age is not None and age < limit and not self.status.get('error'))

    def drive_values(self, linear, angular):
        if self.control_mode == 'gazebo':
            raise PermissionError('Driving is controlled in Gazebo; this dashboard is a camera viewer.')
        validator = DriveLease()
        validator.set(linear, angular)
        linear, angular = validator.get()
        if (linear or angular) and not self.snapshot()['ready']:
            linear, angular = 0.0, 0.0
        return {'linear': linear, 'angular': angular}

    def command(self, linear, angular):
        payload = json.dumps(self.drive_values(linear, angular)).encode()
        req = urllib.request.Request(self.bridge + '/command', payload,
                                     {'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(req, timeout=0.4) as response:
            response.read()

    def run(self):
        previous = None
        etag = None
        last_processed = None
        last_sim_time = None
        while not self.stop.is_set():
            started = time.monotonic()
            try:
                request = urllib.request.Request(self.bridge + '/frame.bgr',
                                                 headers={'If-None-Match': etag} if etag else {})
                with urllib.request.urlopen(request, timeout=2) as response:
                    with self.condition:
                        self.control_mode = control_mode(response.headers.get('X-Control-Mode', 'browser'))
                    headers_received = time.monotonic()
                    sequence = int(response.headers['X-Frame-Sequence'])
                    width, height = int(response.headers['X-Image-Width']), int(response.headers['X-Image-Height'])
                    sim_time = float(response.headers['X-Sim-Time'])
                    age = float(response.headers['X-Frame-Age'])
                    bridge_limit = camera_timeout(response.headers.get('X-Camera-Timeout', self.camera_timeout))
                    received_etag = response.headers.get('ETag')
                    raw = response.read()
                # Age was measured just before response headers, so count body
                # transfer and inference once rather than counting encoding twice.
                captured = headers_received - age
                if age >= min(self.camera_timeout, bridge_limit):
                    raise ValueError('Camera stopped updating; driving is paused.')
                if (received_etag, sequence, sim_time) == previous:
                    self.stop.wait(0.05)
                    continue
                if width <= 0 or height <= 0 or len(raw) != width * height * 3:
                    raise ValueError('Invalid raw camera frame dimensions')
                image = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy()
                infer_start = time.monotonic()
                result = self.model.detect(image, conf=self.conf)
                inference_ms = (time.monotonic() - infer_start) * 1000
                rows = result.detections
                annotated = draw_detections(image, rows)
                encode_started = time.monotonic()
                ok, encoded = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok:
                    raise ValueError('Cannot encode annotated image')
                completed = time.monotonic()
                status = {'error': None, 'sequence': sequence, 'sim_time': sim_time,
                          'inference_ms': inference_ms,
                          'detector_timings': result.timings,
                          'detector_capacity_fps': 1000 / max(inference_ms, 1e-6),
                          'loop_fps': 0.0 if last_processed is None else 1 / max(completed - last_processed, 1e-6),
                          'sim_speed': None if last_processed is None else
                              max(0, sim_time - last_sim_time) / max(completed - last_processed, 1e-6),
                          'image_size': [image.shape[1], image.shape[0]],
                          'camera_fetch_ms': (infer_start - started) * 1000,
                          'annotation_ms': (encode_started - infer_start) * 1000 - inference_ms,
                          'encoding_ms': (completed - encode_started) * 1000,
                          'annotation_encode_ms': (completed - infer_start) * 1000 - inference_ms,
                          'detections': rows}
                last_processed = completed
                last_sim_time = sim_time
                if self.record:
                    stem = f'{sequence:09d}_{sim_time:.6f}'
                    if not cv2.imwrite(str(self.record / (stem + '.png')), image):
                        raise ValueError('Cannot record raw camera PNG')
                    (self.record / (stem + '_boxes.jpg')).write_bytes(encoded.tobytes())
                    with (self.record / 'predictions.jsonl').open('a', encoding='utf-8') as output:
                        output.write(json.dumps(dict(status, image=stem + '.png', wall_time=time.time())) + '\n')
                with self.condition:
                    self.jpeg = encoded.tobytes()
                    self.captured = captured
                    self.bridge_camera_timeout = bridge_limit
                    self.version += 1
                    self.processed_frames += 1
                    self.status = status
                    self.condition.notify_all()
                # Cache only successfully processed samples, so failed inference
                # can retry the same frame and a restarted bridge has a new ETag.
                previous = (received_etag, sequence, sim_time)
                etag = received_etag
            except urllib.error.HTTPError as error:
                mode = error.headers.get('X-Control-Mode')
                if mode in ('browser', 'gazebo'):
                    with self.condition:
                        self.control_mode = mode
                if error.code == 304:
                    # A normal wait for the next camera sample; preserve the
                    # last processed frame's age and the current held controls.
                    error.close()
                else:
                    try:
                        message = json.loads(error.read(4096)).get('error')
                    except (ValueError, AttributeError, OSError, HTTPException):
                        message = None
                    finally:
                        error.close()
                    with self.condition:
                        if error.code == 404:
                            message = 'Installed camera bridge is outdated. Extract the update with tar --touch, rebuild and restart the simulator.'
                        self.status = dict(self.status, error=message or f'Camera bridge returned HTTP {error.code}')
                    self.stop.wait(0.25)
            except Exception as error:
                with self.condition:
                    self.status = dict(self.status, error=f'{type(error).__name__}: {error}')
                self.stop.wait(0.25)
            self.stop.wait(max(0, 1 / self.fps - (time.monotonic() - started)))

    def close(self):
        self.stop.set()
        try:
            if self.control_mode == 'browser':
                self.command(0, 0)
        except (OSError, ValueError):
            pass
        with self.condition:
            self.condition.notify_all()
        if self.thread.is_alive():
            self.thread.join(timeout=3)


def dashboard_handler(session, page):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, code, value, kind='application/json'):
            body = value if isinstance(value, bytes) else json.dumps(value).encode()
            self.send_response(code)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == '/':
                self.reply(200, page, 'text/html; charset=utf-8')
            elif self.path == '/status':
                self.reply(200, session.snapshot())
            elif self.path == '/stream.mjpg':
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                version = -1
                try:
                    while not session.stop.is_set():
                        with session.condition:
                            session.condition.wait_for(lambda: session.version != version or session.stop.is_set(), 1)
                            jpeg, version = session.jpeg, session.version
                        if jpeg is None:
                            continue
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' +
                                         str(len(jpeg)).encode() + b'\r\n\r\n' + jpeg + b'\r\n')
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.reply(404, {'error': 'Not found'})

        def do_POST(self):
            try:
                if session.snapshot()['control_mode'] == 'gazebo':
                    raise PermissionError('Driving is controlled in Gazebo; this dashboard is a camera viewer.')
                origin = self.headers.get('Origin')
                if origin and urlparse(origin).netloc != self.headers.get('Host'):
                    raise ValueError('Use this dashboard to send controls')
                if self.path != '/command' or self.headers.get('Content-Type') != 'application/json':
                    raise ValueError('Expected JSON /command')
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 1024:
                    raise ValueError('Invalid command length')
                command = json.loads(self.rfile.read(size))
                session.command(command['linear'], command['angular'])
                self.reply(200, {'ok': True})
            except PermissionError as error:
                self.reply(403, {'error': str(error)})
            except (ValueError, KeyError, TypeError) as error:
                try:
                    session.command(0, 0)
                except OSError:
                    pass
                self.reply(400, {'error': str(error)})
            except OSError as error:
                self.reply(503, {'error': f'Simulator bridge unavailable: {error}'})
    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, help='PyTorch checkpoint for the torch backend')
    parser.add_argument('--backend', choices=('torch', 'npu'), default='torch')
    parser.add_argument('--model-manifest', type=Path, help='Prepared ONNX manifest for the npu backend')
    parser.add_argument('--bridge', default='http://127.0.0.1:8770')
    parser.add_argument('--port', type=int, default=8771)
    parser.add_argument('--device', default='0')
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--fps', type=float, default=30)
    parser.add_argument('--square', action='store_true', help='Use full square padding instead of faster rectangular inference')
    parser.add_argument('--camera-timeout', type=float, default=DEFAULT_CAMERA_TIMEOUT,
                        help='Stop driving when camera data is older than this many wall seconds (0.1–10)')
    parser.add_argument('--record-dir', type=Path)
    parser.add_argument('--scene', default='', help='Scene ID for recording provenance')
    args = parser.parse_args(argv)
    if args.backend == 'torch' and (args.model is None or not args.model.is_file()):
        parser.error(f'Checkpoint does not exist: {args.model}')
    if args.backend == 'npu' and (args.model_manifest is None or not args.model_manifest.is_file()):
        parser.error('The npu backend requires --model-manifest pointing to a prepared model manifest')
    if args.backend == 'npu' and args.model is not None:
        parser.error('Use --model-manifest for npu, or --model for torch')
    if args.imgsz <= 0 or not 0 < args.conf <= 1 or not 0 < args.fps <= 30:
        parser.error('Require imgsz > 0, 0 < conf <= 1 and 0 < fps <= 30')
    try:
        camera_timeout(args.camera_timeout)
    except ValueError as error:
        parser.error(str(error))
    bridge = urlparse(args.bridge)
    if bridge.scheme != 'http' or bridge.hostname not in ('127.0.0.1', 'localhost'):
        parser.error('The simulator bridge must be on localhost')
    from .live_web import serve
    cv2.setNumThreads(1)
    try:
        model = create_detector(args.backend, model=args.model, manifest=args.model_manifest,
                                device=args.device, imgsz=args.imgsz, rect=not args.square)
        rover_model_names(model.names)
        print(f'Initializing {args.backend} detector; driving stays disabled until a fresh prediction.', flush=True)
        model.warmup()
        if args.backend == 'npu':
            from .npu_tools import detector_metadata
            evidence = detector_metadata(model)
            if model.metadata.get('proof_state') != 'npu_executed' or not evidence.get('compile_evidence', {}).get('verified'):
                raise RuntimeError('NPU deployment is not verified for this model/runtime. Run npu_prepare compile and inspect compile_report.json first.')
            model.metadata.update(evidence)
    except (ValueError, RuntimeError, ImportError, OSError) as error:
        parser.error(str(error))
    effective_imgsz = model.metadata['input_size'][0] if args.backend == 'npu' else args.imgsz
    effective_rect = args.backend == 'torch' and not args.square
    if args.record_dir:
        args.record_dir.mkdir(parents=True, exist_ok=False)
        artifact = args.model if args.backend == 'torch' else args.model_manifest
        manifest = dict(weights=str(artifact.resolve()),
                        weights_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
                        backend=args.backend, detector=model.metadata,
                        scene=args.scene, imgsz=effective_imgsz, conf=args.conf, device=args.device,
                        rect=effective_rect, target_fps=args.fps,
                        camera_timeout_seconds=args.camera_timeout,
                        class_names=model.names, started_wall_time=time.time())
        (args.record_dir / 'session.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    page = Path(__file__).with_name('live_dashboard.html').read_bytes()
    session = LiveSession(model, args.bridge, effective_imgsz, args.conf, args.device, args.fps, args.record_dir,
                          args.camera_timeout, rect=effective_rect)
    session.thread.start()
    print(f'Live rover view: http://127.0.0.1:{args.port} | backend: {args.backend} '
          f'({model.metadata.get("precision", "unknown")}) | model: {model.metadata.get("model")}', flush=True)
    try:
        serve(session, page, args.port)
    except KeyboardInterrupt:
        pass
    finally:
        session.close()


if __name__ == '__main__':
    main()
