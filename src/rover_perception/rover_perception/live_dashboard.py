"""Host-side live inference and browser driving view (no ROS Python dependency)."""
import argparse
import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from http.client import HTTPException
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

from .live_core import (DEFAULT_CAMERA_TIMEOUT, MAX_LINEAR_SPEED, MAX_ANGULAR_SPEED,
                        DriveLease, camera_timeout, detection_rows)
from .drive_mode import control_mode


class LiveSession:
    def __init__(self, model, bridge, imgsz=960, conf=0.25, device='0', fps=30, record=None,
                 camera_timeout_sec=DEFAULT_CAMERA_TIMEOUT, rect=True):
        self.model, self.bridge = model, bridge.rstrip('/')
        self.imgsz, self.conf, self.device, self.fps = imgsz, conf, device, fps
        self.record = record
        self.rect = rect
        self.camera_timeout = camera_timeout(camera_timeout_sec)
        self.bridge_camera_timeout = self.camera_timeout
        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.jpeg = None
        self.version = 0
        self.captured = 0.0
        self.control_mode = 'unknown'
        self.status = {'error': 'Waiting for the simulator camera', 'detections': []}
        self.thread = threading.Thread(target=self.run, daemon=True)

    def snapshot(self):
        with self.condition:
            age = time.monotonic() - self.captured if self.captured else None
            limit = min(self.camera_timeout, self.bridge_camera_timeout)
            return dict(self.status, frame_age_seconds=age, camera_timeout_seconds=limit,
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
                request = urllib.request.Request(self.bridge + '/frame.png',
                                                 headers={'If-None-Match': etag} if etag else {})
                with urllib.request.urlopen(request, timeout=2) as response:
                    with self.condition:
                        self.control_mode = control_mode(response.headers.get('X-Control-Mode', 'browser'))
                    headers_received = time.monotonic()
                    sequence = int(response.headers['X-Frame-Sequence'])
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
                image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError('Cannot decode simulator PNG')
                infer_start = time.monotonic()
                result = self.model.predict(image, imgsz=self.imgsz, conf=self.conf,
                                            device=self.device, rect=self.rect, verbose=False)[0]
                inference_ms = (time.monotonic() - infer_start) * 1000
                annotated = result.plot()
                ok, encoded = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok:
                    raise ValueError('Cannot encode annotated image')
                rows = detection_rows(result)
                completed = time.monotonic()
                status = {'error': None, 'sequence': sequence, 'sim_time': sim_time,
                          'inference_ms': inference_ms,
                          'loop_fps': 0.0 if last_processed is None else 1 / max(completed - last_processed, 1e-6),
                          'sim_speed': None if last_processed is None else
                              max(0, sim_time - last_sim_time) / max(completed - last_processed, 1e-6),
                          'image_size': [image.shape[1], image.shape[0]],
                          'detections': rows}
                last_processed = completed
                last_sim_time = sim_time
                if self.record:
                    stem = f'{sequence:09d}_{sim_time:.6f}'
                    (self.record / (stem + '.png')).write_bytes(raw)
                    (self.record / (stem + '_boxes.jpg')).write_bytes(encoded.tobytes())
                    with (self.record / 'predictions.jsonl').open('a', encoding='utf-8') as output:
                        output.write(json.dumps(dict(status, image=stem + '.png', wall_time=time.time())) + '\n')
                with self.condition:
                    self.jpeg = encoded.tobytes()
                    self.captured = captured
                    self.bridge_camera_timeout = bridge_limit
                    self.version += 1
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
    parser.add_argument('--model', type=Path, required=True)
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
    if not args.model.is_file():
        parser.error(f'Checkpoint does not exist: {args.model}')
    if args.imgsz <= 0 or not 0 < args.conf <= 1 or not 0 < args.fps <= 30:
        parser.error('Require imgsz > 0, 0 < conf <= 1 and 0 < fps <= 30')
    try:
        camera_timeout(args.camera_timeout)
    except ValueError as error:
        parser.error(str(error))
    bridge = urlparse(args.bridge)
    if bridge.scheme != 'http' or bridge.hostname not in ('127.0.0.1', 'localhost'):
        parser.error('The simulator bridge must be on localhost')
    from ultralytics import YOLO
    from .live_web import serve
    cv2.setNumThreads(1)
    model = YOLO(str(args.model.resolve()))
    if model.names != {0: 'rock', 1: 'pale_soil_patch', 2: 'yellow_crystal_rock'}:
        parser.error(f'Expected the trained three-class rover model; received {model.names}')
    if args.record_dir:
        args.record_dir.mkdir(parents=True, exist_ok=False)
        manifest = dict(weights=str(args.model.resolve()),
                        weights_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                        scene=args.scene, imgsz=args.imgsz, conf=args.conf, device=args.device,
                        rect=not args.square, target_fps=args.fps,
                        camera_timeout_seconds=args.camera_timeout,
                        class_names=model.names, started_wall_time=time.time())
        (args.record_dir / 'session.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    page = Path(__file__).with_name('live_dashboard.html').read_bytes()
    session = LiveSession(model, args.bridge, args.imgsz, args.conf, args.device, args.fps, args.record_dir,
                          args.camera_timeout, rect=not args.square)
    session.thread.start()
    print(f'Live rover view: http://127.0.0.1:{args.port} | weights: {args.model.resolve()}', flush=True)
    try:
        serve(session, page, args.port)
    except KeyboardInterrupt:
        pass
    finally:
        session.close()


if __name__ == '__main__':
    main()
