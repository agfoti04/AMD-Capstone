"""Host-side live inference and browser driving view (no ROS Python dependency)."""
import argparse
import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

from .live_core import DriveLease, detection_rows


class LiveSession:
    def __init__(self, model, bridge, imgsz=960, conf=0.25, device='0', fps=8, record=None):
        self.model, self.bridge = model, bridge.rstrip('/')
        self.imgsz, self.conf, self.device, self.fps = imgsz, conf, device, fps
        self.record = record
        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.jpeg = None
        self.version = 0
        self.captured = 0.0
        self.status = {'error': 'Waiting for the simulator camera', 'detections': []}
        self.thread = threading.Thread(target=self.run, daemon=True)

    def snapshot(self):
        with self.condition:
            age = time.monotonic() - self.captured if self.captured else None
            return dict(self.status, frame_age_seconds=age, ready=age is not None and age < 1.5)

    def command(self, linear, angular):
        validator = DriveLease()
        validator.set(linear, angular)
        linear, angular = validator.get()
        if (linear or angular) and not self.snapshot()['ready']:
            linear, angular = 0.0, 0.0
        payload = json.dumps({'linear': linear, 'angular': angular}).encode()
        req = urllib.request.Request(self.bridge + '/command', payload,
                                     {'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(req, timeout=0.4) as response:
            response.read()

    def run(self):
        previous = None
        last_processed = None
        while not self.stop.is_set():
            started = time.monotonic()
            try:
                with urllib.request.urlopen(self.bridge + '/frame.png', timeout=2) as response:
                    sequence = int(response.headers['X-Frame-Sequence'])
                    sim_time = float(response.headers['X-Sim-Time'])
                    age = float(response.headers['X-Frame-Age'])
                    raw = response.read()
                # Count request/transfer time too; never present an old frame as fresh.
                captured = started - age
                if age > 1.0:
                    raise ValueError('Simulator camera is stale')
                if (sequence, sim_time) == previous:
                    self.stop.wait(0.05)
                    continue
                previous = (sequence, sim_time)
                image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError('Cannot decode simulator PNG')
                infer_start = time.monotonic()
                result = self.model.predict(image, imgsz=self.imgsz, conf=self.conf,
                                            device=self.device, rect=False, verbose=False)[0]
                inference_ms = (time.monotonic() - infer_start) * 1000
                annotated = result.plot()
                ok, encoded = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if not ok:
                    raise ValueError('Cannot encode annotated image')
                rows = detection_rows(result)
                completed = time.monotonic()
                status = {'error': None, 'sequence': sequence, 'sim_time': sim_time,
                          'inference_ms': inference_ms,
                          'loop_fps': 0.0 if last_processed is None else 1 / max(completed - last_processed, 1e-6),
                          'detections': rows}
                last_processed = completed
                if self.record:
                    stem = f'{sequence:09d}_{sim_time:.6f}'
                    (self.record / (stem + '.png')).write_bytes(raw)
                    (self.record / (stem + '_boxes.jpg')).write_bytes(encoded.tobytes())
                    with (self.record / 'predictions.jsonl').open('a', encoding='utf-8') as output:
                        output.write(json.dumps(dict(status, image=stem + '.png', wall_time=time.time())) + '\n')
                with self.condition:
                    self.jpeg = encoded.tobytes()
                    self.captured = captured
                    self.version += 1
                    self.status = status
                    self.condition.notify_all()
            except Exception as error:
                with self.condition:
                    self.status = dict(self.status, error=f'{type(error).__name__}: {error}')
                    self.captured = 0.0
                self.stop.wait(0.25)
            self.stop.wait(max(0, 1 / self.fps - (time.monotonic() - started)))

    def close(self):
        self.stop.set()
        try:
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
    parser.add_argument('--fps', type=float, default=8)
    parser.add_argument('--record-dir', type=Path)
    parser.add_argument('--scene', default='', help='Scene ID for recording provenance')
    args = parser.parse_args(argv)
    if not args.model.is_file():
        parser.error(f'Checkpoint does not exist: {args.model}')
    if args.imgsz <= 0 or not 0 < args.conf <= 1 or not 0 < args.fps <= 30:
        parser.error('Require imgsz > 0, 0 < conf <= 1 and 0 < fps <= 30')
    bridge = urlparse(args.bridge)
    if bridge.scheme != 'http' or bridge.hostname not in ('127.0.0.1', 'localhost'):
        parser.error('The simulator bridge must be on localhost')
    from ultralytics import YOLO
    model = YOLO(str(args.model.resolve()))
    if model.names != {0: 'rock', 1: 'pale_soil_patch', 2: 'yellow_crystal_rock'}:
        parser.error(f'Expected the trained three-class rover model; received {model.names}')
    if args.record_dir:
        args.record_dir.mkdir(parents=True, exist_ok=False)
        manifest = dict(weights=str(args.model.resolve()),
                        weights_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                        scene=args.scene, imgsz=args.imgsz, conf=args.conf, device=args.device,
                        class_names=model.names, started_wall_time=time.time())
        (args.record_dir / 'session.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    page = Path(__file__).with_name('live_dashboard.html').read_bytes()
    session = LiveSession(model, args.bridge, args.imgsz, args.conf, args.device, args.fps, args.record_dir)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), dashboard_handler(session, page))
    server.daemon_threads = True
    session.thread.start()
    print(f'Live rover view: http://127.0.0.1:{args.port} | weights: {args.model.resolve()}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        session.close()
        server.server_close()


if __name__ == '__main__':
    main()
