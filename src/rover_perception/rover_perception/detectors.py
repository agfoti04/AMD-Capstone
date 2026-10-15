"""Detector adapters shared by the live viewer and reproducible NPU experiments.

ONNX Runtime and Ultralytics are deliberately imported only by their adapters.
The camera, controls and drawing code need neither framework installed.
"""
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import threading
import time

import cv2
import numpy as np

from .live_core import detection_rows, rover_model_names


VITIS_PROVIDER = 'VitisAIExecutionProvider'
_REPORT_ENV_LOCK = threading.Lock()
_PREPROCESSING = {'color': 'rgb', 'scale': 1 / 255, 'layout': 'NCHW', 'padding': 114}


@dataclass
class DetectionResult:
    detections: list
    timings: dict


@dataclass(frozen=True)
class LetterboxTransform:
    original_size: tuple  # height, width
    input_size: tuple
    scale: float
    left: int
    top: int


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _image(image):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Detector input must be an HWC uint8 BGR image')
    if min(image.shape[:2]) <= 0:
        raise ValueError('Detector image cannot be empty')


def _confidence(conf):
    if isinstance(conf, bool) or not math.isfinite(float(conf)) or not 0 <= float(conf) <= 1:
        raise ValueError('Confidence must be between 0 and 1')
    return float(conf)


def preprocess_image(image, input_size):
    """Ultralytics-compatible fixed-size letterboxing, BGR -> RGB float NCHW."""
    _image(image)
    height, width = map(int, input_size)
    if min(height, width) <= 0:
        raise ValueError('Input dimensions must be positive')
    source_h, source_w = image.shape[:2]
    scale = min(height / source_h, width / source_w)
    resized_w, resized_h = round(source_w * scale), round(source_h * scale)
    left = round((width - resized_w) / 2 - 0.1)
    top = round((height - resized_h) / 2 - 0.1)
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    canvas = cv2.copyMakeBorder(resized, top, height - resized_h - top,
                                left, width - resized_w - left,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    tensor = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)
    tensor *= np.float32(1 / 255)
    return tensor, LetterboxTransform((source_h, source_w), (height, width), scale, left, top)


def infer_output_format(shape, nclass):
    """Recognize only batch-one detection outputs; never guess a segmentation head."""
    if len(shape) != 3 or shape[0] != 1:
        raise ValueError(f'Expected a static batch-one detection output, received {shape}')
    if shape[-1] == 6:
        return 'yolo_end2end'
    channels = 4 + nclass
    if (shape[1] == channels) != (shape[2] == channels):
        return 'yolo_raw'
    raise ValueError(f'Cannot identify detection output {shape} for {nclass} classes')


def _nms(boxes, scores, classes, iou, max_det):
    """Class-aware NMS with the same default IoU threshold as the Torch adapter."""
    order = np.argsort(-scores, kind='stable')[:30000]
    keep = []
    while order.size and len(keep) < max_det:
        current = order[0]
        keep.append(current)
        other = order[1:]
        if not other.size:
            break
        intersection = np.maximum(0, np.minimum(boxes[current, 2:], boxes[other, 2:]) -
                                  np.maximum(boxes[current, :2], boxes[other, :2])).prod(axis=1)
        area_a = np.maximum(0, boxes[current, 2:] - boxes[current, :2]).prod()
        area_b = np.maximum(0, boxes[other, 2:] - boxes[other, :2]).prod(axis=1)
        overlap = intersection / np.maximum(area_a + area_b - intersection, 1e-12)
        order = other[(classes[other] != classes[current]) | (overlap <= iou)]
    return np.asarray(keep, dtype=np.int64)


def decode_output(output, output_format, names, transform, conf=0.25, iou=0.7, max_det=300):
    """Decode YOLO26 end-to-end or YOLOv8 raw output into original image pixels."""
    conf = _confidence(conf)
    names = rover_model_names(names)
    prediction = np.asarray(output)
    if prediction.ndim == 3 and prediction.shape[0] == 1:
        prediction = prediction[0]
    if prediction.ndim != 2:
        raise ValueError(f'Expected a single batch detection tensor, received {prediction.shape}')
    if output_format == 'yolo_end2end':
        if prediction.shape[1] != 6:
            raise ValueError('End-to-end output must have xyxy, confidence, class ID columns')
        boxes, scores, classes_float = prediction[:, :4], prediction[:, 4], prediction[:, 5]
        valid_class = np.isfinite(classes_float) & (np.abs(classes_float - np.rint(classes_float)) < 1e-5)
        classes = np.rint(np.where(valid_class, classes_float, -1)).astype(np.int64)
    elif output_format == 'yolo_raw':
        channels = 4 + len(names)
        if prediction.shape[0] == channels and prediction.shape[1] != channels:
            prediction = prediction.T
        elif prediction.shape[1] != channels or prediction.shape[0] == channels:
            raise ValueError(f'Raw output must contain exactly {channels} xywh/class channels')
        centers, sizes = prediction[:, :2], prediction[:, 2:4]
        boxes = np.concatenate((centers - sizes / 2, centers + sizes / 2), axis=1)
        class_scores = prediction[:, 4:]
        classes = class_scores.argmax(axis=1)
        scores = class_scores[np.arange(len(class_scores)), classes]
        valid_class = np.isfinite(class_scores).all(axis=1)
    else:
        raise ValueError(f'Unsupported output format: {output_format}')
    valid = (valid_class & np.isfinite(boxes).all(axis=1) & np.isfinite(scores) &
             (scores >= conf) & (scores <= 1) & (classes >= 0) & (classes < len(names)) &
             (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]))
    boxes, scores, classes = boxes[valid].astype(np.float32), scores[valid], classes[valid]
    if output_format == 'yolo_raw':
        keep = _nms(boxes, scores, classes, iou, max_det)
    else:
        # YOLO26 has already selected predictions; applying NMS again changes its semantics.
        keep = np.argsort(-scores, kind='stable')[:max_det]
    boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - transform.left) / transform.scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - transform.top) / transform.scale
    source_h, source_w = transform.original_size
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, source_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, source_h)
    return [{'class_id': int(index), 'class_name': names[int(index)],
             'confidence': float(score), 'xyxy': box.tolist()}
            for box, score, index in zip(boxes, scores, classes)
            if box[2] > box[0] and box[3] > box[1]]


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_manifest(manifest_path):
    """Validate portable deployment metadata and its model before importing ORT."""
    path = Path(manifest_path).resolve()
    with path.open(encoding='utf-8') as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict) or type(manifest.get('schema_version')) is not int or manifest['schema_version'] != 1:
        raise ValueError('Unsupported model manifest schema; expected schema_version 1')
    if manifest.get('opset') != 17:
        raise ValueError('NPU experiments require ONNX opset 17')
    names = manifest.get('class_names')
    if not isinstance(names, list):
        raise ValueError('Manifest class_names must be an ordered list')
    rover_model_names(names)
    size = manifest.get('input_size')
    if (not isinstance(size, list) or len(size) != 2 or
            any(type(n) is not int or n <= 0 for n in size)):
        raise ValueError('Manifest input_size must be [height, width] positive integers')
    if manifest.get('output_format') not in ('yolo_end2end', 'yolo_raw'):
        raise ValueError('Unsupported ONNX output format')
    if manifest.get('precision') not in ('fp32', 'bf16', 'xint8'):
        raise ValueError('Unsupported model precision')
    prep = manifest.get('preprocessing', {})
    if (not isinstance(prep, dict) or any(prep.get(k) != v for k, v in _PREPROCESSING.items())):
        raise ValueError(f'Manifest preprocessing must be {_PREPROCESSING}')
    for key in ('model_sha256', 'source_checkpoint_sha256'):
        if not _digest(manifest.get(key)):
            raise ValueError(f'Manifest {key} must be a lowercase SHA-256 digest')
    model_file = manifest.get('model_file')
    if not isinstance(model_file, str) or not model_file or Path(model_file).is_absolute():
        raise ValueError('Manifest model_file must be a relative path')
    model_path = (path.parent / model_file).resolve()
    if not model_path.is_relative_to(path.parent):
        raise ValueError('Manifest model_file must stay inside its artifact directory')
    if not model_path.is_file() or sha256_file(model_path) != manifest['model_sha256']:
        raise ValueError('ONNX model missing or SHA-256 does not match the manifest')
    options = manifest.get('provider_options', {})
    if not isinstance(options, dict) or any(not isinstance(k, str) or not isinstance(v, (str, int, bool))
                                            for k, v in options.items()):
        raise ValueError('provider_options must contain string keys and scalar values')
    return manifest


def runtime_fingerprint(ort=None):
    """Include runtime binaries, kernel/driver and XRT build in cache identity."""
    versions = {}
    for package in ('onnxruntime', 'onnxruntime-vitisai', 'onnxruntime-vitisai-ep', 'vaip', 'voe', 'flexml'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    driver_files = {}
    for path in (Path('/sys/module/amdxdna/version'), Path('/sys/module/amdxdna/srcversion'),
                 Path('/sys/module/amdxdna/taint'), Path('/opt/xilinx/xrt/version.json'),
                 Path('/opt/xilinx/xrt/share/fw/amdxdna/manifest.json')):
        if path.is_file():
            try:
                driver_files[str(path)] = sha256_file(path)
            except OSError:
                driver_files[str(path)] = 'unreadable'
    xrt_root = Path(os.environ.get('XILINX_XRT', '/opt/xilinx/xrt'))
    active_driver_files = list(Path('/sys/class/accel').glob('accel*/device/fw_version'))
    active_driver_files.extend(xrt_root / name for name in
                              ('version.json', 'lib/libxrt_core.so.2', 'lib/libxrt_coreutil.so.2',
                               'lib/libxrt_driver_xdna.so.2'))
    for path in active_driver_files:
        if path.is_file():
            try:
                driver_files[str(path)] = sha256_file(path)
            except OSError:
                driver_files[str(path)] = 'unreadable'
    binaries = {}
    ort_file = getattr(ort, '__file__', None)
    if ort_file:
        capi = Path(ort_file).resolve().parent / 'capi'
        for candidate in sorted(capi.glob('*')):
            if candidate.is_file() and candidate.suffix.lower() in ('.so', '.dll', '.pyd'):
                try:
                    binaries[candidate.name] = sha256_file(candidate)
                except OSError:
                    binaries[candidate.name] = 'unreadable'
    return {'system': platform.system(), 'kernel': platform.release(),
            'architecture': platform.machine(), 'python': platform.python_version(),
            'onnxruntime': getattr(ort, '__version__', versions.get('onnxruntime', 'unknown')),
            'packages': versions, 'runtime_binaries': binaries, 'driver_files': driver_files}


def model_cache_key(model_sha256, fingerprint, provider_options=None):
    payload = {'model_sha256': model_sha256, 'runtime': fingerprint, 'provider_options': provider_options or {}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read_operator_assignment(path):
    """Read AMD's documented deviceStat report; unknown layouts remain unverified."""
    path = Path(path)
    try:
        report = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    stats = report.get('deviceStat') if isinstance(report, dict) else None
    if not isinstance(stats, list):
        return None
    counts = {}
    ops = {}
    # The integer compiler names its device NPU; the BF16 compiler names the
    # same accelerator VAIML. Unknown targets remain unverified.
    device_names = {'CPU': 'CPU', 'VITIS_EP_CPU': 'CPU', 'NPU': 'NPU',
                    'VAIML': 'NPU', 'all': 'all'}
    for row in stats:
        if not isinstance(row, dict) or row.get('name') not in device_names:
            continue
        count = row.get('nodeNum')
        if type(count) is int and count >= 0:
            name = device_names[row['name']]
            counts[name] = counts.get(name, 0) + count
            types = row.get('supportedOpType', [])
            if not isinstance(types, list):
                return None
            ops.setdefault(name, []).extend(types)
    if 'NPU' not in counts:
        # A CPU entry alone is not enough: compiler reports may omit providers.
        if 'all' not in counts or counts.get('CPU') != counts['all']:
            return None
        counts['NPU'] = 0
    cpu_nodes = counts.get('CPU')
    if cpu_nodes is None and counts.get('all') == counts['NPU']:
        cpu_nodes = 0
    return {'report_path': str(path.resolve()), 'npu_nodes': counts['NPU'],
            'cpu_nodes': cpu_nodes, 'total_nodes': counts.get('all'),
            'npu_ops': ops.get('NPU', []), 'cpu_ops': ops.get('CPU', []),
            'state': 'assigned' if counts['NPU'] else 'cpu_only'}


def compute_offload(assignment):
    """Conservative learned-compute gate, distinct from 'some NPU node ran'.

    A Conv used only for fixed DFL decoding also blocks this gate if reported on
    the CPU: operator-type summaries cannot distinguish it from a learned head.
    Such candidates need node-level inspection, rather than an optimistic pass.
    """
    critical = {'Conv', 'FusedConv', 'QLinearConv', 'ConvInteger', 'MatMul',
                'FusedMatMul', 'Gemm', 'MatMulInteger', 'QLinearMatMul',
                'MatMulNBits', 'Attention', 'MultiHeadAttention', 'QAttention'}

    def learned(ops):
        return sorted({op for op in ops if isinstance(op, str) and
                       op.rsplit('::', 1)[-1].rsplit('.', 1)[-1] in critical})

    result = {'state': 'unverified', 'cpu_compute_ops': [], 'npu_compute_ops': [],
              'reason': 'AMD operator assignment with per-device operation types is required'}
    if not assignment:
        return result
    cpu_ops, npu_ops = assignment.get('cpu_ops'), assignment.get('npu_ops')
    if not isinstance(cpu_ops, list) or not isinstance(npu_ops, list):
        return result
    result['cpu_compute_ops'], result['npu_compute_ops'] = learned(cpu_ops), learned(npu_ops)
    if result['cpu_compute_ops']:
        result.update(state='cpu', reason='CPU Conv/MatMul/Gemm or attention remains; learned computation is not verified on the NPU')
    elif result['npu_compute_ops'] and (assignment.get('cpu_nodes') == 0 or cpu_ops):
        result.update(state='npu', reason='AMD assignment places learned operation types on the NPU and none on the CPU')
    return result


class TorchDetector:
    def __init__(self, model_path=None, model=None, device='0', imgsz=960, rect=True):
        if model is None:
            if model_path is None:
                raise ValueError('A Torch checkpoint or model instance is required')
            from ultralytics import YOLO
            model = YOLO(str(model_path))
        self.model, self.device, self.imgsz, self.rect = model, device, imgsz, rect
        self.names = rover_model_names(model.names)
        self.metadata = {'backend': 'torch', 'model': str(model_path) if model_path else 'injected_model',
                         'precision': 'fp32', 'input_size': [imgsz, imgsz], 'device': str(device),
                         'class_names': list(self.names.values())}
        if model_path is not None and Path(model_path).is_file():
            digest = sha256_file(model_path)
            self.metadata.update(model_sha256=digest, source_checkpoint_sha256=digest)

    def detect(self, image, conf=0.25):
        _image(image)
        conf = _confidence(conf)
        started = time.perf_counter()
        result = self.model.predict(image, imgsz=self.imgsz, conf=conf,
                                    device=self.device, rect=self.rect, verbose=False)[0]
        inference_elapsed = (time.perf_counter() - started) * 1000
        post_started = time.perf_counter()
        self.names = rover_model_names(getattr(result, 'names', self.model.names))
        self.metadata['class_names'] = list(self.names.values())
        rows = detection_rows(result)
        row_ms = (time.perf_counter() - post_started) * 1000
        speed = getattr(result, 'speed', {}) or {}
        timings = {'preprocess_ms': float(speed.get('preprocess', 0) or 0),
                   'execution_ms': float(speed.get('inference', inference_elapsed) or 0),
                   'postprocess_ms': float(speed.get('postprocess', 0) or 0) + row_ms}
        return DetectionResult(rows, timings)

    def warmup(self):
        return self.detect(np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8))


class OnnxDetector:
    def __init__(self, manifest_path, backend='npu'):
        if backend not in ('npu', 'cpu'):
            raise ValueError('ONNX backend must be npu or cpu')
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = validate_manifest(self.manifest_path)
        self.model_path = self.manifest_path.parent / self.manifest['model_file']
        self.names = rover_model_names(self.manifest['class_names'])
        self.input_size = tuple(self.manifest['input_size'])
        self.backend = backend
        import onnxruntime as ort
        expected_provider = VITIS_PROVIDER if backend == 'npu' else 'CPUExecutionProvider'
        if expected_provider not in ort.get_available_providers():
            raise RuntimeError(f'{expected_provider} is not installed in this environment; no automatic fallback is permitted')
        fingerprint = runtime_fingerprint(ort)
        options = dict(self.manifest.get('provider_options', {}))
        config = options.get('config_file')
        if config:
            config_path = Path(config)
            if not config_path.is_absolute():
                config_path = self.manifest_path.parent / config_path
            if not config_path.is_file():
                raise ValueError(f'NPU compiler config is missing: {config_path}')
            options['config_file'] = str(config_path.resolve())
        cache_options = {k: v for k, v in options.items() if k not in ('cache_dir', 'cache_key')}
        if config:
            cache_options['config_sha256'] = sha256_file(options['config_file'])
        self.cache_key = model_cache_key(self.manifest['model_sha256'], fingerprint, cache_options)
        cache_root = self.manifest_path.parent / 'compiled'
        self.cache_dir = cache_root / self.cache_key
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        options.update(cache_dir=str(cache_root), cache_key=self.cache_key, enable_cache_file_io_in_mem='0')
        self.report_name = 'operator_assignment.json'
        self.metadata = {'backend': backend, 'model': str(self.model_path),
                         'manifest': str(self.manifest_path), 'precision': self.manifest['precision'],
                         'input_size': list(self.input_size), 'model_sha256': self.manifest['model_sha256'],
                         'source_checkpoint_sha256': self.manifest['source_checkpoint_sha256'],
                         'class_names': list(self.names.values()),
                         'runtime_fingerprint': fingerprint, 'cache_key': self.cache_key,
                         'cache_dir': str(self.cache_dir), 'providers': [],
                         'operator_assignment': None, 'proof_state': 'unverified',
                         'computation_offload_verified': False, 'compute_offload': compute_offload(None),
                         'execution_providers_seen': [], 'profile_path': None}
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 1 if backend == 'npu' else 2
        session_options.enable_profiling = backend == 'npu'
        if backend == 'npu':
            session_options.profile_file_prefix = str(self.cache_dir / 'runtime_profile')
        # ORT may leave a provider registered while placing the whole graph on CPU.
        # Capture AMD's real assignment report separately from provider availability.
        with _REPORT_ENV_LOCK:
            previous_report = os.environ.get('XLNX_ONNX_EP_REPORT_FILE')
            try:
                if backend == 'npu':
                    os.environ['XLNX_ONNX_EP_REPORT_FILE'] = self.report_name
                self.session = ort.InferenceSession(str(self.model_path), sess_options=session_options,
                                                   providers=[expected_provider],
                                                   provider_options=[{str(k): str(v).lower() if isinstance(v, bool) else str(v)
                                                                      for k, v in options.items()}] if backend == 'npu' else [{}])
            finally:
                if previous_report is None:
                    os.environ.pop('XLNX_ONNX_EP_REPORT_FILE', None)
                else:
                    os.environ['XLNX_ONNX_EP_REPORT_FILE'] = previous_report
        self.metadata['providers'] = self.session.get_providers()
        if expected_provider not in self.metadata['providers']:
            raise RuntimeError(f'{expected_provider} failed to initialize; refusing CPU fallback')
        if hasattr(self.session, 'disable_fallback'):
            self.session.disable_fallback()
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        expected_shape = [1, 3, *self.input_size]
        if len(inputs) != 1 or list(inputs[0].shape) != expected_shape or inputs[0].type != 'tensor(float)':
            raise ValueError(f'ONNX model must have one float32 input of shape {expected_shape}')
        if len(outputs) != 1 or infer_output_format(outputs[0].shape, len(self.names)) != self.manifest['output_format']:
            raise ValueError('ONNX model output shape does not match its declared detection format')
        if self.manifest.get('input_name', inputs[0].name) != inputs[0].name:
            raise ValueError('ONNX input name does not match its manifest')
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        self._profile_finished = False
        self.refresh_proof()

    def refresh_proof(self):
        if self.backend != 'npu':
            return self.metadata
        reports = sorted(self.cache_dir.rglob(self.report_name))
        for path in reports:
            assignment = read_operator_assignment(path)
            if assignment is not None:
                self.metadata['operator_assignment'] = assignment
                self.metadata['proof_state'] = assignment['state']
                if assignment['state'] == 'cpu_only':
                    raise RuntimeError('AMD operator assignment reports zero NPU nodes; refusing an all-CPU detector')
                break
        profile_path = self.metadata.get('profile_path')
        if profile_path:
            try:
                events = json.loads(Path(profile_path).read_text(encoding='utf-8'))
                providers = sorted({event.get('args', {}).get('provider') for event in events
                                    if isinstance(event, dict) and event.get('cat') == 'Node' and
                                    event.get('name', '').endswith('_kernel_time') and
                                    isinstance(event.get('dur'), (int, float)) and event['dur'] > 0 and
                                    isinstance(event.get('args'), dict) and event['args'].get('provider')})
            except (OSError, ValueError, TypeError, AttributeError):
                providers = []
            self.metadata['execution_providers_seen'] = providers
            if VITIS_PROVIDER in providers:
                self.metadata['proof_state'] = 'npu_executed'
        self.metadata['compute_offload'] = compute_offload(self.metadata.get('operator_assignment'))
        self.metadata['computation_offload_verified'] = (
            self.metadata['proof_state'] == 'npu_executed' and
            self.metadata['compute_offload']['state'] == 'npu')
        return self.metadata

    def detect(self, image, conf=0.25):
        conf = _confidence(conf)
        started = time.perf_counter()
        tensor, transform = preprocess_image(image, self.input_size)
        prepared = time.perf_counter()
        output = self.session.run([self.output_name], {self.input_name: tensor})[0]
        executed = time.perf_counter()
        rows = decode_output(output, self.manifest['output_format'], self.names, transform, conf=conf)
        finished = time.perf_counter()
        return DetectionResult(rows, {'preprocess_ms': (prepared - started) * 1000,
                                      'execution_ms': (executed - prepared) * 1000,
                                      'postprocess_ms': (finished - executed) * 1000})

    def warmup(self):
        result = self.detect(np.zeros((*self.input_size, 3), dtype=np.uint8))
        if self.backend == 'npu' and not self._profile_finished:
            self.metadata['profile_path'] = self.session.end_profiling()
            self._profile_finished = True
        self.refresh_proof()
        return result


def create_detector(backend='torch', model=None, manifest=None, device='0', imgsz=960, rect=True):
    if backend == 'torch':
        if isinstance(model, (str, Path)):
            return TorchDetector(model_path=model, device=device, imgsz=imgsz, rect=rect)
        return TorchDetector(model=model, device=device, imgsz=imgsz, rect=rect)
    if backend in ('npu', 'cpu'):
        if manifest is None:
            raise ValueError(f'{backend} backend requires a model manifest')
        return OnnxDetector(manifest, backend=backend)
    raise ValueError(f'Unsupported detector backend: {backend}')


def draw_detections(image, rows, names=None):
    """Render either backend with the same small OpenCV dependency."""
    _image(image)
    annotated = image.copy()
    colors = ((255, 128, 0), (0, 200, 255), (80, 220, 80), (200, 80, 220))
    for row in rows:
        box = np.asarray(row['xyxy'])
        if box.shape != (4,) or not np.isfinite(box).all():
            continue
        x1, y1, x2, y2 = np.rint(box).astype(int)
        color = colors[int(row['class_id']) % len(colors)]
        label = f"{row.get('class_name', (names or {}).get(row['class_id'], row['class_id']))} {row['confidence']:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(text_size[1] + baseline, y1)
        cv2.rectangle(annotated, (x1, label_y - text_size[1] - baseline),
                      (x1 + text_size[0], label_y), color, -1)
        cv2.putText(annotated, label, (x1, label_y - baseline), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return annotated
