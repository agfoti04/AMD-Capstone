"""Readiness checks and reproducible detector experiments, without ROS imports.

The preflight subcommand uses only the standard library and never installs software.
Experiment reports are evidence, not a claim that compilation implies NPU execution.
"""
import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import socket
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


SCHEMA_VERSION = 1
IOU_THRESHOLDS = tuple(0.5 + 0.05 * i for i in range(10))


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def command(argv, timeout=20, cwd=None):
    try:
        process = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, cwd=cwd,
                                 errors='replace', check=False)
        return {'command': list(map(str, argv)), 'returncode': process.returncode,
                'stdout': process.stdout, 'stderr': process.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'command': list(map(str, argv)), 'returncode': None,
                'stdout': '', 'stderr': str(error)}


def runtime_versions():
    versions = {'python': platform.python_version()}
    for package in ('onnxruntime', 'onnxruntime-vitisai', 'onnxruntime-ryzenai', 'onnx',
                    'torch', 'ultralytics', 'amd-quark', 'numpy', 'opencv-python', 'aiohttp'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def host_identity():
    return {'hostname': socket.gethostname(), 'system': platform.system(),
            'machine': platform.machine(), 'platform': platform.platform(),
            'kernel': platform.release()}


def performance_context():
    """Read settings that affect timing, without changing hardware policy."""
    result = {'cpu_count': os.cpu_count(), 'npu_power_mode': None}
    for key, filename in (
            ('cpu_governor', '/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'),
            ('platform_profile', '/sys/firmware/acpi/platform_profile')):
        try:
            result[key] = Path(filename).read_text().strip()
        except OSError:
            result[key] = None
    executable = shutil.which('xrt-smi')
    if executable:
        report = command([executable, 'examine', '--report', 'platform', '--batch'], timeout=10)
        match = re.search(r'Power Mode\s*:\s*(\w+)', report['stdout'])
        result['npu_power_mode'] = match.group(1) if match and report['returncode'] == 0 else None
    return result


def resource_sample():
    """Read process memory and GPU utilization without loading GPU libraries."""
    sample = {'process_rss_mb': None, 'available_memory_mb': None, 'gpu_busy_percent': {}}
    try:
        sample['process_rss_mb'] = int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE') / 1048576
        memory = Path('/proc/meminfo').read_text()
        sample['available_memory_mb'] = int(re.search(r'MemAvailable:\s+(\d+)', memory).group(1)) / 1024
    except (OSError, ValueError, IndexError, AttributeError):
        pass
    for device in Path('/sys/class/drm').glob('card[0-9]*/device/gpu_busy_percent'):
        try:
            sample['gpu_busy_percent'][device.parent.parent.name] = float(device.read_text())
        except (OSError, ValueError):
            pass
    return sample


def kernel_at_least(version, required=(6, 10)):
    match = re.match(r'(\d+)\.(\d+)', version)
    return bool(match and tuple(map(int, match.groups())) >= required)


def preflight_report():
    os_release = {}
    release_file = Path('/etc/os-release')
    if release_file.exists():
        for line in release_file.read_text().splitlines():
            if '=' in line:
                key, value = line.split('=', 1)
                os_release[key] = value.strip('"')
    devices = []
    for directory in ('/dev/accel', '/dev/dri'):
        base = Path(directory)
        if base.exists():
            for device in sorted(base.iterdir()):
                info = device.stat()
                devices.append({'path': str(device), 'mode': stat.filemode(info.st_mode),
                                'uid': info.st_uid, 'gid': info.st_gid,
                                'readable': os.access(device, os.R_OK),
                                'writable': os.access(device, os.W_OK)})
    xrt = shutil.which('xrt-smi') or '/opt/xilinx/xrt/bin/xrt-smi'
    results = {
        'cpu': command(['lscpu']), 'pci': command(['lspci', '-nnk']),
        'drivers': command(['lsmod']), 'npu_driver': command(['modinfo', 'amdxdna']),
        'packages': command(['dpkg-query', '-W', '-f=${Package} ${Version}\n', '*xrt*', '*amdxdna*']),
        'xrt_smi': command([xrt, 'examine']),
        'python_providers': command([sys.executable, '-c',
            'import json,onnxruntime as o; print(json.dumps(o.get_available_providers()))']),
    }
    accel_ready = any(d['path'].startswith('/dev/accel/') and d['readable'] and d['writable']
                      for d in devices)
    prerequisites = {
        'ubuntu_24_04': os_release.get('ID') == 'ubuntu' and os_release.get('VERSION_ID') == '24.04',
        'kernel_at_least_6_10': platform.system() == 'Linux' and kernel_at_least(platform.release()),
        'python_3_12': sys.version_info[:2] == (3, 12),
        'npu_device_access': accel_ready,
        'xrt_smi_available': results['xrt_smi']['returncode'] == 0,
        'vitisai_provider': 'VitisAIExecutionProvider' in results['python_providers']['stdout'],
    }
    return {'schema_version': SCHEMA_VERSION, 'kind': 'npu_preflight', 'created_utc': utc_now(),
            'host': host_identity(), 'os_release': os_release, 'devices': devices,
            'runtime_versions': runtime_versions(), 'checks': results,
            'prerequisites': prerequisites, 'ready_for_reference_test': all(prerequisites.values()),
            'missing_prerequisites': [name for name, passed in prerequisites.items() if not passed],
            'note': 'Read-only inspection; NPU reference execution is a separate required gate.'}


def npu_operator_count(output):
    """Accept AMD's human-readable assignment output, never provider names alone."""
    patterns = (r'Operators\s*:\s*NPU\s+(\d+)', r'Operators[^\n]*?NPU\s*[:=]\s*(\d+)',
                r'\bNPU\s*:\s*(\d+)\s+(?:operators|nodes)', r'\bNPU\s+(\d+)\s+VITIS_EP_CPU')
    counts = [int(match.group(1)) for expression in patterns
              for match in re.finditer(expression, output, re.IGNORECASE)]
    # Ryzen AI 1.8 logs its assignment table on separate timestamped lines.
    # Scope this search to the operator table, excluding subgraph-only counts.
    for table in re.findall(r'No\.\s*of\s+Operators\s*:(.*?)(?=No\.\s*of\s+Subgraphs|\Z)',
                            output, re.I | re.S):
        counts.extend(int(value) for value in re.findall(r'\bNPU\s+(\d+)\b', table))
    return max(counts, default=0)


def verify_reference(quicktest, output, timeout=1200):
    quicktest = Path(quicktest).resolve(strict=True)
    if quicktest.name != 'quicktest.py':
        raise ValueError('Use the AMD SDK quicktest.py, which compiles AND runs its reference CNN.')
    xrt = shutil.which('xrt-smi') or '/opt/xilinx/xrt/bin/xrt-smi'
    before = command([xrt, 'examine'])
    execution = command([sys.executable, str(quicktest)], timeout=timeout, cwd=quicktest.parent)
    after = command([xrt, 'examine'])
    text = execution['stdout'] + '\n' + execution['stderr']
    operators = npu_operator_count(text)
    # Official Linux SDK documents "Test Finished" only after its inference run.
    completed = bool(re.search(r'\bTest\s+(?:Finished|Passed)\b|\bInference\s+completed\b', text, re.I))
    enumerated = before['returncode'] == 0 and bool(re.search(r'NPU|aie2p', before['stdout'], re.I))
    verified = execution['returncode'] == 0 and operators > 0 and completed and enumerated
    report = {'schema_version': SCHEMA_VERSION, 'kind': 'amd_npu_reference',
              'created_utc': utc_now(), 'host': host_identity(), 'verified': verified,
              'reference_script': str(quicktest), 'reference_script_sha256': sha256(quicktest),
              'reference_command': execution['command'], 'returncode': execution['returncode'],
              'stdout': execution['stdout'], 'stderr': execution['stderr'],
              'runtime_versions': runtime_versions(), 'xrt_smi_before': before, 'xrt_smi_after': after,
              'verification': {'operators_npu': operators, 'inference_completed': completed,
                               'device_enumerated': enumerated},
              'note': 'Missing execution/assignment evidence fails closed; inspect the saved SDK logs.'}
    write_json(output, report)
    return report


def detector_from_args(args):
    from .detectors import create_detector
    return create_detector(args.backend, model=Path(args.model) if args.model else None,
                           manifest=Path(args.manifest) if args.manifest else None,
                           device=args.device, imgsz=args.imgsz, rect=False)


def detector_metadata(detector):
    refresh = getattr(detector, 'refresh_proof', None)
    if refresh:
        refresh()
    metadata = dict(detector.metadata)
    if metadata.get('backend') == 'npu' and metadata.get('manifest'):
        manifest = Path(metadata['manifest'])
        path = manifest.parent / 'compile_report.json'
        valid = False
        if path.is_file():
            evidence = read_json(path)
            compiled_backend = evidence.get('backend', {})
            valid = (evidence.get('verified') is True
                     and metadata.get('proof_state') == 'npu_executed'
                     and evidence.get('candidate_inference_completed') is True
                     and evidence.get('compute_offload', {}).get('verified') is True
                     and evidence.get('manifest_sha256') == sha256(manifest)
                     and evidence.get('model_sha256') == metadata.get('model_sha256')
                     and compiled_backend.get('cache_key') == metadata.get('cache_key')
                     and compiled_backend.get('runtime_options', {}) == metadata.get('runtime_options', {})
                     and compiled_backend.get('runtime_fingerprint') == metadata.get('runtime_fingerprint'))
        metadata['compile_evidence'] = {'path': str(path), 'verified': valid,
                                        'sha256': sha256(path) if path.is_file() else None}
        if valid:
            # The node-level audit can distinguish fixed DFL decoding from a
            # learned convolution, unlike the raw operator-type summary.
            metadata['computation_offload_verified'] = True
            metadata['compute_offload'] = dict(evidence['compute_offload'], state='npu')
    return metadata


def names_list(names):
    if isinstance(names, dict):
        keys = sorted(map(int, names))
        if keys != list(range(len(keys))):
            raise ValueError('Class IDs must be contiguous from zero')
        return [str(names[key] if key in names else names[str(key)]) for key in keys]
    return [str(name) for name in names]


def summary(values):
    import numpy as np
    if not values:
        return {'count': 0, 'mean': None, 'median': None, 'p95': None}
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError('Non-finite timing values')
    return {'count': len(values), 'mean': float(array.mean()),
            'median': float(np.median(array)), 'p95': float(np.percentile(array, 95))}


def input_images(directory):
    directory = Path(directory).resolve(strict=True)
    images = sorted(path for path in directory.rglob('*.png') if not path.stem.endswith('_boxes'))
    if not images:
        raise ValueError(f'No raw PNG camera frames in {directory}')
    items = [{'name': str(path.relative_to(directory)).replace('\\', '/'), 'sha256': sha256(path)}
             for path in images]
    return images, json_hash(items)


def record_frames(bridge, output, seconds=60, fps=30):
    import cv2
    import numpy as np
    if seconds <= 0 or fps <= 0:
        raise ValueError('Recording duration and FPS must be positive')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if list(output.glob('*.png')) or (output / 'recording.json').exists():
        raise ValueError('Use an empty recording directory to avoid mixing sessions')
    records, errors, previous = [], [], None
    start = time.monotonic()
    while time.monotonic() - start < seconds:
        tick = time.monotonic()
        request = urllib.request.Request(bridge.rstrip('/') + '/frame.bgr',
                                         headers={'If-None-Match': previous} if previous else {})
        try:
            with urllib.request.urlopen(request, timeout=3) as response:
                sequence = int(response.headers['X-Frame-Sequence'])
                width, height = int(response.headers['X-Image-Width']), int(response.headers['X-Image-Height'])
                sim_time = float(response.headers['X-Sim-Time'])
                age = float(response.headers['X-Frame-Age'])
                etag = response.headers.get('ETag')
                raw = response.read()
            if age >= 3 or width <= 0 or height <= 0 or len(raw) != width * height * 3:
                raise ValueError('Camera frame is stale or malformed')
            if etag and etag == previous:
                continue
            image = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
            filename = f'{len(records):06d}_{sequence:09d}.png'
            if not cv2.imwrite(str(output / filename), image):
                raise OSError('Failed to save lossless camera frame')
            records.append({'image': filename, 'sequence': sequence, 'sim_time': sim_time,
                            'camera_age_seconds': age, 'wall_offset_seconds': time.monotonic() - start})
            previous = etag
        except urllib.error.HTTPError as error:
            if error.code != 304:
                errors.append(str(error))
            error.close()
        except (OSError, ValueError, KeyError) as error:
            errors.append(str(error))
        time.sleep(max(0, 1 / fps - (time.monotonic() - tick)))
    elapsed = time.monotonic() - start
    report = {'schema_version': SCHEMA_VERSION, 'kind': 'camera_recording', 'created_utc': utc_now(),
              'host': host_identity(), 'bridge': bridge, 'seconds': elapsed, 'frames': records,
              'recorded_fps': len(records) / elapsed, 'errors': errors}
    if records:
        _, report['input_sha256'] = input_images(output)
    write_json(output / 'recording.json', report)
    if not records:
        raise RuntimeError('No camera frames recorded; inspect recording.json')
    return report


def benchmark_detector(detector, frames, seconds=60, repeats=3, warmup=10,
                       jpeg_quality=80, max_memory_mb=4096):
    import cv2
    from .detectors import draw_detections
    if seconds <= 0 or repeats < 1 or warmup < 0:
        raise ValueError('Positive duration/repeats and nonnegative warmup are required')
    paths, fingerprint = input_images(frames)
    images, total_bytes = [], 0
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Unreadable frame: {path}')
        total_bytes += image.nbytes
        if total_bytes > max_memory_mb * 1024 * 1024:
            raise ValueError('Recording exceeds memory budget; use a shorter shared recording or --max-memory-mb')
        images.append(image)
    detector.warmup()
    for index in range(warmup):
        detector.detect(images[index % len(images)], conf=0.25)
    runs, all_timings = [], {name: [] for name in ('preprocess_ms', 'execution_ms', 'postprocess_ms',
                                                  'annotation_ms', 'encoding_ms', 'total_ms')}
    for repeat in range(repeats):
        context_before = performance_context()
        timings = {name: [] for name in ('preprocess_ms', 'execution_ms', 'postprocess_ms',
                                        'annotation_ms', 'encoding_ms', 'total_ms')}
        start = time.perf_counter()
        cpu_start = os.times()
        resources, next_resource = [], start
        index = 0
        while time.perf_counter() - start < seconds:
            tick = time.perf_counter()
            result = detector.detect(images[index % len(images)], conf=0.25)
            annotation_start = time.perf_counter()
            annotated = draw_detections(images[index % len(images)], result.detections)
            encode_start = time.perf_counter()
            ok, _ = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            if not ok:
                raise ValueError('JPEG encoding failed')
            completed = time.perf_counter()
            for key in ('preprocess_ms', 'execution_ms', 'postprocess_ms'):
                timings[key].append(float(result.timings[key]))
            timings['annotation_ms'].append((encode_start - annotation_start) * 1000)
            timings['encoding_ms'].append((completed - encode_start) * 1000)
            timings['total_ms'].append((completed - tick) * 1000)
            index += 1
            if completed >= next_resource:
                resources.append(resource_sample())
                next_resource = completed + .5
        elapsed = time.perf_counter() - start
        cpu_end = os.times()
        usage = {'cpu_percent_one_core': 100 * (cpu_end.user + cpu_end.system - cpu_start.user - cpu_start.system) / elapsed,
                 'sample_count': len(resources),
                 'process_rss_mb': summary([r['process_rss_mb'] for r in resources if r['process_rss_mb'] is not None]),
                 'available_memory_mb': summary([r['available_memory_mb'] for r in resources if r['available_memory_mb'] is not None]),
                 'gpu_busy_percent': {name: summary([r['gpu_busy_percent'][name] for r in resources if name in r['gpu_busy_percent']])
                                      for name in sorted({key for r in resources for key in r['gpu_busy_percent']})}}
        runs.append({'repeat': repeat + 1, 'frames': index, 'seconds': elapsed,
                     'processed_fps': index / elapsed,
                     'resource_usage': usage,
                     'context_before': context_before, 'context_after': performance_context(),
                     'timings': {name: summary(values) for name, values in timings.items()}})
        for name, values in timings.items():
            all_timings[name].extend(values)
    return {'schema_version': SCHEMA_VERSION, 'kind': 'detector_benchmark', 'created_utc': utc_now(),
            'host': host_identity(), 'runtime_versions': runtime_versions(),
            'detector': detector_metadata(detector), 'input_sha256': fingerprint,
            'input_frames': len(images), 'input_shapes': sorted({tuple(im.shape) for im in images}),
            'batch': 1, 'confidence': 0.25, 'jpeg_quality': jpeg_quality,
            'includes': 'preprocessing, execution, decoding, annotation, JPEG; excludes disk IO and network',
            'runs': runs, 'processed_fps': summary([run['processed_fps'] for run in runs]),
            'timings': {name: summary(values) for name, values in all_timings.items()},
            'total_latency_ms': summary(all_timings['total_ms']),
            'minimum_run_fps': min(run['processed_fps'] for run in runs)}


def load_dataset(data=None, dataset_manifest=None, split='val'):
    """Resolve a reviewed Ultralytics export without importing Ultralytics."""
    import yaml
    if bool(data) == bool(dataset_manifest):
        raise ValueError('Specify exactly one of --data and --dataset-manifest')
    manifest_path = Path(dataset_manifest).resolve(strict=True) if dataset_manifest else None
    if manifest_path:
        training = read_json(manifest_path)
        data = training['data_yaml']
        if not Path(data).is_absolute():
            # Training helper writes absolute paths; relative manifests resolve beside themselves.
            data = manifest_path.parent / data
    data_path = Path(data).resolve(strict=True)
    document = yaml.safe_load(data_path.read_text(encoding='utf-8'))
    names = names_list(document['names'])
    root = Path(document.get('path', data_path.parent))
    if not root.is_absolute():
        root = data_path.parent / root
    root = root.resolve()
    entries = document.get(split)
    if not entries:
        raise ValueError(f'Dataset has no {split} split')
    images = []
    for entry in entries if isinstance(entries, list) else [entries]:
        path = Path(entry)
        if not path.is_absolute():
            path = root / path
        if path.is_dir():
            images.extend(p for p in path.rglob('*') if p.suffix.lower() in ('.png', '.jpg', '.jpeg'))
        elif path.suffix.lower() == '.txt':
            for line in path.read_text().splitlines():
                if line.strip():
                    item = Path(line.strip())
                    images.append(item if item.is_absolute() else path.parent / item)
        else:
            raise ValueError(f'Dataset split must be an image directory or list: {path}')
    images = sorted(set(path.resolve(strict=True) for path in images))
    if not images:
        raise ValueError(f'No images in {split} split')
    samples = []
    fingerprint = []
    for image in images:
        parts = list(image.parts)
        positions = [index for index, part in enumerate(parts) if part == 'images']
        if not positions:
            raise ValueError(f'Expected YOLO images/ directory: {image}')
        parts[positions[-1]] = 'labels'
        label = Path(*parts).with_suffix('.txt')
        if not label.exists():
            raise ValueError(f'Missing label file (empty frames need an empty file): {label}')
        rows = []
        for line in label.read_text().splitlines():
            if not line.strip():
                continue
            values = list(map(float, line.split()))
            if len(values) != 5 or int(values[0]) != values[0] or not 0 <= values[0] < len(names):
                raise ValueError(f'Invalid YOLO class/box in {label}')
            if not all(math.isfinite(value) for value in values) or not all(0 <= v <= 1 for v in values[1:]):
                raise ValueError(f'Invalid normalized coordinates in {label}')
            if values[3] <= 0 or values[4] <= 0:
                raise ValueError(f'Degenerate box in {label}')
            rows.append(values)
        relative = str(image.relative_to(root)).replace('\\', '/') if image.is_relative_to(root) else image.name
        samples.append({'image': image, 'label': label, 'name': relative, 'labels': rows})
        fingerprint.append({'image': relative, 'image_sha256': sha256(image), 'label_sha256': sha256(label)})
    return samples, {'data_yaml': str(data_path), 'data_yaml_sha256': sha256(data_path),
                     'training_manifest': str(manifest_path) if manifest_path else None,
                     'split': split, 'class_names': names, 'frames': len(samples),
                     'input_sha256': json_hash(fingerprint)}


def box_iou(boxes_a, boxes_b):
    import numpy as np
    a = np.asarray(boxes_a, dtype=float).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=float).reshape(-1, 4)
    intersection = np.maximum(np.minimum(a[:, None, 2:], b[None, :, 2:]) -
                              np.maximum(a[:, None, :2], b[None, :, :2]), 0).prod(axis=2)
    area_a = np.maximum(a[:, 2:] - a[:, :2], 0).prod(axis=1)
    area_b = np.maximum(b[:, 2:] - b[:, :2], 0).prod(axis=1)
    return intersection / np.maximum(area_a[:, None] + area_b[None, :] - intersection, 1e-12)


def match_predictions(predictions, targets, threshold):
    """Greedy IoU one-to-one matching, matching Ultralytics validation semantics."""
    import numpy as np
    correct = np.zeros(len(predictions), dtype=bool)
    if not predictions or not targets:
        return correct
    ious = box_iou([row['xyxy'] for row in targets], [row['xyxy'] for row in predictions])
    classes = np.asarray([row['class_id'] for row in targets])[:, None] == np.asarray(
        [row['class_id'] for row in predictions])[None, :]
    label_index, prediction_index = np.where((ious >= threshold) & classes)
    if not len(label_index):
        return correct
    matches = np.stack((label_index, prediction_index, ious[label_index, prediction_index]), axis=1)
    matches = matches[np.argsort(-matches[:, 2], kind='stable')]
    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
    matches = matches[np.argsort(-matches[:, 2], kind='stable')]
    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
    correct[matches[:, 1].astype(int)] = True
    return correct


def average_precision(recall, precision):
    import numpy as np
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    grid = np.linspace(0, 1, 101)
    integral = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    return float(integral(np.interp(grid, mrec, mpre), grid))


def detection_metrics(frames, class_names, fixed_confidence=0.25):
    import numpy as np
    records, target_counts = [], np.zeros(len(class_names), dtype=int)
    fixed_tp, fixed_predictions = np.zeros(len(class_names), dtype=int), np.zeros(len(class_names), dtype=int)
    for frame in frames:
        predictions, targets = frame['predictions'], frame['targets']
        for row in targets:
            target_counts[int(row['class_id'])] += 1
        matched = np.column_stack([match_predictions(predictions, targets, iou) for iou in IOU_THRESHOLDS])
        for row, correct in zip(predictions, matched):
            records.append((int(row['class_id']), float(row['confidence']), correct))
        fixed = [row for row in predictions if row['confidence'] >= fixed_confidence]
        correct = match_predictions(fixed, targets, 0.5)
        for row, accepted in zip(fixed, correct):
            class_id = int(row['class_id'])
            fixed_predictions[class_id] += 1
            fixed_tp[class_id] += int(accepted)
    classes = {}
    for class_id, name in enumerate(class_names):
        subset = sorted((record for record in records if record[0] == class_id), key=lambda row: -row[1])
        count = int(target_counts[class_id])
        ap = [0.0] * len(IOU_THRESHOLDS)
        if subset and count:
            true_positive = np.asarray([row[2] for row in subset], dtype=float).cumsum(axis=0)
            false_positive = (1 - np.asarray([row[2] for row in subset], dtype=float)).cumsum(axis=0)
            for index in range(len(IOU_THRESHOLDS)):
                recall = true_positive[:, index] / count
                precision = true_positive[:, index] / np.maximum(true_positive[:, index] + false_positive[:, index], 1e-12)
                ap[index] = average_precision(recall, precision)
        classes[name] = {'class_id': class_id, 'instances': count,
                         'map50': ap[0] if count else None,
                         'map50_95': float(np.mean(ap)) if count else None,
                         'recall_at_025': float(fixed_tp[class_id] / count) if count else None,
                         'precision_at_025': float(fixed_tp[class_id] / fixed_predictions[class_id]) if fixed_predictions[class_id] else 0.0,
                         'true_positives_at_025': int(fixed_tp[class_id]),
                         'predictions_at_025': int(fixed_predictions[class_id])}
    valid = [row for row in classes.values() if row['instances']]
    return {'map50': float(np.mean([row['map50'] for row in valid])) if valid else None,
            'map50_95': float(np.mean([row['map50_95'] for row in valid])) if valid else None,
            'classes': classes, 'fixed_confidence': fixed_confidence, 'fixed_iou': 0.5,
            'ap_iou_thresholds': list(IOU_THRESHOLDS), 'ap_interpolation': '101-point precision envelope trapezoid',
            'matching': 'per-image class-aware greedy IoU, one-to-one'}


def evaluate_detector(detector, samples, dataset, confidence=0.001):
    import cv2
    names = names_list(detector.names)
    if names != dataset['class_names']:
        raise ValueError('Detector class names/order do not match dataset')
    if not 0 <= confidence < 0.25:
        raise ValueError('AP predictions require a low confidence below the fixed recall threshold 0.25')
    detector.warmup()
    frames, timings = [], []
    for sample in samples:
        image = cv2.imread(str(sample['image']), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Unreadable image: {sample["image"]}')
        height, width = image.shape[:2]
        targets = []
        for class_id, x, y, w, h in sample['labels']:
            targets.append({'class_id': int(class_id), 'xyxy': [(x - w / 2) * width,
                            (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height]})
        result = detector.detect(image, conf=confidence)
        for prediction in result.detections:
            class_id = prediction['class_id']
            if class_id != int(class_id) or not 0 <= class_id < len(names):
                raise ValueError('Detector returned an invalid class ID')
            if not all(math.isfinite(float(value)) for value in prediction['xyxy'] + [prediction['confidence']]):
                raise ValueError('Detector returned non-finite predictions')
        frames.append({'image': sample['name'], 'targets': targets, 'predictions': result.detections})
        timings.append(result.timings)
    return {'schema_version': SCHEMA_VERSION, 'kind': 'detector_evaluation', 'created_utc': utc_now(),
            'host': host_identity(), 'runtime_versions': runtime_versions(),
            'detector': detector_metadata(detector), 'dataset': dataset,
            'prediction_confidence': confidence, 'metrics': detection_metrics(frames, names),
            'timings': {key: summary([float(row[key]) for row in timings]) for key in
                        ('preprocess_ms', 'execution_ms', 'postprocess_ms')},
            'frames': frames}


def model_identity(metadata):
    """Join evaluation/benchmark only when they describe the same executable model."""
    keys = ('backend', 'model', 'model_sha256', 'source_checkpoint_sha256', 'precision',
            'input_size', 'cache_key', 'runtime_fingerprint', 'runtime_options')
    return json_hash({key: metadata.get(key) for key in keys})


def compare_candidates(baseline, candidates, benchmarks, minimum_fps=20, max_map_drop=0.02,
                       max_recall_drop=0.05):
    if baseline.get('kind') != 'detector_evaluation' or baseline['dataset']['split'] != 'val':
        raise ValueError('Candidate selection requires a validation baseline, never a test baseline')
    if baseline['detector'].get('backend') != 'torch':
        raise ValueError('The baseline must be the current torch GPU detector')
    baseline_metrics = baseline['metrics']
    if baseline_metrics['map50_95'] is None:
        raise ValueError('Baseline has no labeled ground truth')
    if baseline_metrics.get('fixed_confidence') != 0.25 or baseline_metrics.get('fixed_iou') != 0.5:
        raise ValueError('Baseline recall must use confidence 0.25 and IoU 0.5')
    benchmark_groups = {}
    input_hashes = set()
    for benchmark in benchmarks:
        if benchmark.get('kind') != 'detector_benchmark':
            raise ValueError('Expected detector benchmark reports')
        if benchmark.get('host', {}).get('hostname') != baseline.get('host', {}).get('hostname'):
            raise ValueError('Replay benchmarks and baseline evaluation must run on the same host')
        input_hashes.add(benchmark['input_sha256'])
        benchmark_groups.setdefault(model_identity(benchmark['detector']), []).append(benchmark)
    if len(input_hashes) != 1:
        raise ValueError('All benchmarks must replay the same recorded camera input set')
    baseline_benchmarks = benchmark_groups.get(model_identity(baseline['detector']), [])
    if len(baseline_benchmarks) != 1:
        raise ValueError('Exactly one matching GPU baseline benchmark is required')
    if len(baseline_benchmarks[0]['runs']) != 3 or any(row['seconds'] < 60 for row in baseline_benchmarks[0]['runs']):
        raise ValueError('GPU baseline requires three complete 60-second replay runs')
    results = []
    for candidate in candidates:
        if candidate.get('kind') != 'detector_evaluation' or candidate['dataset']['split'] != 'val':
            raise ValueError('Use validation reports to select candidates; reserve test for the selected model')
        if candidate['dataset']['input_sha256'] != baseline['dataset']['input_sha256']:
            raise ValueError('Evaluation datasets or labels differ')
        if candidate['dataset']['class_names'] != baseline['dataset']['class_names']:
            raise ValueError('Evaluation class names/order differ')
        if candidate.get('host', {}).get('hostname') != baseline.get('host', {}).get('hostname'):
            raise ValueError('Candidate and baseline evaluation must run on the same host')
        if candidate.get('prediction_confidence') != baseline.get('prediction_confidence'):
            raise ValueError('AP confidence thresholds differ')
        if candidate['metrics'].get('fixed_confidence') != 0.25 or candidate['metrics'].get('fixed_iou') != 0.5:
            raise ValueError('Expected recall at confidence 0.25 and IoU 0.5')
        matched_benchmarks = benchmark_groups.get(model_identity(candidate['detector']), [])
        reasons, checks = [], {}
        benchmark = matched_benchmarks[0] if len(matched_benchmarks) == 1 else None
        proof = candidate['detector'].get('proof_state') == 'npu_executed'
        proof = proof and candidate['detector'].get('backend') == 'npu'
        proof = proof and candidate['detector'].get('compile_evidence', {}).get('verified') is True
        if benchmark:
            proof = proof and benchmark['detector'].get('proof_state') == 'npu_executed'
            proof = proof and benchmark['detector'].get('compile_evidence', {}).get('verified') is True
            proof = proof and benchmark['detector'].get('compile_evidence', {}).get('sha256') == candidate['detector'].get('compile_evidence', {}).get('sha256')
        checks['npu_execution_proven'] = bool(proof)
        if not proof:
            reasons.append('NPU execution and learned-compute offload are not proven in both evaluation and benchmark')
        map_drop = baseline_metrics['map50_95'] - candidate['metrics']['map50_95'] if candidate['metrics']['map50_95'] is not None else None
        checks['map50_95_drop'] = map_drop
        if map_drop is None or map_drop > max_map_drop + 1e-9:
            reasons.append('mAP50-95 accuracy gate failed')
        recalls = {}
        for name in baseline['dataset']['class_names']:
            left = baseline_metrics['classes'][name]['recall_at_025']
            right = candidate['metrics']['classes'][name]['recall_at_025']
            recalls[name] = left - right if left is not None and right is not None else None
            if recalls[name] is None or recalls[name] > max_recall_drop + 1e-9:
                reasons.append(f'{name}: fixed-threshold recall gate failed or unsupported by labels')
        checks['class_recall_drops'] = recalls
        if not benchmark:
            reasons.append('Missing or ambiguous matching benchmark')
        else:
            runs = benchmark['runs']
            complete = len(runs) == 3 and all(row['seconds'] >= 60 and row['frames'] > 0 for row in runs)
            checks['three_60_second_runs'] = complete
            if not complete:
                reasons.append('Three complete 60-second benchmark runs are required')
            checks['minimum_run_fps'] = min(row['processed_fps'] for row in runs) if runs else 0
            if checks['minimum_run_fps'] < minimum_fps:
                reasons.append('Processed FPS gate failed')
        results.append({'detector': candidate['detector'], 'metrics': candidate['metrics'],
                        'checks': checks, 'qualified': not reasons, 'reasons': reasons,
                        'median_total_latency_ms': benchmark['total_latency_ms']['median'] if benchmark else None})
    qualified = [row for row in results if row['qualified']]
    qualified.sort(key=lambda row: (-row['metrics']['map50_95'], row['median_total_latency_ms']))
    return {'schema_version': SCHEMA_VERSION, 'kind': 'npu_comparison', 'created_utc': utc_now(),
            'selection_split': 'val', 'baseline_detector': baseline['detector'],
            'baseline_metrics': baseline_metrics, 'baseline_benchmark': baseline_benchmarks[0],
            'gates': {'minimum_processed_fps': minimum_fps, 'maximum_map50_95_drop': max_map_drop,
                      'maximum_class_recall_drop': max_recall_drop},
            'candidates': results, 'selected_detector': qualified[0]['detector'] if qualified else None,
            'status': 'qualified_pending_live_and_test' if qualified else 'retain_gpu',
            'remaining': ['Five-minute live drive acceptance and one final held-out test evaluation'] if qualified else []}


def live_check(url, seconds=300, interval=0.25):
    if seconds <= 0 or interval <= 0:
        raise ValueError('Duration and sample interval must be positive')
    url = url.rstrip('/')
    if not url.endswith('/status'):
        url += '/status'
    records, errors = [], []
    start = time.monotonic()
    deadline = start + seconds
    while True:
        tick = time.monotonic()
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                status = json.load(response)
            records.append({'offset_seconds': time.monotonic() - start, 'status': status})
            if len(records) == 1:
                # Include complete minute windows after the first observation.
                deadline = time.monotonic() + seconds
        except (OSError, ValueError) as error:
            errors.append({'offset_seconds': time.monotonic() - start, 'error': str(error)})
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(remaining, max(0, interval - (time.monotonic() - tick))))
    elapsed = time.monotonic() - start
    report = summarize_live_samples(records, errors, elapsed)
    report['url'] = url
    return report


def summarize_live_samples(records, errors, elapsed):
    """Use completed-frame counters, not the number of HTTP status samples."""
    ages = [row['status']['frame_age_seconds'] for row in records
            if row['status'].get('frame_age_seconds') is not None]
    fps = [float(row['status'].get('loop_fps', 0)) for row in records]
    ready = [bool(row['status'].get('ready')) for row in records]
    points, failures = [], []
    session_id = None
    last_change = None
    for record in records:
        status = record['status']
        value, identity = status.get('processed_frames'), status.get('processing_session_id')
        offset = record['offset_seconds']
        if (isinstance(value, bool) or not isinstance(value, int) or value < 0
                or not isinstance(identity, str) or not identity):
            failures.append('Missing or invalid processed-frame/session telemetry')
            continue
        if session_id is None:
            session_id, last_change = identity, offset
        elif identity != session_id:
            failures.append('Processing session restarted during observation')
        if points:
            if offset <= points[-1][0] or value < points[-1][1]:
                failures.append('Processed-frame counter reset or observation clock reversed')
            if value != points[-1][1]:
                last_change = offset
            elif offset - last_change >= 1:
                failures.append('Processing stalled for at least one second')
        points.append((offset, value))
    duration = points[-1][0] - points[0][0] if len(points) > 1 else 0
    unique_fps = (points[-1][1] - points[0][1]) / duration if duration > 0 and not failures else 0
    windows = []
    if len(points) > 1 and not failures:
        def count_at(offset):
            for (ta, ca), (tb, cb) in zip(points, points[1:]):
                if ta <= offset <= tb:
                    return ca + (cb - ca) * (offset - ta) / (tb - ta)
            raise ValueError('Counter interpolation outside observed interval')
        for index in range(int((duration + 1e-6) // 60)):
            begin = points[0][0] + index * 60
            end = min(begin + 60, points[-1][0])
            windows.append({'start_seconds': begin, 'seconds': end - begin,
                            'processed_fps': (count_at(end) - count_at(begin)) / (end - begin)})
    stage_keys = ('camera_fetch_ms', 'preprocess_ms', 'execution_ms', 'postprocess_ms',
                  'annotation_encode_ms', 'annotation_ms', 'encoding_ms')
    stage_timings = {}
    for key in stage_keys:
        values = []
        for item in records:
            status = item['status']
            value = status.get(key, status.get('detector_timings', status.get('timings', {})).get(key))
            if isinstance(value, (int, float)) and math.isfinite(value):
                values.append(float(value))
        if values:
            stage_timings[key] = summary(values)
    age_growth = None
    if len(ages) >= 8:
        window = len(ages) // 4
        age_growth = summary(ages[-window:])['median'] - summary(ages[:window])['median']
    compute_ms = sum(stage_timings.get(key, {}).get('median', 0) for key in
                     ('preprocess_ms', 'execution_ms', 'postprocess_ms'))
    fetch_ms = stage_timings.get('camera_fetch_ms', {}).get('median')
    if unique_fps >= 20:
        bottleneck = 'No below-target server processing rate observed'
    elif fetch_ms is not None and fetch_ms > compute_ms:
        bottleneck = 'Camera acquisition/source is slower than measured detector computation'
    elif compute_ms:
        bottleneck = 'Detector computation contributes to the below-target server processing rate'
    else:
        bottleneck = 'Insufficient stage telemetry; compare camera bridge and detector status'
    report = {'schema_version': SCHEMA_VERSION, 'kind': 'live_acceptance', 'created_utc': utc_now(),
              'host': host_identity(), 'seconds': elapsed, 'samples': records, 'errors': errors,
              'processed_fps_observed': unique_fps, 'reported_fps': summary(fps),
              'processing_session_id': session_id, 'counter_observation_seconds': duration,
              'minute_windows': windows, 'telemetry_failures': sorted(set(failures)),
              'camera_age_seconds': summary(ages),
              'camera_age_growth_seconds': age_growth, 'stage_timings_ms': stage_timings,
              'bottleneck_observation': bottleneck,
              'ready_fraction': sum(ready) / len(ready) if ready else 0,
              'five_minute_observation': elapsed >= 300,
              'note': 'Passive status sampling never sends drive commands. Verify held/released WASD and disconnection stop manually; browser/network latency requires client measurements.'}
    report['telemetry_gate_passed'] = (elapsed >= 300 and unique_fps >= 20 and all(ready)
                                        and len(windows) >= 5 and all(w['processed_fps'] >= 20 for w in windows)
                                        and not failures
                                        and not errors and age_growth is not None and age_growth <= 0.25)
    return report


def add_detector_arguments(parser):
    parser.add_argument('--backend', choices=('torch', 'npu', 'cpu'), required=True)
    parser.add_argument('--model')
    parser.add_argument('--manifest')
    parser.add_argument('--device', default='0')
    parser.add_argument('--imgsz', type=int, default=960)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    preflight = sub.add_parser('preflight', help='Read-only hardware/runtime report')
    preflight.add_argument('--output', required=True)
    reference = sub.add_parser('verify-reference', help='Run AMD SDK reference inference and save evidence')
    reference.add_argument('--quicktest', required=True)
    reference.add_argument('--output', required=True)
    reference.add_argument('--timeout', type=float, default=1200)
    record = sub.add_parser('record', help='Record raw bridge frames once for paired replay')
    record.add_argument('--bridge', default='http://127.0.0.1:8770')
    record.add_argument('--output', required=True)
    record.add_argument('--seconds', type=float, default=60)
    record.add_argument('--fps', type=float, default=30)
    benchmark = sub.add_parser('benchmark', help='Batch-one replay including annotation and JPEG')
    add_detector_arguments(benchmark)
    benchmark.add_argument('--frames', required=True)
    benchmark.add_argument('--output', required=True)
    benchmark.add_argument('--seconds', type=float, default=60)
    benchmark.add_argument('--repeats', type=int, default=3)
    benchmark.add_argument('--warmup', type=int, default=10)
    benchmark.add_argument('--max-memory-mb', type=int, default=4096)
    evaluate = sub.add_parser('evaluate', help='Shared accuracy evaluator for both detector backends')
    add_detector_arguments(evaluate)
    source = evaluate.add_mutually_exclusive_group(required=True)
    source.add_argument('--data')
    source.add_argument('--dataset-manifest')
    evaluate.add_argument('--split', choices=('val', 'test'), default='val')
    evaluate.add_argument('--output', required=True)
    evaluate.add_argument('--confidence', type=float, default=0.001)
    compare = sub.add_parser('compare', help='Select using validation accuracy and matching replay reports')
    compare.add_argument('--baseline-evaluation', required=True)
    compare.add_argument('--candidate-evaluations', nargs='+', required=True)
    compare.add_argument('--benchmarks', nargs='+', required=True)
    compare.add_argument('--output', required=True)
    live = sub.add_parser('live-check', help='Passive five-minute dashboard health observation')
    live.add_argument('--url', default='http://127.0.0.1:8771')
    live.add_argument('--seconds', type=float, default=300)
    live.add_argument('--interval', type=float, default=0.25)
    live.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'preflight':
            report = preflight_report()
        elif args.command == 'verify-reference':
            report = verify_reference(args.quicktest, args.output, args.timeout)
        elif args.command == 'record':
            report = record_frames(args.bridge, args.output, args.seconds, args.fps)
            print(json.dumps({'recording': str(Path(args.output) / 'recording.json'),
                              'frames': len(report['frames']), 'fps': report['recorded_fps']}))
            return 0
        elif args.command == 'benchmark':
            report = benchmark_detector(detector_from_args(args), args.frames, args.seconds,
                                        args.repeats, args.warmup, max_memory_mb=args.max_memory_mb)
        elif args.command == 'evaluate':
            samples, dataset = load_dataset(args.data, args.dataset_manifest, args.split)
            report = evaluate_detector(detector_from_args(args), samples, dataset, args.confidence)
        elif args.command == 'compare':
            report = compare_candidates(read_json(args.baseline_evaluation),
                        [read_json(path) for path in args.candidate_evaluations],
                        [read_json(path) for path in args.benchmarks])
        else:
            report = live_check(args.url, args.seconds, args.interval)
        write_json(args.output, report)
        print(json.dumps({'report': str(Path(args.output).resolve()), 'kind': report['kind'],
                          'verified': report.get('verified'), 'status': report.get('status'),
                          'missing_prerequisites': report.get('missing_prerequisites')}, indent=2))
        return 2 if args.command == 'verify-reference' and not report['verified'] else 0
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        parser.exit(2, f'{type(error).__name__}: {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
