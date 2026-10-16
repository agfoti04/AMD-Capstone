"""Prepare traceable detector candidates without changing the training environment.

Run ``python -m rover_perception.npu_prepare --help``. Heavy dependencies are
imported by their command so help and provenance checks work without Ryzen AI.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
import shutil
import socket
import tempfile
from datetime import datetime, timezone
from pathlib import Path


CLASS_NAMES = ['rock', 'pale_soil_patch', 'yellow_crystal_rock', 'mudstone_slab']
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def _versions():
    result = {'python': platform.python_version()}
    for package in ('ultralytics', 'torch', 'onnx', 'onnxruntime', 'onnxruntime-vitisai', 'amd-quark'):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def _now():
    return datetime.now(timezone.utc).isoformat()


def _relative_file(path, parent):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else parent / path).resolve(strict=True)


def resolve_dataset_manifest(path):
    """Resolve the exact reviewed export recorded by a training run."""
    path = Path(path).expanduser().resolve(strict=True)
    document = _json(path)
    if not isinstance(document.get('data_yaml'), str):
        raise ValueError('Training dataset manifest must contain data_yaml')
    return path, _relative_file(document['data_yaml'], path.parent)


def _split_images(value, root):
    values = value if isinstance(value, list) else [value]
    images = []
    for item in values:
        if not isinstance(item, str):
            raise ValueError('Dataset split must contain image directories or file lists')
        path = _relative_file(item, root)
        if path.is_dir():
            images.extend(p.resolve() for p in path.rglob('*') if p.suffix.lower() in IMAGE_SUFFIXES)
        elif path.suffix.lower() == '.txt':
            images.extend(_relative_file(line.strip(), path.parent) for line in
                          path.read_text(encoding='utf-8').splitlines() if line.strip())
        elif path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(path)
        else:
            raise ValueError(f'Unsupported dataset split path: {path}')
    if len(images) != len(set(images)):
        raise ValueError('Dataset split repeats an image')
    return sorted(images, key=str)


def create_calibration_manifest(data_yaml, expected_count=300):
    """Freeze all and only the reviewed training images, detecting split leakage."""
    import yaml

    data_yaml = Path(data_yaml).resolve(strict=True)
    data = yaml.safe_load(data_yaml.read_text(encoding='utf-8'))
    names = data.get('names')
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=int)]
    if names != CLASS_NAMES:
        raise ValueError(f'Expected the reviewed four-class dataset: {CLASS_NAMES}')
    root = _relative_file(data.get('path', '.'), data_yaml.parent)
    splits = {split: _split_images(data[split], root) for split in ('train', 'val', 'test')}
    if len(splits['train']) != expected_count:
        raise ValueError(f'Expected {expected_count} reviewed training images, found {len(splits["train"])}')
    held_out = set(splits['val']) | set(splits['test'])
    if held_out.intersection(splits['train']):
        raise ValueError('Training images overlap validation/test images')
    audit_path = data_yaml.parent / 'review_export.json'
    if not audit_path.is_file():
        raise ValueError('Use a reviewed export containing review_export.json')
    audit = _json(audit_path)
    indexed = {(row['split'], Path(row['source_image']).name): row for row in audit['audit']}
    held_out_hashes = {sha256(p) for p in held_out}
    images = []
    for image in splits['train']:
        row = indexed.get(('train', image.name))
        digest = sha256(image)
        if not row or row.get('image_sha256') != digest:
            raise ValueError(f'Training image does not match reviewed export audit: {image}')
        if digest in held_out_hashes:
            raise ValueError(f'Training image content also occurs in validation/test: {image}')
        # Restrict calibration to the export itself, including through symlinks.
        try:
            image.relative_to(data_yaml.parent)
        except ValueError as error:
            raise ValueError(f'Calibration image is outside the reviewed export: {image}') from error
        images.append({'path': str(image), 'sha256': digest, 'split': 'train'})
    digest = hashlib.sha256(json.dumps(images, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {'schema_version': 1, 'split': 'train', 'count': len(images), 'images': images,
            'sha256': digest, 'data_yaml': str(data_yaml), 'data_yaml_sha256': sha256(data_yaml),
            'review_audit_sha256': sha256(audit_path), 'class_names': names}


def validate_calibration(path):
    document = _json(path)
    images = document.get('images', [])
    if document.get('split') != 'train' or not images or document.get('count') != len(images):
        raise ValueError('Calibration manifest must list only training images')
    digest = hashlib.sha256(json.dumps(images, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if digest != document.get('sha256'):
        raise ValueError('Calibration manifest checksum changed')
    data_yaml = Path(document['data_yaml'])
    current = create_calibration_manifest(data_yaml, expected_count=len(images))
    if current != document:
        raise ValueError('Calibration images or reviewed dataset changed; prepare a fresh candidate')
    return document


def _candidate_directory(path):
    path = Path(path).expanduser().resolve()
    if path.exists():
        raise ValueError(f'Candidate already exists; choose a new output directory: {path}')
    path.mkdir(parents=True)
    return path


def _primary_prediction(value, shape):
    """Extract the exported tensor from Ultralytics' prediction/features tuple."""
    if hasattr(value, 'detach') and tuple(value.shape) == tuple(shape):
        return value.detach().cpu().numpy()
    if isinstance(value, (list, tuple)):
        for child in value:
            found = _primary_prediction(child, shape)
            if found is not None:
                return found
    if isinstance(value, dict):
        for child in value.values():
            found = _primary_prediction(child, shape)
            if found is not None:
                return found
    return None


def _decoded_parity(left, right, box_tolerance=0.5, confidence_tolerance=0.001):
    """Match independently of score-tie ordering and reject missing detections."""
    if len(left) != len(right):
        return False
    unmatched = list(right)
    for row in left:
        match = next((other for other in unmatched if row['class_id'] == other['class_id']
                      and abs(row['confidence'] - other['confidence']) <= confidence_tolerance
                      and max(abs(a-b) for a, b in zip(row['xyxy'], other['xyxy'])) <= box_tolerance), None)
        if match is None:
            return False
        unmatched.remove(match)
    return True


def configure_parity_reference(model, output_format):
    # Ultralytics 8.4.175's exporter explicitly selects one-to-one inference for
    # nms=False on its deep copy. Loaded training checkpoints can default to the
    # one-to-many head instead. Match the actual export before fuse removes the
    # unselected branch, rather than mistaking that mode difference for FP32 drift.
    reference = model.model
    if hasattr(reference, 'end2end'):
        reference.end2end = output_format == 'yolo_end2end'
    reference.eval()
    reference.fuse(verbose=False)
    return reference


def check_parity(model, onnx_path, calibration, input_size, output_format, sample_count=5):
    import cv2
    import numpy as np
    import onnxruntime as ort
    import torch
    from .detectors import decode_output, preprocess_image

    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    samples = []
    reference_model = configure_parity_reference(model, output_format)
    # Spread checks across scenes: the first scene can contain only background.
    selected = np.linspace(0, len(calibration['images']) - 1,
                           min(sample_count, len(calibration['images'])), dtype=int)
    for sample_index in selected:
        item = calibration['images'][sample_index]
        image = cv2.imread(item['path'])
        if image is None:
            raise ValueError(f'Cannot read parity sample: {item["path"]}')
        tensor, transform = preprocess_image(image, input_size)
        output = session.run(None, {session.get_inputs()[0].name: tensor})
        if len(output) != 1:
            raise ValueError('Expected one exported detection output')
        with torch.inference_mode():
            reference = _primary_prediction(reference_model(torch.from_numpy(tensor)), output[0].shape)
        if reference is None:
            raise ValueError(f'PyTorch output does not match exported shape {output[0].shape}')
        actual = output[0]
        finite = bool(np.isfinite(reference).all() and np.isfinite(actual).all())
        # ONNX may reorder equal-score end-to-end TopK detections. Compare rows
        # after deterministic sorting while decoded matching remains order free.
        compared_reference, compared_actual = reference, actual
        if output_format == 'yolo_end2end':
            def ordered(rows):
                return rows[np.lexsort(tuple(rows[:, index] for index in (3, 2, 1, 0, 4, 5)))][None]
            compared_reference, compared_actual = ordered(reference[0]), ordered(actual[0])
        raw_passed = finite and bool(np.allclose(compared_reference, compared_actual, rtol=1e-3, atol=1e-3))
        reference_rows = decode_output(reference, output_format, CLASS_NAMES, transform, conf=0.25)
        actual_rows = decode_output(actual, output_format, CLASS_NAMES, transform, conf=0.25)
        decoded_passed = _decoded_parity(reference_rows, actual_rows)
        samples.append({'image': item['path'], 'image_sha256': item['sha256'], 'finite': finite,
                        'raw_passed': raw_passed, 'decoded_passed': decoded_passed,
                        'max_absolute_error': float(np.max(np.abs(compared_reference-compared_actual))),
                        'pytorch_detections': len(reference_rows), 'onnx_detections': len(actual_rows)})
    return {'schema_version': 1, 'passed': bool(samples) and all(s['raw_passed'] and s['decoded_passed'] for s in samples),
            'sample_split': 'train', 'samples': samples, 'rtol': 1e-3, 'atol': 1e-3,
            'export_head': 'one2one' if output_format == 'yolo_end2end' else 'one2many',
            'box_tolerance_pixels': 0.5, 'confidence_tolerance': 0.001,
            'model_sha256': sha256(onnx_path), 'runtime_versions': _versions()}


def export_candidate(weights, dataset_manifest, output, imgsz=960, parity_images=5, input_height=None):
    import onnx
    from ultralytics import YOLO
    input_size = [imgsz if input_height is None else input_height, imgsz]
    if any(type(size) is not int or size <= 0 or size % 32 for size in input_size):
        raise ValueError('Static export dimensions must be positive multiples of 32')
    from .detectors import infer_output_format

    weights = Path(weights).expanduser().resolve(strict=True)
    source_manifest, data_yaml = resolve_dataset_manifest(dataset_manifest)
    calibration = create_calibration_manifest(data_yaml)
    if parity_images < 1:
        raise ValueError('At least one parity image is required')
    source_hash = sha256(weights)
    output = _candidate_directory(output)
    _write_json(output / 'calibration.json', calibration)
    with tempfile.TemporaryDirectory(prefix='export-', dir=output) as temp:
        # Ultralytics writes beside its checkpoint. Export from a copy so the
        # original checkpoint and its directory never receive model artifacts.
        copied = Path(temp) / weights.name
        shutil.copyfile(weights, copied)
        model = YOLO(str(copied))
        names = model.names
        names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
        if names != CLASS_NAMES:
            raise ValueError(f'Checkpoint classes must match the reviewed dataset: {CLASS_NAMES}')
        # In 8.4.175 nms=False forces YOLO26's one-to-one branch even when the
        # checkpoint was trained with its one-to-many branch. None exports raw
        # predictions without embedding NMS and preserves that trained branch.
        source_end2end = bool(getattr(model.model, 'end2end', False))
        exported = Path(model.export(format='onnx', imgsz=input_size, batch=1, opset=17,
                                     dynamic=False, simplify=False, half=False,
                                     nms=False if source_end2end else None, device='cpu'))
        onnx.checker.check_model(str(exported))
        graph = onnx.load(str(exported))
        def dimensions(value):
            return [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]
        if len(graph.graph.input) != 1 or dimensions(graph.graph.input[0]) != [1, 3, *input_size]:
            raise ValueError('Export must have one static batch-one RGB input')
        if len(graph.graph.output) != 1:
            raise ValueError('Export must expose one detection tensor')
        output_shape = dimensions(graph.graph.output[0])
        output_format = infer_output_format(output_shape, len(CLASS_NAMES))
        model_file = output / 'model.onnx'
        shutil.copyfile(exported, model_file)
        parity = check_parity(model, model_file, calibration, input_size, output_format, parity_images)
    if sha256(weights) != source_hash:
        raise ValueError('Source checkpoint changed while exporting')
    _write_json(output / 'parity.json', parity)
    manifest = {'schema_version': 1, 'created_utc': _now(), 'model_file': 'model.onnx',
                'model_sha256': sha256(model_file), 'source_checkpoint_sha256': source_hash,
                'source_checkpoint': str(weights), 'class_names': CLASS_NAMES, 'input_size': input_size,
                'input_name': graph.graph.input[0].name, 'output_shapes': [output_shape],
                'output_format': output_format, 'precision': 'fp32', 'opset': 17,
                'export_head': 'one2one' if output_format == 'yolo_end2end' else 'one2many',
                'preprocessing': {'color': 'rgb', 'scale': 1 / 255, 'layout': 'NCHW', 'padding': 114},
                'parity_report': 'parity.json', 'parity_report_sha256': sha256(output / 'parity.json'),
                'calibration_manifest': 'calibration.json',
                'calibration_manifest_sha256': sha256(output / 'calibration.json'),
                'dataset': {'data_yaml': str(data_yaml), 'source_manifest': str(source_manifest),
                            'source_manifest_sha256': sha256(source_manifest)},
                'runtime_versions': _versions()}
    _write_json(output / 'manifest.json', manifest)
    if not parity['passed']:
        raise ValueError(f'FP32 parity failed; quantization is blocked. Inspect {output / "parity.json"}')
    return output / 'manifest.json'


def postprocessing_nodes(graph):
    """Identify graph output processing after the learned prediction convolutions.

    Stops at the learned head convolutions rather than hard-coding YOLO node
    names. The fixed DFL projection convolution belongs to decoding, so its
    upstream softmax and reshapes are excluded too.
    """
    producers = {name: node for node in graph.graph.node for name in node.output}
    initializers = {value.name: value for value in graph.graph.initializer}
    pending = [value.name for value in graph.graph.output]
    seen, excluded, boundaries = set(), set(), set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        node = producers.get(name)
        if node is None:
            continue
        if node.op_type == 'Conv':
            weight = initializers.get(node.input[1]) if len(node.input) > 1 else None
            is_dfl = weight is not None and list(weight.dims) == [1, 16, 1, 1]
            if not is_dfl:
                boundaries.add(node.name)
                continue
        if not node.name:
            raise ValueError('Unnamed postprocessing node; cannot safely exclude it from XINT8 quantization')
        excluded.add(node.name)
        pending.extend(node.input)
    if len(boundaries) < 2 or not excluded:
        raise ValueError('Cannot identify detector head/postprocessing boundary; inspect the ONNX graph before quantization')
    return sorted(excluded)


def _load_prepared(path):
    from .detectors import validate_manifest

    path = Path(path).resolve(strict=True)
    manifest = validate_manifest(path)
    calibration_path = _relative_file(manifest['calibration_manifest'], path.parent)
    if sha256(calibration_path) != manifest.get('calibration_manifest_sha256'):
        raise ValueError('Calibration manifest changed')
    calibration = validate_calibration(calibration_path)
    return path, manifest, calibration


def runtime_decoding_names(graph, decoding_nodes):
    """Resolve ORT's NCHWc names only from known decoding graph outputs.

    ORT names an optimized CPU convolution after its output, rather than its
    original node. Do not strip arbitrary suffixes from unknown runtime nodes.
    """
    names = set(decoding_nodes)
    for node in graph.graph.node:
        if node.name in decoding_nodes and node.op_type == 'Conv':
            names.update(output + '_nchwc' for output in node.output)
    return names


def quantize_candidate(manifest_path, precision, output, per_channel=False, align_concat=None):
    import onnx
    from onnxruntime.quantization import CalibrationDataReader
    from quark.onnx import ModelQuantizer
    from quark.onnx.quantization.config.config import Config
    from quark.onnx.quantization.config.custom_config import get_default_config
    from .detectors import preprocess_image

    path, source, calibration = _load_prepared(manifest_path)
    if source['precision'] != 'fp32':
        raise ValueError('Quantize the original FP32 export, not an already quantized model')
    parity_path = _relative_file(source['parity_report'], path.parent)
    parity = _json(parity_path)
    if (sha256(parity_path) != source.get('parity_report_sha256') or not parity.get('passed')
            or parity.get('model_sha256') != source['model_sha256']):
        raise ValueError('A successful unchanged FP32 parity report is required before quantization')
    if precision not in ('bf16', 'xint8', 'a8w8', 'a16w8'):
        raise ValueError('Precision must be bf16, xint8, a8w8 or a16w8')
    if per_channel and precision not in ('a8w8', 'a16w8'):
        raise ValueError('Per-channel experiments require floating-scale A8W8 or A16W8')
    input_model = _relative_file(source['model_file'], path.parent)
    graph = onnx.load(str(input_model))
    decoding_nodes = postprocessing_nodes(graph)
    # Keep coordinates and probabilities in FP32 for both precisions. BF16
    # rounding near 960-pixel coordinates can move a small box several pixels.
    excluded = decoding_nodes
    config = copy.deepcopy(get_default_config(precision.upper()))
    if per_channel:
        config.per_channel = True
    if align_concat is not None:
        if precision not in ('a8w8', 'a16w8'):
            raise ValueError('Concat alignment override requires floating-scale integer quantization')
        config.extra_options['AlignConcat'] = align_concat
    config.extra_options['BF16QDQToCast'] = True
    if excluded:
        config.nodes_to_exclude = excluded

    class TrainingReader(CalibrationDataReader):
        # Images are loaded lazily, so 300 960px tensors never fill RAM at once.
        def __init__(self):
            self.index = 0

        def get_next(self):
            import cv2
            if self.index >= len(calibration['images']):
                return None
            item = calibration['images'][self.index]
            self.index += 1
            if sha256(item['path']) != item['sha256']:
                raise ValueError(f'Calibration image changed: {item["path"]}')
            image = cv2.imread(item['path'])
            if image is None:
                raise ValueError(f'Cannot read calibration image: {item["path"]}')
            tensor, _ = preprocess_image(image, source['input_size'])
            return {graph.graph.input[0].name: tensor}

        def rewind(self):
            self.index = 0

    output = _candidate_directory(output)
    quantizer = ModelQuantizer(Config(global_quant_config=config))
    quantizer.quantize_model(str(input_model), str(output / 'model.onnx'), TrainingReader())
    onnx.checker.check_model(str(output / 'model.onnx'))
    shutil.copyfile(path.parent / source['calibration_manifest'], output / 'calibration.json')
    shutil.copyfile(parity_path, output / 'fp32_parity.json')
    manifest = dict(source, created_utc=_now(), model_file='model.onnx',
                    model_sha256=sha256(output / 'model.onnx'), precision=precision,
                    parent_manifest=str(path), parent_model_sha256=source['model_sha256'],
                    calibration_manifest='calibration.json',
                    calibration_manifest_sha256=sha256(output / 'calibration.json'),
                    parity_report='fp32_parity.json', parity_report_sha256=sha256(output / 'fp32_parity.json'),
                    quantization={'tool': 'amd-quark', 'config': precision.upper(),
                                  'per_channel': per_channel,
                                  'align_concat': config.extra_options.get('AlignConcat'),
                                  'excluded_nodes': excluded, 'decoding_nodes': decoding_nodes,
                                  'BF16QDQToCast': True}, runtime_versions=_versions())
    if precision == 'bf16':
        # BF16 requires the VAIML compiler; the default integer provider target
        # is not a substitute for selecting this documented compilation flow.
        compiler_config = {
            'passes': [{'name': 'init', 'plugin': 'vaip-pass_init'},
                       {'name': 'vaiml_partition', 'plugin': 'vaip-pass_vaiml_partition',
                        'vaiml_config': {'optimize_level': 1, 'preferred_data_storage': 'vectorized'}}],
            'target': 'VAIML', 'targets': [{'name': 'VAIML', 'pass': ['init', 'vaiml_partition']}]}
        _write_json(output / 'vitisai_config.json', compiler_config)
        manifest['provider_options'] = {'config_file': 'vitisai_config.json'}
    else:
        manifest['provider_options'] = {'target': 'X2'}
    _write_json(output / 'manifest.json', manifest)
    return output / 'manifest.json'


def validate_hardware_evidence(path):
    evidence = _json(path)
    verification = evidence.get('verification', {})
    if (evidence.get('kind') != 'amd_npu_reference' or evidence.get('verified') is not True
            or evidence.get('returncode') != 0 or verification.get('inference_completed') is not True
            or verification.get('operators_npu', 0) <= 0):
        raise ValueError('Hardware gate requires successful AMD reference inference and NPU operator evidence')
    hostname = evidence.get('host', {}).get('hostname')
    if not hostname or hostname != socket.gethostname():
        raise ValueError('Reference evidence must be produced on this Halo host')
    return evidence


def audit_compute_offload(metadata, decoding_nodes):
    """Require learned compute on the NPU, allowing named CPU decoding nodes."""
    compute_ops = {'Conv', 'FusedConv', 'NhwcConv', 'QLinearConv', 'ConvInteger',
                   'Gemm', 'MatMul', 'FusedMatMul', 'MatMulInteger', 'QLinearMatMul',
                   'MatMulNBits', 'Attention', 'MultiHeadAttention', 'QAttention'}
    assignment = metadata.get('operator_assignment') or {}
    npu_ops = {str(op).split('::')[-1] for op in assignment.get('npu_ops', [])}
    cpu_ops = {str(op).split('::')[-1] for op in assignment.get('cpu_ops', [])}
    report = {'verified': False, 'learned_cpu_nodes': [], 'allowed_cpu_decoding_nodes': [],
              'npu_compute_ops': sorted(npu_ops.intersection(compute_ops)),
              'cpu_compute_ops': sorted(cpu_ops.intersection(compute_ops))}
    profile = metadata.get('profile_path')
    if not profile or not Path(profile).is_file():
        report['reason'] = 'Runtime profile is missing'
        return report
    events = _json(profile)
    observed_cpu_compute = set()
    for event in events:
        args = event.get('args', {})
        if (event.get('cat') != 'Node' or not event.get('name', '').endswith('_kernel_time')
                or not isinstance(event.get('dur'), (int, float)) or event['dur'] <= 0
                or args.get('provider') != 'CPUExecutionProvider' or args.get('op_name') not in compute_ops):
            continue
        observed_cpu_compute.add(args['op_name'])
        name = event.get('name', '').removesuffix('_kernel_time')
        detail = {'name': name, 'op_type': args['op_name']}
        if name in decoding_nodes:
            report['allowed_cpu_decoding_nodes'].append(detail)
        else:
            report['learned_cpu_nodes'].append(detail)
    if not report['npu_compute_ops'] or assignment.get('npu_nodes', 0) <= 0:
        report['reason'] = 'AMD operator assignment does not establish NPU convolution/matrix computation'
    elif cpu_ops.intersection(compute_ops) - observed_cpu_compute:
        report['reason'] = 'Compiler reports CPU computation without complete runtime node evidence'
    elif report['learned_cpu_nodes']:
        report['reason'] = 'Runtime profile places learned detector computation on CPU'
    else:
        report['verified'] = metadata.get('proof_state') == 'npu_executed'
    return report


def compile_candidate(manifest_path, hardware_evidence):
    import cv2
    import onnx
    from .detectors import OnnxDetector

    evidence_path = Path(hardware_evidence).resolve(strict=True)
    evidence = validate_hardware_evidence(evidence_path)
    path, manifest, calibration = _load_prepared(manifest_path)
    if manifest['precision'] not in ('bf16', 'xint8', 'a8w8', 'a16w8'):
        raise ValueError('Compile a BF16 or integer candidate after FP32 parity')
    report_path = path.parent / 'compile_report.json'
    detector = OnnxDetector(path, backend='npu')
    detector.warmup()
    image = cv2.imread(calibration['images'][0]['path'])
    if image is None:
        raise ValueError('Cannot read reference training image for candidate inference')
    result = detector.detect(image)
    detector.refresh_proof()
    metadata = detector.metadata
    if callable(metadata):
        metadata = metadata()
    graph = onnx.load(str(path.parent / manifest['model_file']))
    # Quantization adds Q/DQ nodes but preserves exported head/decoding names.
    decoding_nodes = manifest.get('quantization', {}).get('decoding_nodes') or postprocessing_nodes(graph)
    offload = audit_compute_offload(metadata, runtime_decoding_names(graph, decoding_nodes))
    report = {'schema_version': 1, 'created_utc': _now(), 'manifest': str(path),
              'manifest_sha256': sha256(path), 'model_sha256': manifest['model_sha256'],
              'hardware_evidence': str(evidence_path), 'hardware_evidence_sha256': sha256(evidence_path),
              'reference_verification': evidence['verification'], 'runtime_versions': _versions(),
              'candidate_inference_completed': True, 'backend': metadata,
              'compute_offload': offload,
              'timings': result.timings, 'detections': len(result.detections)}
    report['verified'] = metadata.get('proof_state') == 'npu_executed' and offload['verified']
    _write_json(report_path, report)
    if not report['verified']:
        raise ValueError(f'Candidate offload is unverified: {offload.get("reason", "NPU runtime proof missing")}. '
                         f'Inspect {report_path}')
    return report_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    export = sub.add_parser('export', help='CPU export and mandatory FP32 parity against the checkpoint')
    export.add_argument('--weights', required=True)
    export.add_argument('--dataset-manifest', required=True)
    export.add_argument('--output', required=True)
    export.add_argument('--imgsz', type=int, choices=(640, 960), default=960)
    export.add_argument('--input-height', type=int, help='Optional static rectangular height, multiple of 32')
    export.add_argument('--parity-images', type=int, default=5)
    quantize = sub.add_parser('quantize', help='AMD Quark BF16 or integer quantization; train-only calibration')
    quantize.add_argument('--manifest', required=True)
    quantize.add_argument('--precision', choices=('bf16', 'xint8', 'a8w8', 'a16w8'), required=True)
    quantize.add_argument('--per-channel', action='store_true', help='Per-channel weights for floating-scale integer experiments')
    quantize.add_argument('--no-align-concat', action='store_true', help='Disable Quark concat alignment for graphs with shared concat outputs')
    quantize.add_argument('--output', required=True)
    compile_parser = sub.add_parser('compile', help='Compile, execute and verify NPU assignment on the Halo')
    compile_parser.add_argument('--manifest', required=True)
    compile_parser.add_argument('--hardware-evidence', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'export':
            result = export_candidate(args.weights, args.dataset_manifest, args.output, args.imgsz,
                                      args.parity_images, args.input_height)
        elif args.command == 'quantize':
            result = quantize_candidate(args.manifest, args.precision, args.output, args.per_channel,
                                        False if args.no_align_concat else None)
        else:
            result = compile_candidate(args.manifest, args.hardware_evidence)
    except (ValueError, FileNotFoundError, ImportError, RuntimeError) as error:
        parser.exit(1, f'{type(error).__name__}: {error}\n')
    print(result)


if __name__ == '__main__':
    main()
