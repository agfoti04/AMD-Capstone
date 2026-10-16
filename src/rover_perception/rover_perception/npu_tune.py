"""Create isolated scheduling/compiler experiments from a prepared NPU model.

No quantization, dataset editing, package installation or power-mode changes.
Compile each new manifest with npu_prepare before evaluation or live use.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil

from .detectors import validate_manifest, validate_runtime_options
from .npu_tools import sha256, utc_now, write_json


def create_variant(manifest_path, output, threads=None, spinning=None, opt_level=None, external_dfl=False):
    source = Path(manifest_path).resolve(strict=True)
    manifest = copy.deepcopy(validate_manifest(source))
    runtime = dict(manifest.get('runtime_options', {}))
    if threads is not None:
        runtime['intra_op_num_threads'] = threads
    if spinning is not None:
        runtime['allow_spinning'] = spinning
    validate_runtime_options(runtime)
    if opt_level is not None:
        if manifest['precision'] not in ('xint8', 'a8w8', 'a16w8') or opt_level not in (0, 1, 2, 3, 65536):
            raise ValueError('Compiler opt_level requires integer quantization and 0, 1, 2, 3 or 65536')
        manifest.setdefault('provider_options', {})['opt_level'] = opt_level
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite experiment: {destination}')
    # Validate every source before creating the destination. Copies preserve
    # originals, including calibration hashes and FP32 parity evidence.
    artifacts = {}
    for key in ('model_file', 'calibration_manifest', 'parity_report', 'dfl_parity_report'):
        if key in manifest:
            path = (source.parent / manifest[key]).resolve(strict=True)
            if not path.is_file():
                raise ValueError(f'Missing candidate artifact: {path}')
            artifacts[key] = path
    config = manifest.get('provider_options', {}).get('config_file')
    if config:
        artifacts['config_file'] = (source.parent / config).resolve(strict=True)
    if len({path.name for path in artifacts.values()}) != len(artifacts):
        raise ValueError('Candidate artifacts must have distinct filenames')
    destination.mkdir(parents=True)
    for key, path in artifacts.items():
        shutil.copy2(path, destination / path.name)
        if key == 'config_file':
            manifest['provider_options'][key] = path.name
        else:
            manifest[key] = path.name
    manifest.update(created_utc=utc_now(), runtime_options=runtime)
    manifest['optimization_parent'] = {'manifest': str(source), 'sha256': sha256(source)}
    if external_dfl:
        externalize_dfl(destination, manifest)
    write_json(destination / 'manifest.json', manifest)
    validate_manifest(destination / 'manifest.json')
    return destination / 'manifest.json'


def externalize_dfl(directory, manifest):
    """Cut only a verified YOLOv8 fixed decoder, preserving all learned layers.

    This is deliberately specific to the exported 16-bin YOLOv8 head. Refuse
    unfamiliar graphs instead of guessing which tensors represent distances.
    """
    import cv2
    import numpy as np
    import onnx
    import onnxruntime as ort
    from .detectors import decode_output, preprocess_image
    from .npu_prepare import _decoded_parity

    if manifest['output_format'] != 'yolo_raw':
        raise ValueError('External DFL requires a standard YOLOv8 raw-output model')
    path = directory / manifest['model_file']
    model = onnx.shape_inference.infer_shapes(onnx.load(str(path)))
    nodes = {node.name: node for node in model.graph.node}
    dfl = nodes.get('/model.22/dfl/conv/Conv')
    sigmoid = nodes.get('/model.22/Sigmoid')
    boxes = nodes.get('/model.22/Concat')
    if dfl is None or sigmoid is None or boxes is None or sigmoid.op_type != 'Sigmoid':
        raise ValueError('Unrecognized YOLOv8 DFL graph')
    weights = {value.name: value for value in model.graph.initializer}
    weight = onnx.numpy_helper.to_array(weights[dfl.input[1]])
    if weight.shape != (1, 16, 1, 1) or not np.array_equal(weight.ravel(), np.arange(16)):
        raise ValueError('DFL projection must be the fixed 0..15 expectation')
    values = {value.name: value for value in list(model.graph.value_info) + list(model.graph.output)}
    box_name, class_name = boxes.output[0], sigmoid.input[0]
    shapes = [tuple(dim.dim_value for dim in values[name].type.tensor_type.shape.dim)
              for name in (box_name, class_name)]
    h, w = manifest['input_size']
    anchors = sum((h // stride) * (w // stride) for stride in (8, 16, 32))
    if h % 32 or w % 32 or shapes != [(1, 64, anchors), (1, len(manifest['class_names']), anchors)]:
        raise ValueError(f'Unrecognized DFL head shapes: {shapes}')
    original_options = ort.SessionOptions()
    original_options.intra_op_num_threads = 1
    original = ort.InferenceSession(str(path), sess_options=original_options,
                                   providers=['CPUExecutionProvider'])
    output_name = 'rover_dfl_logits'
    model.graph.node.append(onnx.helper.make_node('Concat', [box_name, class_name], [output_name],
                                                axis=1, name='rover_dfl_concat'))
    model.graph.output.append(onnx.helper.make_tensor_value_info(
        output_name, onnx.TensorProto.FLOAT, [1, 64 + len(manifest['class_names']), anchors]))
    reduced = onnx.utils.Extractor(model).extract_model([model.graph.input[0].name], [output_name])
    onnx.checker.check_model(reduced)
    onnx.save(reduced, str(path))
    candidate = ort.InferenceSession(str(path), sess_options=original_options,
                                    providers=['CPUExecutionProvider'])
    calibration = json.loads((directory / manifest['calibration_manifest']).read_text())
    checks = []
    for item in calibration['images'][::max(1, len(calibration['images']) // 5)][:5]:
        if sha256(item['path']) != item['sha256']:
            raise ValueError('DFL parity training image hash changed')
        image = cv2.imread(item['path'])
        tensor, transform = preprocess_image(image, manifest['input_size'])
        left = original.run(None, {original.get_inputs()[0].name: tensor})[0]
        right = candidate.run(None, {candidate.get_inputs()[0].name: tensor})[0]
        for confidence in (0.001, 0.25):
            passed = _decoded_parity(decode_output(left, 'yolo_raw', manifest['class_names'], transform, confidence),
                                     decode_output(right, 'yolo_dfl', manifest['class_names'], transform, confidence),
                                     box_tolerance=0.01, confidence_tolerance=1e-6)
            checks.append({'image': item['path'], 'confidence': confidence, 'passed': passed})
    report = {'passed': bool(checks) and all(row['passed'] for row in checks), 'checks': checks,
              'parent_model_sha256': manifest['model_sha256'], 'model_sha256': sha256(path),
              'box_tolerance_px': 0.01, 'confidence_tolerance': 1e-6}
    write_json(directory / 'dfl_parity.json', report)
    if not report['passed']:
        raise ValueError(f'External DFL parity failed; inspect {directory / "dfl_parity.json"}')
    manifest.update(model_sha256=report['model_sha256'], output_format='yolo_dfl',
                    output_shapes=[[1, 64 + len(manifest['class_names']), anchors]],
                    decoder={'reg_max': 16, 'strides': [8, 16, 32]},
                    dfl_parity_report='dfl_parity.json', dfl_parity_report_sha256=sha256(directory / 'dfl_parity.json'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--threads', type=int)
    parser.add_argument('--spinning', choices=('on', 'off'))
    parser.add_argument('--opt-level', type=int, choices=(0, 1, 2, 3, 65536))
    parser.add_argument('--external-dfl', action='store_true', help='Move verified fixed DFL decoding to sparse FP32 NumPy')
    args = parser.parse_args(argv)
    try:
        result = create_variant(args.manifest, args.output, args.threads,
                                None if args.spinning is None else args.spinning == 'on', args.opt_level,
                                args.external_dfl)
    except (ValueError, OSError) as error:
        parser.exit(2, f'{type(error).__name__}: {error}\n')
    print(json.dumps({'manifest': str(result)}))


if __name__ == '__main__':
    main()
