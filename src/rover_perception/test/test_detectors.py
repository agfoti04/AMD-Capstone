"""Backend behavior tests without ROCm, ONNX Runtime, an NPU or model downloads."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from rover_perception.detectors import (
    DetectionResult, OnnxDetector, TorchDetector, VITIS_PROVIDER, create_detector,
    compute_offload, decode_output, draw_detections, infer_output_format, model_cache_key,
    preprocess_image, read_operator_assignment, validate_manifest,
)


NAMES = ['rock', 'pale_soil_patch', 'yellow_crystal_rock', 'mudstone_slab']


class FakeOrt:
    __version__ = 'test-runtime'

    def __init__(self):
        self.available = [VITIS_PROVIDER, 'CPUExecutionProvider']
        self.initialized = None
        self.input_shape = [1, 3, 8, 8]
        self.input_type = 'tensor(float)'
        self.output_shape = [1, 2, 6]
        self.output = np.array([[[2, 2, 6, 6, 0.9, 3], [0, 0, 0, 0, 0, 0]]], dtype=np.float32)
        self.assignment = None
        self.profile_providers = [VITIS_PROVIDER]
        self.run_error = None
        self.calls = []
        self.SessionOptions = types.SimpleNamespace

    def get_available_providers(self):
        return self.available

    def InferenceSession(self, model_path, sess_options, providers, provider_options):
        owner = self
        options = provider_options[0]
        self.calls.append((model_path, sess_options, providers, options))
        cache = Path(options['cache_dir']) / options['cache_key'] if options else Path(model_path).parent
        cache.mkdir(parents=True, exist_ok=True)
        report_name = os.environ.get('XLNX_ONNX_EP_REPORT_FILE')
        if self.assignment is not None and report_name:
            (cache / report_name).write_text(json.dumps(self.assignment), encoding='utf-8')

        class Session:
            fallback_disabled = False
            last_input = None

            def get_providers(self):
                return owner.initialized if owner.initialized is not None else providers + (
                    ['CPUExecutionProvider'] if VITIS_PROVIDER in providers else [])

            def disable_fallback(self):
                self.fallback_disabled = True

            def get_inputs(self):
                return [types.SimpleNamespace(name='images', shape=owner.input_shape, type=owner.input_type)]

            def get_outputs(self):
                return [types.SimpleNamespace(name='output0', shape=owner.output_shape)]

            def run(self, outputs, feeds):
                if owner.run_error:
                    raise owner.run_error
                self.last_input = feeds['images']
                return [owner.output]

            def end_profiling(self):
                target = cache / 'profile.json'
                events = [{'cat': 'Node', 'name': 'fused_kernel_time', 'dur': 100,
                           'args': {'provider': provider}} for provider in owner.profile_providers]
                target.write_text(json.dumps(events), encoding='utf-8')
                return str(target)

        return Session()


class PreparationTests(unittest.TestCase):
    def test_bgr_to_rgb_normalization_and_padding(self):
        image = np.full((2, 4, 3), (10, 20, 30), dtype=np.uint8)
        tensor, transform = preprocess_image(image, (8, 8))
        self.assertEqual(tensor.shape, (1, 3, 8, 8))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(tensor.flags.c_contiguous)
        np.testing.assert_allclose(tensor[0, :, 2, 0], [30 / 255, 20 / 255, 10 / 255])
        np.testing.assert_allclose(tensor[0, :, 0, 0], [114 / 255] * 3)
        self.assertEqual((transform.scale, transform.left, transform.top), (2, 0, 2))

    def test_live_camera_has_210_pixels_padding_at_960(self):
        tensor, transform = preprocess_image(np.zeros((540, 960, 3), np.uint8), (960, 960))
        self.assertEqual((transform.scale, transform.top, transform.left), (1, 210, 0))
        self.assertEqual(tensor.shape, (1, 3, 960, 960))

    def test_odd_padding_is_asymmetric_without_stretching(self):
        _, transform = preprocess_image(np.zeros((3, 5, 3), np.uint8), (8, 8))
        self.assertEqual(transform.top, 1)
        self.assertEqual(transform.scale, 1.6)

    def test_invalid_images_rejected(self):
        for image in (np.zeros((2, 2, 3), np.float32), np.zeros((2, 2), np.uint8),
                      np.zeros((0, 2, 3), np.uint8), None):
            with self.subTest(image=type(image)), self.assertRaises(ValueError):
                preprocess_image(image, (8, 8))

    def test_identify_output_layouts(self):
        self.assertEqual(infer_output_format([1, 300, 6], 4), 'yolo_end2end')
        self.assertEqual(infer_output_format([1, 8, 18900], 4), 'yolo_raw')
        self.assertEqual(infer_output_format([1, 18900, 8], 4), 'yolo_raw')
        for shape in ([2, 300, 6], [1, 8, 8], [1, 32, 100], [300, 6]):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                infer_output_format(shape, 4)


class DecodeTests(unittest.TestCase):
    def setUp(self):
        _, self.transform = preprocess_image(np.zeros((2, 4, 3), np.uint8), (8, 8))

    def test_end2end_inverse_scaling_and_four_class_mapping(self):
        output = np.array([[[2, 2, 6, 6, 0.8, 3]]], np.float32)
        rows = decode_output(output, 'yolo_end2end', NAMES, self.transform)
        self.assertEqual(rows[0]['class_name'], 'mudstone_slab')
        np.testing.assert_allclose(rows[0]['xyxy'], [1, 0, 3, 2])

    def test_end2end_does_not_apply_second_nms(self):
        output = np.array([[[2, 2, 6, 6, 0.8, 3], [2, 2, 6, 6, 0.7, 3]]], np.float32)
        self.assertEqual(len(decode_output(output, 'yolo_end2end', NAMES, self.transform)), 2)

    def test_raw_classwise_nms_keeps_other_classes(self):
        raw_rows = np.array([[4, 4, 4, 4, 0.9, 0.0, 0.0, 0.0],
                             [4, 4, 4, 4, 0.8, 0.0, 0.0, 0.0],
                             [4, 4, 4, 4, 0.1, 0.85, 0.0, 0.0]], np.float32)
        for output in (raw_rows[None], raw_rows.T[None]):
            rows = decode_output(output, 'yolo_raw', NAMES, self.transform)
            self.assertEqual([row['class_id'] for row in rows], [0, 1])
            np.testing.assert_allclose(rows[0]['xyxy'], [1, 0, 3, 2])

    def test_three_class_checkpoint_compatibility(self):
        output = np.array([[[2, 2, 6, 6, 0.8, 2]]], np.float32)
        rows = decode_output(output, 'yolo_end2end', NAMES[:3], self.transform)
        self.assertEqual(rows[0]['class_name'], 'yellow_crystal_rock')

    def test_empty_predictions(self):
        self.assertEqual(decode_output(np.zeros((1, 0, 6)), 'yolo_end2end', NAMES, self.transform), [])
        self.assertEqual(decode_output(np.zeros((1, 8, 0)), 'yolo_raw', NAMES, self.transform), [])

    def test_invalid_predictions_dropped(self):
        invalid = [[2, 2, 6, 6, np.nan, 0], [2, 2, np.inf, 6, 0.8, 0],
                   [2, 2, 6, 6, 0.8, 4], [2, 2, 6, 6, 0.8, 1.2],
                   [2, 2, 6, 6, 1.2, 0], [6, 2, 2, 6, 0.8, 0],
                   [0, 0, 2, 1, 0.8, 0], [2, 2, 6, 6, 0.1, 0]]
        self.assertEqual(decode_output(np.array([invalid]), 'yolo_end2end', NAMES, self.transform), [])

    def test_wrong_output_and_confidence_are_errors(self):
        for output, fmt in ((np.zeros((2, 3, 6)), 'yolo_end2end'),
                            (np.zeros((1, 3, 5)), 'yolo_end2end'),
                            (np.zeros((1, 6, 3)), 'yolo_raw')):
            with self.assertRaises(ValueError):
                decode_output(output, fmt, NAMES, self.transform)
        for conf in (-1, 2, np.nan, True):
            with self.assertRaises(ValueError):
                decode_output(np.zeros((1, 0, 6)), 'yolo_end2end', NAMES, self.transform, conf=conf)


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.path = self.directory / 'manifest.json'
        self.model = self.directory / 'model.onnx'
        self.model.write_bytes(b'fake model content')
        self.manifest = {'schema_version': 1, 'model_file': 'model.onnx',
                         'model_sha256': hashlib.sha256(self.model.read_bytes()).hexdigest(),
                         'source_checkpoint_sha256': 'a' * 64, 'class_names': NAMES,
                         'input_size': [8, 8], 'output_format': 'yolo_end2end', 'precision': 'bf16',
                         'opset': 17, 'preprocessing': {'color': 'rgb', 'scale': 1 / 255,
                                                       'layout': 'NCHW', 'padding': 114}}
        self.write_manifest()
        self.ort = FakeOrt()
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules, {'onnxruntime': self.ort}).start()
        patch('rover_perception.detectors.runtime_fingerprint', return_value={'runtime': 'one', 'driver': 'one'}).start()

    def write_manifest(self, **changes):
        self.manifest.update(changes)
        self.path.write_text(json.dumps(self.manifest), encoding='utf-8')

    def test_manifest_checks_hash_class_order_and_preprocessing(self):
        self.assertEqual(validate_manifest(self.path)['precision'], 'bf16')
        for key, value in (('model_sha256', '0' * 64), ('class_names', list(reversed(NAMES))),
                           ('opset', 18), ('input_size', [8, '8']), ('precision', 'fp16'),
                           ('model_file', '../model.onnx'), ('preprocessing', {'color': 'bgr'})):
            previous = self.manifest[key]
            self.write_manifest(**{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_manifest(self.path)
            self.write_manifest(**{key: previous})

    def test_missing_provider_fails_before_session_creation(self):
        self.ort.available = ['CPUExecutionProvider']
        with self.assertRaisesRegex(RuntimeError, 'no automatic fallback'):
            OnnxDetector(self.path)
        self.assertEqual(self.ort.calls, [])

    def test_provider_initialization_fallback_is_rejected(self):
        self.ort.initialized = ['CPUExecutionProvider']
        with self.assertRaisesRegex(RuntimeError, 'refusing CPU fallback'):
            OnnxDetector(self.path)

    def test_bad_graph_input_shape_or_dtype_rejected(self):
        for shape, dtype in (([1, 3, 'height', 8], 'tensor(float)'),
                             ([2, 3, 8, 8], 'tensor(float)'),
                             ([1, 8, 8, 3], 'tensor(float)'),
                             ([1, 3, 8, 8], 'tensor(float16)')):
            self.ort.input_shape, self.ort.input_type = shape, dtype
            with self.subTest(shape=shape, dtype=dtype), self.assertRaisesRegex(ValueError, 'one float32 input'):
                OnnxDetector(self.path)

    def test_output_format_and_input_name_checked_against_graph(self):
        self.write_manifest(output_format='yolo_raw')
        with self.assertRaisesRegex(ValueError, 'output shape'):
            OnnxDetector(self.path)
        self.write_manifest(output_format='yolo_end2end', input_name='wrong')
        with self.assertRaisesRegex(ValueError, 'input name'):
            OnnxDetector(self.path)

    def test_detection_matches_contract_and_disables_runtime_fallback(self):
        detector = OnnxDetector(self.path)
        result = detector.detect(np.zeros((2, 4, 3), np.uint8))
        self.assertIsInstance(result, DetectionResult)
        self.assertTrue(detector.session.fallback_disabled)
        self.assertEqual(result.detections[0]['class_name'], 'mudstone_slab')
        self.assertEqual(set(result.timings), {'preprocess_ms', 'execution_ms', 'postprocess_ms'})
        self.assertTrue(all(n >= 0 for n in result.timings.values()))
        self.assertEqual(detector.session.last_input.shape, (1, 3, 8, 8))

    def test_provider_registration_does_not_prove_npu_execution(self):
        detector = OnnxDetector(self.path)
        self.assertEqual(detector.metadata['proof_state'], 'unverified')
        self.ort.profile_providers = ['CPUExecutionProvider']
        detector.warmup()
        self.assertEqual(detector.metadata['proof_state'], 'unverified')

    def test_assignment_and_runtime_proof_are_distinct(self):
        self.ort.assignment = {'deviceStat': [{'name': 'all', 'nodeNum': 10},
                                             {'name': 'NPU', 'nodeNum': 9, 'supportedOpType': ['::Conv']},
                                             {'name': 'CPU', 'nodeNum': 1, 'supportedOpType': ['::Reshape']}]}
        detector = OnnxDetector(self.path)
        self.assertEqual(detector.metadata['proof_state'], 'assigned')
        self.assertFalse(detector.metadata['computation_offload_verified'])
        self.assertEqual(detector.metadata['operator_assignment']['npu_nodes'], 9)
        detector.warmup()
        self.assertEqual(detector.metadata['proof_state'], 'npu_executed')
        self.assertEqual(detector.metadata['execution_providers_seen'], [VITIS_PROVIDER])
        self.assertTrue(detector.metadata['computation_offload_verified'])
        self.assertTrue(Path(detector.metadata['profile_path']).is_file())

    def test_npu_runtime_event_alone_does_not_prove_learned_compute_offload(self):
        self.ort.assignment = {'deviceStat': [{'name': 'all', 'nodeNum': 10},
                                             {'name': 'NPU', 'nodeNum': 1, 'supportedOpType': ['::Reshape']},
                                             {'name': 'CPU', 'nodeNum': 9, 'supportedOpType': ['::Conv']}]}
        detector = OnnxDetector(self.path)
        detector.warmup()
        self.assertEqual(detector.metadata['proof_state'], 'npu_executed')
        self.assertFalse(detector.metadata['computation_offload_verified'])
        self.assertEqual(detector.metadata['compute_offload']['cpu_compute_ops'], ['::Conv'])

    def test_bf16_vaiml_assignment_requires_runtime_execution_too(self):
        self.ort.assignment = {'deviceStat': [{'name': 'all', 'nodeNum': 100},
                                             {'name': 'VAIML', 'nodeNum': 100,
                                              'supportedOpType': ['::Conv', '::MatMul']}]}
        detector = OnnxDetector(self.path)
        self.assertFalse(detector.metadata['computation_offload_verified'])
        detector.warmup()
        self.assertEqual(detector.metadata['operator_assignment']['npu_nodes'], 100)
        self.assertEqual(detector.metadata['operator_assignment']['cpu_nodes'], 0)
        self.assertTrue(detector.metadata['computation_offload_verified'])

    def test_cpu_compute_and_incomplete_assignment_not_approved(self):
        npu_only = {'npu_nodes': 10, 'cpu_nodes': 0, 'npu_ops': ['::Conv'], 'cpu_ops': []}
        self.assertEqual(compute_offload(npu_only)['state'], 'npu')
        for critical in ('Conv', 'MatMul', 'Gemm', 'Attention'):
            assignment = dict(npu_only, cpu_nodes=1, cpu_ops=['::' + critical])
            self.assertEqual(compute_offload(assignment)['state'], 'cpu')
        self.assertEqual(compute_offload(dict(npu_only, cpu_nodes=1))['state'], 'unverified')
        self.assertEqual(compute_offload(None)['state'], 'unverified')

    def test_runtime_fence_event_is_not_kernel_execution_proof(self):
        detector = OnnxDetector(self.path)
        profile = self.directory / 'fence.json'
        profile.write_text(json.dumps([{'cat': 'Node', 'name': 'fence_before', 'dur': 10,
                                       'args': {'provider': VITIS_PROVIDER}}]), encoding='utf-8')
        detector.metadata['profile_path'] = str(profile)
        detector.refresh_proof()
        self.assertEqual(detector.metadata['proof_state'], 'unverified')

    def test_confirmed_all_cpu_graph_is_rejected(self):
        self.ort.assignment = {'deviceStat': [{'name': 'all', 'nodeNum': 10},
                                             {'name': 'CPU', 'nodeNum': 10}, {'name': 'NPU', 'nodeNum': 0}]}
        with self.assertRaisesRegex(RuntimeError, 'zero NPU nodes'):
            OnnxDetector(self.path)

    def test_cpu_parity_adapter_and_unknown_backend(self):
        detector = create_detector('cpu', manifest=self.path)
        detector.warmup()
        self.assertEqual(detector.metadata['providers'], ['CPUExecutionProvider'])
        self.assertFalse(self.ort.calls[0][1].enable_profiling)
        with self.assertRaises(ValueError):
            create_detector('unknown')
        with self.assertRaises(ValueError):
            create_detector('npu')

    def test_inference_errors_are_not_hidden(self):
        detector = OnnxDetector(self.path)
        self.ort.run_error = RuntimeError('NPU execution failed')
        with self.assertRaisesRegex(RuntimeError, 'NPU execution failed'):
            detector.warmup()
        self.assertEqual(detector.metadata['proof_state'], 'unverified')

    def test_report_environment_restored_after_compilation(self):
        with patch.dict(os.environ, {'XLNX_ONNX_EP_REPORT_FILE': 'previous_report.json'}):
            OnnxDetector(self.path)
            self.assertEqual(os.environ['XLNX_ONNX_EP_REPORT_FILE'], 'previous_report.json')

    def test_cache_changes_with_runtime_driver_model_and_options(self):
        base = model_cache_key('a' * 64, {'runtime': 'one', 'driver': 'one'}, {'target': 'X2'})
        for model, fingerprint, options in (
                ('b' * 64, {'runtime': 'one', 'driver': 'one'}, {'target': 'X2'}),
                ('a' * 64, {'runtime': 'two', 'driver': 'one'}, {'target': 'X2'}),
                ('a' * 64, {'runtime': 'one', 'driver': 'two'}, {'target': 'X2'}),
                ('a' * 64, {'runtime': 'one', 'driver': 'one'}, {'target': 'X1'})):
            self.assertNotEqual(base, model_cache_key(model, fingerprint, options))
        first = OnnxDetector(self.path).cache_key
        with patch('rover_perception.detectors.runtime_fingerprint', return_value={'runtime': 'two', 'driver': 'one'}):
            self.assertNotEqual(first, OnnxDetector(self.path).cache_key)

    def test_unknown_assignment_layout_is_not_proof(self):
        path = self.directory / 'unknown.json'
        path.write_text('{"provider":"VitisAIExecutionProvider"}', encoding='utf-8')
        self.assertIsNone(read_operator_assignment(path))


class TorchAndDrawingTests(unittest.TestCase):
    def test_injected_torch_model_needs_no_framework_import(self):
        model = types.SimpleNamespace(names=dict(enumerate(NAMES)), calls=[])

        def predict(image, **options):
            model.calls.append(options)
            box = types.SimpleNamespace(cls=[3], conf=[0.9], xyxy=np.array([[1, 1, 3, 3]]))
            return [types.SimpleNamespace(boxes=[box], names=model.names,
                                          speed={'preprocess': 1.2, 'inference': 2.3, 'postprocess': 0.4})]

        model.predict = predict
        with patch.dict(sys.modules, {'ultralytics': None, 'onnxruntime': None}):
            detector = create_detector('torch', model=model, imgsz=8)
            result = detector.detect(np.zeros((4, 4, 3), np.uint8))
            detector.warmup()
        self.assertEqual(result.detections[0]['class_id'], 3)
        self.assertEqual(result.timings['execution_ms'], 2.3)
        self.assertEqual(model.calls[0], {'imgsz': 8, 'conf': 0.25, 'device': '0', 'rect': True, 'verbose': False})

    def test_torch_invalid_classes_rejected(self):
        with self.assertRaises(ValueError):
            TorchDetector(model=types.SimpleNamespace(names={0: 'person'}))

    def test_torch_refreshes_valid_names_and_hashes_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'weights.pt'
            checkpoint.write_bytes(b'checkpoint')
            model = types.SimpleNamespace(names=dict(enumerate(NAMES[:3])))
            model.predict = lambda *args, **kwargs: [types.SimpleNamespace(boxes=None, names=model.names)]
            detector = TorchDetector(model=model, model_path=checkpoint, imgsz=8)
            model.names[3] = NAMES[3]
            detector.detect(np.zeros((8, 8, 3), np.uint8))
            self.assertEqual(detector.names[3], 'mudstone_slab')
            self.assertEqual(detector.metadata['class_names'], NAMES)
            digest = hashlib.sha256(b'checkpoint').hexdigest()
            self.assertEqual(detector.metadata['model_sha256'], digest)
            self.assertEqual(detector.metadata['source_checkpoint_sha256'], digest)

    def test_shared_drawing_preserves_original_and_handles_no_boxes(self):
        image = np.zeros((80, 160, 3), np.uint8)
        empty = draw_detections(image, [])
        self.assertFalse(np.shares_memory(image, empty))
        annotated = draw_detections(image, [{'class_id': 3, 'class_name': 'mudstone_slab',
                                            'confidence': 0.9, 'xyxy': [10, 30, 80, 60]}])
        self.assertFalse(image.any())
        self.assertTrue(annotated.any())


if __name__ == '__main__':
    unittest.main()
