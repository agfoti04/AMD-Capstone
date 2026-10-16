"""Ground-truth matching, experiment provenance and hardware evidence gates."""
import copy
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from rover_perception import npu_tools as tools


def row(class_id=0, confidence=0.9, box=(10, 10, 30, 30)):
    return {'class_id': class_id, 'class_name': ['rock', 'slab'][class_id],
            'confidence': confidence, 'xyxy': list(box)}


def target(class_id=0, box=(10, 10, 30, 30)):
    return {'class_id': class_id, 'xyxy': list(box)}


def metadata(backend='npu'):
    return {'backend': backend, 'model': '/artifact/model.onnx' if backend == 'npu' else '/weights/best.pt',
            'precision': 'bf16' if backend == 'npu' else 'fp32', 'input_size': [960, 960],
            'cache_key': 'npu-cache' if backend == 'npu' else None,
            'runtime_fingerprint': 'runtime', 'proof_state': 'npu_executed' if backend == 'npu' else 'unverified',
            'compile_evidence': {'verified': True, 'sha256': 'compile-report'}}


def evaluation(backend='npu', ap=0.8, recall=0.9):
    return {'kind': 'detector_evaluation', 'detector': metadata(backend),
            'prediction_confidence': 0.001,
            'dataset': {'split': 'val', 'input_sha256': 'images-labels', 'class_names': ['rock', 'slab']},
            'metrics': {'map50_95': ap, 'fixed_confidence': 0.25, 'fixed_iou': 0.5,
                        'classes': {name: {'recall_at_025': recall} for name in ['rock', 'slab']}}}


def benchmark(backend='npu', fps=25):
    return {'kind': 'detector_benchmark', 'detector': metadata(backend), 'input_sha256': 'recording',
            'runs': [{'seconds': 60.01, 'frames': 1500, 'processed_fps': fps} for _ in range(3)],
            'total_latency_ms': {'median': 40}}


class MatchingTests(unittest.TestCase):
    def test_sdk_timestamped_multiline_operator_table_counts_operators_not_subgraphs(self):
        log = ('[Vitis AI EP] No. of Operators :\n'
               '2026-10-14 [I:onnxruntime] NPU 398\n'
               '2026-10-14 [I:onnxruntime] VITIS_EP_CPU 2\n'
               '[Vitis AI EP] No. of Subgraphs :\n'
               '2026-10-14 [I:onnxruntime] NPU 1\n')
        self.assertEqual(tools.npu_operator_count(log), 398)
        self.assertEqual(tools.npu_operator_count('[Vitis AI EP] No. of Subgraphs :\n NPU 1'), 0)

    def test_duplicate_detection_cannot_match_the_same_object_twice(self):
        predictions = [row(confidence=0.9), row(confidence=0.8)]
        correct = tools.match_predictions(predictions, [target()], 0.5)
        self.assertEqual(int(correct.sum()), 1)

    def test_wrong_class_and_wrong_location_are_false_positive(self):
        correct = tools.match_predictions([row(class_id=1), row(box=(50, 50, 80, 80))], [target()], 0.5)
        self.assertFalse(correct.any())

    def test_empty_predictions_keep_ground_truth_denominator(self):
        result = tools.detection_metrics([{'predictions': [], 'targets': [target()]}], ['rock', 'slab'])
        self.assertEqual(result['classes']['rock']['recall_at_025'], 0)
        self.assertEqual(result['map50_95'], 0)
        self.assertIsNone(result['classes']['slab']['recall_at_025'])

    def test_ap_uses_low_confidence_but_fixed_recall_does_not(self):
        result = tools.detection_metrics([{'predictions': [row(confidence=0.1)], 'targets': [target()]}], ['rock'])
        self.assertGreater(result['map50_95'], 0.99)
        self.assertEqual(result['classes']['rock']['recall_at_025'], 0)

    def test_iou_sweep_penalizes_loose_box(self):
        result = tools.detection_metrics([{'predictions': [row(box=(8, 8, 32, 32))],
                                           'targets': [target()]}], ['rock'])
        self.assertGreater(result['map50'], 0.99)
        self.assertLess(result['map50_95'], 0.5)
        self.assertEqual(result['classes']['rock']['recall_at_025'], 1)

    def test_background_false_positive_reduces_ap(self):
        frames = [{'predictions': [row(confidence=0.99, box=(70, 70, 90, 90)), row(confidence=0.9)],
                   'targets': [target()]}]
        result = tools.detection_metrics(frames, ['rock'])
        self.assertLess(result['map50'], 0.6)
        self.assertEqual(result['classes']['rock']['precision_at_025'], 0.5)


class ComparisonTests(unittest.TestCase):
    def compare(self, candidate=None, replay=None):
        return tools.compare_candidates(evaluation('torch'), [candidate or evaluation()],
                                        [benchmark('torch'), replay or benchmark()])

    def test_qualifies_only_with_live_and_test_still_pending(self):
        result = self.compare()
        self.assertEqual(result['status'], 'qualified_pending_live_and_test')
        self.assertIsNotNone(result['selected_detector'])

    def test_cpu_fallback_cannot_qualify(self):
        candidate = evaluation()
        candidate['detector']['proof_state'] = 'assigned'
        result = self.compare(candidate)
        self.assertEqual(result['status'], 'retain_gpu')
        self.assertIn('NPU execution', result['candidates'][0]['reasons'][0])

    def test_partial_compute_offload_does_not_qualify(self):
        candidate = evaluation()
        candidate['detector']['compile_evidence']['verified'] = False
        self.assertEqual(self.compare(candidate)['status'], 'retain_gpu')

    def test_accuracy_drop_and_class_recall_gate(self):
        for candidate in (evaluation(ap=0.779), evaluation(recall=0.849)):
            result = self.compare(candidate)
            self.assertFalse(result['candidates'][0]['qualified'])

    def test_boundary_is_inclusive(self):
        result = self.compare(evaluation(ap=0.78, recall=0.85), benchmark(fps=20))
        self.assertTrue(result['candidates'][0]['qualified'])

    def test_short_run_or_low_fps_fails(self):
        replay = benchmark()
        replay['runs'][0]['seconds'] = 59
        self.assertEqual(self.compare(replay=replay)['status'], 'retain_gpu')
        self.assertEqual(self.compare(replay=benchmark(fps=19.9))['status'], 'retain_gpu')

    def test_test_split_cannot_select_candidates(self):
        candidate = evaluation()
        candidate['dataset']['split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'validation'):
            self.compare(candidate)

    def test_mismatched_labels_or_recording_rejected(self):
        candidate = evaluation()
        candidate['dataset']['input_sha256'] = 'changed-label'
        with self.assertRaisesRegex(ValueError, 'datasets or labels differ'):
            self.compare(candidate)
        replay = benchmark()
        replay['input_sha256'] = 'different-camera-run'
        with self.assertRaisesRegex(ValueError, 'same recorded'):
            self.compare(replay=replay)

    def test_benchmark_from_different_precision_cannot_join(self):
        replay = benchmark()
        replay['detector']['precision'] = 'xint8'
        result = self.compare(replay=replay)
        self.assertEqual(result['status'], 'retain_gpu')

    def test_select_highest_ap_then_latency(self):
        first, second = evaluation(ap=0.8), evaluation(ap=0.81)
        second['detector']['model'] = '/artifact/other.onnx'
        second_benchmark = benchmark()
        second_benchmark['detector'] = copy.deepcopy(second['detector'])
        result = tools.compare_candidates(evaluation('torch'), [first, second],
                                         [benchmark('torch'), benchmark(), second_benchmark])
        self.assertEqual(result['selected_detector']['model'], '/artifact/other.onnx')


class ReferenceTests(unittest.TestCase):
    def test_compile_proof_is_bound_to_manifest_model_cache_and_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / 'manifest.json'
            manifest.write_text('{}')
            original = dict(metadata(), manifest=str(manifest), model_sha256='a' * 64,
                            computation_offload_verified=False)
            evidence = {'verified': True, 'candidate_inference_completed': True,
                        'compute_offload': {'verified': True, 'allowed_cpu_decoding_nodes': ['fixed_dfl']},
                        'manifest_sha256': tools.sha256(manifest), 'model_sha256': original['model_sha256'],
                        'backend': {'cache_key': original['cache_key'],
                                    'runtime_fingerprint': original['runtime_fingerprint']}}
            (manifest.parent / 'compile_report.json').write_text(json.dumps(evidence))
            result = tools.detector_metadata(types.SimpleNamespace(metadata=original))
            self.assertTrue(result['compile_evidence']['verified'])
            self.assertTrue(result['computation_offload_verified'])
            self.assertEqual(result['compute_offload']['allowed_cpu_decoding_nodes'], ['fixed_dfl'])
            for field in ('model_sha256', 'cache_key', 'runtime_fingerprint', 'proof_state'):
                with self.subTest(field=field):
                    changed = dict(original, **{field: 'changed'})
                    result = tools.detector_metadata(types.SimpleNamespace(metadata=changed))
                    self.assertFalse(result['compile_evidence']['verified'])
            manifest.write_text('{"changed": true}')
            self.assertFalse(tools.detector_metadata(types.SimpleNamespace(metadata=original))['compile_evidence']['verified'])

    def test_operator_count_requires_assignment_numeric_evidence(self):
        self.assertEqual(tools.npu_operator_count('VitisAIExecutionProvider is available'), 0)
        self.assertEqual(tools.npu_operator_count('[Vitis AI EP] No. of Operators : NPU 398 VITIS_EP_CPU 2'), 398)

    def test_reference_compile_only_and_cpu_only_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / 'quicktest.py'
            script.write_text('print("example")')
            for output in ('Operators : NPU 398; Compilation successful', 'Test Finished'):
                execution = {'command': ['python', str(script)], 'returncode': 0, 'stdout': output, 'stderr': ''}
                xrt = {'returncode': 0, 'stdout': 'NPU Strix aie2p', 'stderr': ''}
                with patch.object(tools, 'command', side_effect=[xrt, execution, xrt]):
                    result = tools.verify_reference(script, Path(directory) / 'evidence.json')
                self.assertFalse(result['verified'])

    def test_reference_inference_npu_assignment_and_device_required(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / 'quicktest.py'
            script.write_text('print("example")')
            execution = {'command': ['python', str(script)], 'returncode': 0,
                         'stdout': 'Operators : NPU 398 VITIS_EP_CPU 2\nTest Finished', 'stderr': ''}
            xrt = {'returncode': 0, 'stdout': 'NPU Strix aie2p', 'stderr': ''}
            with patch.object(tools, 'command', side_effect=[xrt, execution, xrt]):
                result = tools.verify_reference(script, Path(directory) / 'evidence.json')
            self.assertTrue(result['verified'])
            self.assertTrue(result['verification']['inference_completed'])
            self.assertEqual(len(result['reference_script_sha256']), 64)

    def test_kernel_version_prerequisite(self):
        for version in ('6.10.1-generic', '6.14.0-28-generic', '7.0.0'):
            self.assertTrue(tools.kernel_at_least(version))
        for version in ('6.8.0', '5.15.0', 'invalid'):
            self.assertFalse(tools.kernel_at_least(version))


class LiveCounterTests(unittest.TestCase):
    def samples(self):
        return [{'offset_seconds': index / 4, 'status': {
            'processed_frames': int(index / 4 * 30), 'processing_session_id': 'session-one',
            'frame_age_seconds': .05, 'loop_fps': 30, 'ready': True}}
            for index in range(1201)]

    def test_four_hz_polling_measures_thirty_processed_fps(self):
        result = tools.summarize_live_samples(self.samples(), [], 300)
        self.assertEqual(result['processed_fps_observed'], 30)
        self.assertEqual(len(result['minute_windows']), 5)
        self.assertTrue(result['telemetry_gate_passed'])

    def test_missing_restarted_reset_or_stalled_counters_fail(self):
        for change in ('missing', 'restart', 'reset', 'stall'):
            samples = self.samples()
            if change == 'missing':
                samples[20]['status'].pop('processed_frames')
            elif change == 'restart':
                samples[20]['status']['processing_session_id'] = 'another-session'
            elif change == 'reset':
                samples[20]['status']['processed_frames'] = 0
            else:
                for item in samples[20:30]:
                    item['status']['processed_frames'] = samples[19]['status']['processed_frames']
            with self.subTest(change=change):
                result = tools.summarize_live_samples(samples, [], 300)
                self.assertFalse(result['telemetry_gate_passed'])
                self.assertTrue(result['telemetry_failures'])

    def test_short_slow_minute_camera_growth_errors_and_unready_fail(self):
        samples = self.samples()
        self.assertFalse(tools.summarize_live_samples(samples[:100], [], 25)['telemetry_gate_passed'])
        self.assertFalse(tools.summarize_live_samples(samples, [{'error': 'offline'}], 300)['telemetry_gate_passed'])
        samples[10]['status']['ready'] = False
        self.assertFalse(tools.summarize_live_samples(samples, [], 300)['telemetry_gate_passed'])
        samples = self.samples()
        for item in samples[900:]:
            item['status']['frame_age_seconds'] = 1
        self.assertFalse(tools.summarize_live_samples(samples, [], 300)['telemetry_gate_passed'])
        samples = self.samples()
        for item in samples:
            t = item['offset_seconds']
            item['status']['processed_frames'] = int(10 * min(t, 60) + 40 * max(0, t - 60))
        result = tools.summarize_live_samples(samples, [], 300)
        self.assertGreater(result['processed_fps_observed'], 20)
        self.assertFalse(result['telemetry_gate_passed'])


class DatasetTests(unittest.TestCase):
    def create_export(self, directory):
        root = Path(directory)
        images = root / 'images' / 'val'
        labels = root / 'labels' / 'val'
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        cv2.imwrite(str(images / 'frame.png'), np.zeros((40, 60, 3), dtype=np.uint8))
        (labels / 'frame.txt').write_text('0 0.5 0.5 0.5 0.5\n')
        data = root / 'data.yaml'
        data.write_text('path: .\nval: images/val\nnames: [rock, slab]\n')
        return data

    def test_export_resolution_and_label_hash_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            data = self.create_export(directory)
            samples, first = tools.load_dataset(data=data)
            self.assertEqual(len(samples), 1)
            self.assertEqual(first['class_names'], ['rock', 'slab'])
            label = samples[0]['label']
            label.write_text('1 0.5 0.5 0.5 0.5\n')
            _, second = tools.load_dataset(data=data)
            self.assertNotEqual(first['input_sha256'], second['input_sha256'])

    def test_missing_label_never_assumed_background(self):
        with tempfile.TemporaryDirectory() as directory:
            data = self.create_export(directory)
            (Path(directory) / 'labels/val/frame.txt').unlink()
            with self.assertRaisesRegex(ValueError, 'Missing label'):
                tools.load_dataset(data=data)

    def test_invalid_class_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            data = self.create_export(directory)
            (Path(directory) / 'labels/val/frame.txt').write_text('2 0.5 0.5 0.5 0.5')
            with self.assertRaisesRegex(ValueError, 'Invalid YOLO class'):
                tools.load_dataset(data=data)

    def test_input_frame_content_changes_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / 'frame.png'
            cv2.imwrite(str(image), np.zeros((8, 8, 3), dtype=np.uint8))
            _, first = tools.input_images(directory)
            cv2.imwrite(str(image), np.full((8, 8, 3), 20, dtype=np.uint8))
            _, second = tools.input_images(directory)
            self.assertNotEqual(first, second)


if __name__ == '__main__':
    unittest.main()
