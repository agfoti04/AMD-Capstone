"""Acceptance binds a frozen model to accuracy, live data and human stop checks."""
import copy
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from rover_perception import npu_acceptance as acceptance
from rover_perception.npu_tools import sha256, write_json


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        model = self.root / 'model.onnx'
        model.write_bytes(b'frozen model')
        manifest = self.root / 'manifest.json'
        write_json(manifest, {'model_file': model.name, 'model_sha256': sha256(model)})
        self.npu = {'backend': 'npu', 'model': str(model), 'model_sha256': sha256(model),
                    'manifest': str(manifest), 'precision': 'a16w8', 'input_size': [576, 960],
                    'runtime_fingerprint': {'kernel': 'test'}, 'cache_key': 'test-cache',
                    'proof_state': 'npu_executed'}
        compile_path = self.root / 'compile_report.json'
        write_json(compile_path, {'verified': True, 'candidate_inference_completed': True,
                                 'compute_offload': {'verified': True},
                                 'manifest_sha256': sha256(manifest), 'model_sha256': sha256(model),
                                 'backend': copy.deepcopy(self.npu)})
        self.npu['compile_evidence'] = {'verified': True, 'sha256': sha256(compile_path)}
        self.gpu = dict(self.npu, backend='torch', model='/baseline/best.pt')
        names = ['rock', 'pale_soil_patch', 'yellow_crystal_rock', 'mudstone_slab']
        def evaluation(detector):
            return {'kind': 'detector_evaluation', 'detector': detector, 'frames': [{}] * 100,
                    'prediction_confidence': .001,
                    'dataset': {'split': 'test', 'input_sha256': 'same-data', 'class_names': names},
                    'metrics': {'map50_95': .8, 'fixed_confidence': .25, 'fixed_iou': .5,
                                'classes': {n: {'recall_at_025': .9} for n in names}}}
        self.reports = {
            'comparison': {'status': 'qualified_pending_live_and_test', 'selected_detector': self.npu,
                           'baseline_detector': self.gpu,
                           'candidates': [{'qualified': True, 'detector': self.npu}]},
            'gpu_test': evaluation(self.gpu), 'npu_test': evaluation(self.npu),
            'live': {'seconds': 300, 'errors': [], 'telemetry_gate_passed': True,
                     'samples': [{'offset_seconds': i / 4, 'status': {
                         'processed_frames': int(i / 4 * 30), 'processing_session_id': 'one',
                         'frame_age_seconds': .05, 'ready': True, 'detector': self.npu}}
                         for i in range(1201)]},
            'controls': {'kind': 'manual_controls_review', 'reviewed_by': 'human',
                         'processing_session_id': 'one',
                         'checks': {n: True for n in acceptance.CONTROL_CHECKS}}}

    def promote(self):
        paths = {'manifest': self.root / 'manifest.json'}
        for name, report in self.reports.items():
            paths[name] = self.root / (name + '.json')
            write_json(paths[name], report)
        return acceptance.promotion_record(**paths)

    def test_complete_acceptance_and_runtime_bound_default(self):
        record = self.promote()
        path = self.root / 'active.json'
        write_json(path, record)
        with patch.dict('sys.modules', {'onnxruntime': types.SimpleNamespace()}), \
                patch('rover_perception.detectors.runtime_fingerprint', return_value={'kernel': 'test'}):
            self.assertEqual(acceptance.active_manifest(path), self.root / 'manifest.json')
            self.assertEqual(acceptance.launch_arguments(['--scene', 'example'], path),
                             ['--backend', 'npu', '--model-manifest', str(self.root / 'manifest.json'), '--scene', 'example'])
            with patch('rover_perception.detectors.runtime_fingerprint', return_value={'kernel': 'new'}):
                with self.assertRaisesRegex(ValueError, 'runtime/kernel changed'):
                    acceptance.active_manifest(path)
            (self.root / 'model.onnx').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'artifact/evidence changed'):
                acceptance.active_manifest(path)

    def test_explicit_gpu_and_experimental_npu_do_not_load_active_record(self):
        missing = self.root / 'missing.json'
        self.assertEqual(acceptance.launch_arguments(['--model', 'best.pt'], missing),
                         ['--backend', 'torch', '--model', 'best.pt'])
        self.assertEqual(acceptance.launch_arguments(['--backend', 'npu', '--model-manifest', 'experiment.json'], missing),
                         ['--backend', 'npu', '--model-manifest', 'experiment.json'])
        self.assertEqual(acceptance.launch_arguments(['--help'], missing), ['--help'])

    def test_missing_manual_checks_and_wrong_live_model_fail(self):
        for failure in ('review', 'session', 'model', 'fps'):
            original = copy.deepcopy(self.reports)
            if failure == 'review':
                self.reports['controls']['checks']['disconnect_stop'] = False
            elif failure == 'session':
                self.reports['controls']['processing_session_id'] = 'another'
            elif failure == 'model':
                self.reports['live']['samples'][2]['status']['detector'] = dict(self.npu, backend='torch')
            else:
                self.reports['live']['samples'][2]['status']['processed_frames'] = 0
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                self.promote()
            self.reports = original

    def test_test_accuracy_incomplete_data_and_frozen_identity_fail(self):
        for failure in ('recall', 'ap', 'frames', 'data', 'identity'):
            original = copy.deepcopy(self.reports)
            candidate = self.reports['npu_test']
            if failure == 'recall':
                candidate['metrics']['classes']['mudstone_slab']['recall_at_025'] = .84
            elif failure == 'ap':
                candidate['metrics']['map50_95'] = .77
            elif failure == 'frames':
                candidate['frames'] = candidate['frames'][:-1]
            elif failure == 'data':
                candidate['dataset']['input_sha256'] = 'different'
            else:
                candidate['detector'] = dict(self.npu, precision='xint8')
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                self.promote()
            self.reports = original


if __name__ == '__main__':
    unittest.main()
