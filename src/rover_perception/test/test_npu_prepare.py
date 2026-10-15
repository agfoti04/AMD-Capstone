"""Preparation provenance and gate tests; no torch, ORT or NPU needed."""
import json
import socket
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from rover_perception import npu_prepare as prepare


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / 'data.yaml'
        document = {'path': str(self.root), 'train': 'images/train', 'val': 'images/val',
                    'test': 'images/test', 'names': prepare.CLASS_NAMES}
        self.data.write_text(yaml.safe_dump(document))
        self.audit = []
        for split, count in [('train', 3), ('val', 1), ('test', 1)]:
            folder = self.root / 'images' / split
            folder.mkdir(parents=True)
            for index in range(count):
                image = folder / f'{split}_{index}.png'
                image.write_bytes(f'image-{split}-{index}'.encode())
                self.audit.append({'split': split, 'source_image': f'images/{split}/{image.name}',
                                   'image_sha256': prepare.sha256(image)})
        self.write_audit()

    def write_audit(self):
        (self.root / 'review_export.json').write_text(json.dumps({'audit': self.audit}))

    def calibration(self):
        return prepare.create_calibration_manifest(self.data, expected_count=3)

    def test_calibration_is_train_only_sorted_and_reproducible(self):
        first = self.calibration()
        self.assertEqual(first, self.calibration())
        self.assertEqual(first['count'], 3)
        self.assertEqual([row['split'] for row in first['images']], ['train'] * 3)
        self.assertEqual([Path(row['path']).name for row in first['images']],
                         ['train_0.png', 'train_1.png', 'train_2.png'])

    def test_wrong_training_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Expected 300'):
            prepare.create_calibration_manifest(self.data)

    def test_missing_review_audit_is_rejected(self):
        (self.root / 'review_export.json').unlink()
        with self.assertRaisesRegex(ValueError, 'review_export'):
            self.calibration()

    def test_changed_image_is_rejected(self):
        (self.root / 'images/train/train_0.png').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'audit'):
            self.calibration()

    def test_held_out_image_content_cannot_calibrate(self):
        train = self.root / 'images/train/train_0.png'
        train.write_bytes((self.root / 'images/test/test_0.png').read_bytes())
        self.audit[0]['image_sha256'] = prepare.sha256(train)
        self.write_audit()
        with self.assertRaisesRegex(ValueError, 'validation/test'):
            self.calibration()

    def test_held_out_image_path_cannot_calibrate(self):
        document = yaml.safe_load(self.data.read_text())
        document['test'] = 'images/train'
        self.data.write_text(yaml.safe_dump(document))
        with self.assertRaisesRegex(ValueError, 'overlap'):
            self.calibration()

    def test_wrong_class_mapping_is_rejected(self):
        document = yaml.safe_load(self.data.read_text())
        document['names'][0] = 'slab'
        self.data.write_text(yaml.safe_dump(document))
        with self.assertRaisesRegex(ValueError, 'four-class'):
            self.calibration()

    def test_frozen_manifest_detects_tampering(self):
        manifest = self.calibration()
        path = self.root / 'calibration.json'
        path.write_text(json.dumps(manifest))
        self.assertEqual(prepare.validate_calibration(path), manifest)
        manifest['images'][0]['split'] = 'val'
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'checksum'):
            prepare.validate_calibration(path)

    def test_frozen_manifest_detects_review_changes(self):
        path = self.root / 'calibration.json'
        path.write_text(json.dumps(self.calibration()))
        self.audit[0]['note'] = 'updated review'
        self.write_audit()
        with self.assertRaisesRegex(ValueError, 'dataset changed'):
            prepare.validate_calibration(path)

    def test_manifest_relative_yaml_resolves_from_manifest(self):
        path = self.root / 'training.json'
        path.write_text(json.dumps({'data_yaml': 'data.yaml'}))
        self.assertEqual(prepare.resolve_dataset_manifest(path), (path.resolve(), self.data.resolve()))

    def test_existing_output_never_overwritten(self):
        marker = self.root / 'model.onnx'
        marker.write_bytes(b'original')
        with self.assertRaisesRegex(ValueError, 'already exists'):
            prepare._candidate_directory(self.root)
        self.assertEqual(marker.read_bytes(), b'original')


class HardwareGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'reference.json'
        self.evidence = {'kind': 'amd_npu_reference', 'verified': True, 'returncode': 0,
                         'host': {'hostname': socket.gethostname()},
                         'verification': {'inference_completed': True, 'operators_npu': 12}}

    def validate(self):
        self.path.write_text(json.dumps(self.evidence))
        return prepare.validate_hardware_evidence(self.path)

    def test_successful_reference_inference_is_accepted(self):
        self.assertEqual(self.validate(), self.evidence)

    def test_compile_only_evidence_is_rejected(self):
        self.evidence['verification']['inference_completed'] = False
        with self.assertRaisesRegex(ValueError, 'reference inference'):
            self.validate()

    def test_cpu_only_reference_is_rejected(self):
        self.evidence['verification']['operators_npu'] = 0
        with self.assertRaisesRegex(ValueError, 'reference inference'):
            self.validate()

    def test_evidence_from_another_host_is_rejected(self):
        self.evidence['host']['hostname'] = 'another-halo'
        with self.assertRaisesRegex(ValueError, 'this Halo'):
            self.validate()


class PostprocessingTests(unittest.TestCase):
    def graph(self):
        def node(name, op, inputs, output):
            return types.SimpleNamespace(name=name, op_type=op, input=inputs, output=[output])
        return types.SimpleNamespace(graph=types.SimpleNamespace(
            initializer=[types.SimpleNamespace(name='dfl_weight', dims=[1, 16, 1, 1]),
                         types.SimpleNamespace(name='head_weight', dims=[64, 64, 1, 1])],
            node=[node('backbone', 'Conv', ['images', 'head_weight'], 'features'),
                  node('box_prediction', 'Conv', ['features', 'head_weight'], 'box_logits'),
                  node('class_prediction', 'Conv', ['features', 'head_weight'], 'class_logits'),
                  node('dfl_softmax', 'Softmax', ['box_logits'], 'distribution'),
                  node('dfl_projection', 'Conv', ['distribution', 'dfl_weight'], 'decoded'),
                  node('class_sigmoid', 'Sigmoid', ['class_logits'], 'scores'),
                  node('output_concat', 'Concat', ['decoded', 'scores'], 'output')],
            output=[types.SimpleNamespace(name='output')]))

    def test_decoding_excluded_but_learned_head_remains_quantized(self):
        excluded = prepare.postprocessing_nodes(self.graph())
        self.assertEqual(excluded, ['class_sigmoid', 'dfl_projection', 'dfl_softmax', 'output_concat'])

    def test_unknown_graph_rejected_instead_of_quantizing_blindly(self):
        graph = self.graph()
        graph.graph.node = []
        with self.assertRaisesRegex(ValueError, 'boundary'):
            prepare.postprocessing_nodes(graph)

    def test_cpu_optimization_aliases_are_limited_to_known_decoding_outputs(self):
        graph = self.graph()
        names = prepare.runtime_decoding_names(graph, prepare.postprocessing_nodes(graph))
        self.assertIn('decoded_nchwc', names)
        self.assertNotIn('box_logits_nchwc', names)
        self.assertNotIn('features_nchwc', names)

    def test_unnamed_postprocessing_node_rejected(self):
        graph = self.graph()
        graph.graph.node[-1].name = ''
        with self.assertRaisesRegex(ValueError, 'Unnamed'):
            prepare.postprocessing_nodes(graph)


class ParityTests(unittest.TestCase):
    def row(self, score=0.9, x=0, class_id=3):
        return {'class_id': class_id, 'confidence': score, 'xyxy': [x, 0, 20, 20]}

    def test_parity_ignores_tied_output_order(self):
        left = [self.row(), self.row(x=10)]
        self.assertTrue(prepare._decoded_parity(left, list(reversed(left))))

    def test_parity_rejects_class_confidence_and_box_errors(self):
        for row in [self.row(class_id=0), self.row(score=0.8), self.row(x=5)]:
            with self.subTest(row=row):
                self.assertFalse(prepare._decoded_parity([self.row()], [row]))

    def test_empty_predictions_are_valid_but_missing_boxes_are_not(self):
        self.assertTrue(prepare._decoded_parity([], []))
        self.assertFalse(prepare._decoded_parity([self.row()], []))

    def test_export_head_is_selected_before_fusion_removes_other_branch(self):
        calls = []
        reference = types.SimpleNamespace(end2end=False)
        reference.eval = lambda: calls.append(('eval', reference.end2end))
        reference.fuse = lambda **kwargs: calls.append(('fuse', reference.end2end))
        model = types.SimpleNamespace(model=reference)
        prepare.configure_parity_reference(model, 'yolo_end2end')
        self.assertEqual(calls, [('eval', True), ('fuse', True)])
        calls.clear()
        prepare.configure_parity_reference(model, 'yolo_raw')
        self.assertEqual(calls, [('eval', False), ('fuse', False)])


class ComputeOffloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.profile = Path(self.temp.name) / 'runtime_profile.json'
        self.metadata = {'proof_state': 'npu_executed', 'profile_path': str(self.profile),
                         'operator_assignment': {'npu_nodes': 10, 'npu_ops': ['::Conv']}}

    def audit(self, op='Reshape', name='reshape', allowed=()):
        self.profile.write_text(json.dumps([{'name': name + '_kernel_time', 'cat': 'Node', 'dur': 10,
                                            'args': {'provider': 'CPUExecutionProvider', 'op_name': op}}]))
        return prepare.audit_compute_offload(self.metadata, allowed)

    def test_cpu_reshaping_and_decoding_allowed(self):
        self.assertTrue(self.audit()['verified'])
        result = self.audit('Conv', 'dfl_projection', ['dfl_projection'])
        self.assertTrue(result['verified'])
        self.assertEqual(result['allowed_cpu_decoding_nodes'][0]['name'], 'dfl_projection')

    def test_learned_head_or_backbone_on_cpu_rejected(self):
        result = self.audit('Conv', 'head_prediction')
        self.assertFalse(result['verified'])
        self.assertEqual(result['learned_cpu_nodes'][0]['name'], 'head_prediction')

    def test_provider_registration_without_compute_assignment_rejected(self):
        self.metadata['operator_assignment']['npu_ops'] = ['::Reshape']
        self.assertFalse(self.audit()['verified'])

    def test_incomplete_runtime_profile_cannot_hide_cpu_compute_assignment(self):
        self.metadata['operator_assignment']['cpu_ops'] = ['::Conv']
        result = self.audit()
        self.assertFalse(result['verified'])
        self.assertIn('complete runtime', result['reason'])


if __name__ == '__main__':
    unittest.main()
