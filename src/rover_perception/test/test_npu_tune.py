import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from rover_perception.npu_tune import create_variant
from rover_perception.npu_tools import model_identity


class VariantTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        (self.source / 'model.onnx').write_bytes(b'immutable model')
        (self.source / 'calibration.json').write_text('{}')
        (self.source / 'fp32_parity.json').write_text('{"passed": true}')
        self.manifest = {
            'schema_version': 1, 'opset': 17, 'input_size': [960, 960],
            'class_names': ['rock', 'pale_soil_patch', 'yellow_crystal_rock', 'mudstone_slab'],
            'output_format': 'yolo_raw', 'precision': 'xint8',
            'preprocessing': {'color': 'rgb', 'scale': 1 / 255, 'layout': 'NCHW', 'padding': 114},
            'model_file': 'model.onnx', 'model_sha256': hashlib.sha256(b'immutable model').hexdigest(),
            'source_checkpoint_sha256': 'a' * 64, 'calibration_manifest': 'calibration.json',
            'parity_report': 'fp32_parity.json', 'provider_options': {'target': 'X2'},
        }
        self.path = self.source / 'manifest.json'
        self.path.write_text(json.dumps(self.manifest))

    def test_variant_preserves_artifacts_and_original_manifest(self):
        before = self.path.read_bytes()
        result = create_variant(self.path, self.root / 'variant', 1, False, 3)
        tuned = json.loads(result.read_text())
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(tuned['model_sha256'], self.manifest['model_sha256'])
        self.assertEqual(tuned['runtime_options'], {'intra_op_num_threads': 1, 'allow_spinning': False})
        self.assertEqual(tuned['provider_options'], {'target': 'X2', 'opt_level': 3})
        self.assertEqual((result.parent / 'calibration.json').read_bytes(), b'{}')
        self.assertEqual(tuned['optimization_parent']['sha256'], hashlib.sha256(before).hexdigest())
        self.assertFalse((result.parent / 'compile_report.json').exists())
        with self.assertRaises(FileExistsError):
            create_variant(self.path, result.parent)

    def test_missing_artifact_or_invalid_settings_leave_no_candidate(self):
        out = self.root / 'bad'
        with self.assertRaises(ValueError):
            create_variant(self.path, out, threads=-1)
        self.assertFalse(out.exists())
        (self.source / 'calibration.json').unlink()
        with self.assertRaises(FileNotFoundError):
            create_variant(self.path, out)
        self.assertFalse(out.exists())

    def test_benchmarks_cannot_be_joined_across_thread_settings(self):
        base = {'backend': 'npu', 'model_sha256': 'a' * 64}
        self.assertNotEqual(model_identity(base), model_identity(dict(base, runtime_options={
            'intra_op_num_threads': 1, 'allow_spinning': False})))

    def test_integer_compiler_tuning_preserves_float_scale_precision(self):
        self.manifest['precision'] = 'a8w8'
        self.manifest['quantization'] = {'config': 'A8W8', 'per_channel': True}
        self.path.write_text(json.dumps(self.manifest))
        result = create_variant(self.path, self.root / 'float_scales', 1, False, 0)
        tuned = json.loads(result.read_text())
        self.assertEqual(tuned['precision'], 'a8w8')
        self.assertTrue(tuned['quantization']['per_channel'])


if __name__ == '__main__':
    unittest.main()
