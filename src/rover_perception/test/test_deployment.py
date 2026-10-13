"""Catch partial-package transfers even when the developer checkout builds."""
import importlib.util
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('halo_packaging', ROOT / 'scripts/package_halo_live.py')
packaging = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packaging)


class DeploymentTests(unittest.TestCase):
    def test_archive_contains_complete_new_package(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / 'patch.tar.gz'
            packaging.make_patch(ROOT, archive)
            extracted = Path(folder) / 'fresh-halo-workspace'
            with tarfile.open(archive) as source:
                source.extractall(extracted, filter='data')
            packaging.validate_package(extracted)
            for name in packaging.PERCEPTION_REQUIRED:
                self.assertEqual((extracted / 'src/rover_perception' / name).read_bytes(),
                                 packaging.normalized_bytes(ROOT / 'src/rover_perception' / name))
            self.assertNotIn('build', [part for p in extracted.rglob('*') for part in p.relative_to(extracted).parts])

    def test_missing_bootstrap_files_fail_before_packaging(self):
        for missing in ('resource/rover_perception', 'setup.cfg', 'rover_perception/__init__.py'):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                shutil.copytree(ROOT / 'src/rover_perception', root / 'src/rover_perception',
                                ignore=shutil.ignore_patterns('build', '*.egg-info', '__pycache__'))
                (root / 'src/rover_perception' / missing).unlink()
                with self.assertRaisesRegex(ValueError, 'Incomplete rover_perception'):
                    packaging.validate_package(root)

    def test_wrong_executable_install_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            shutil.copytree(ROOT / 'src/rover_perception', root / 'src/rover_perception',
                            ignore=shutil.ignore_patterns('build', '*.egg-info', '__pycache__'))
            (root / 'src/rover_perception/setup.cfg').write_text(
                '[develop]\nscript_dir=$base/bin\n[install]\ninstall_scripts=$base/bin\n')
            with self.assertRaisesRegex(ValueError, 'ROS package lib directory'):
                packaging.validate_package(root)


if __name__ == '__main__':
    unittest.main()
