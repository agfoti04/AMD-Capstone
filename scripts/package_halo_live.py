"""Build a source-only Halo patch, including the complete new ROS package.

Run with --verify-build to build rover_perception from the extracted archive,
not from the developer checkout. No ROS runtime, dataset or weights are needed.
"""
import argparse
import ast
import configparser
import io
import subprocess
import sys
import tarfile
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


EXISTING_PACKAGE_FILES = (
    'README.md', 'scripts/HALO_LIVE_VISION.md',
    'scripts/halo_live_sim.sh', 'scripts/halo_live_vision.sh',
    'scripts/package_halo_live.py',
    'src/mars_training_environment/package.xml',
    'src/mars_training_environment/config/live_eval.json',
    'src/mars_training_environment/launch/mars_live.launch.py',
)
PERCEPTION_REQUIRED = (
    'setup.py', 'setup.cfg', 'package.xml', 'resource/rover_perception',
    'rover_perception/__init__.py', 'rover_perception/detector_node.py',
    'rover_perception/live_core.py', 'rover_perception/live_bridge.py',
    'rover_perception/live_dashboard.py', 'rover_perception/live_dashboard.html',
    'rover_perception/drive_mode.py', 'rover_perception/gazebo_drive.py',
    'launch/detector.launch.py', 'config/sim.yaml', 'config/oak.yaml',
)


def validate_package(root):
    package = root / 'src/rover_perception'
    missing = [name for name in PERCEPTION_REQUIRED if not (package / name).is_file()]
    if missing:
        raise ValueError(f'Incomplete rover_perception package: {missing}')
    config = configparser.ConfigParser(interpolation=None)
    config.read(package / 'setup.cfg')
    if (config.get('develop', 'script_dir', fallback='') != '$base/lib/rover_perception'
            or config.get('install', 'install_scripts', fallback='') != '$base/lib/rover_perception'):
        raise ValueError('setup.cfg must install executables in the ROS package lib directory')


def collect_sources(root):
    validate_package(root)
    files = set(EXISTING_PACKAGE_FILES)
    package = root / 'src/rover_perception'
    # Transfer every source/config/resource file in this newly introduced package.
    # Build products in a developer checkout must never substitute for source files.
    for source in package.rglob('*'):
        relative = source.relative_to(package)
        if any(part.startswith('.') or part in ('build', 'dist', '__pycache__')
               or part.endswith('.egg-info') for part in relative.parts):
            continue
        if source.is_file():
            files.add(source.relative_to(root).as_posix())
    for name in files:
        source = root / name
        if not source.is_file():
            raise ValueError(f'Missing deployment source: {name}')
        if source.suffix == '.py':
            ast.parse(source.read_text(encoding='utf-8'), filename=name)
        elif source.suffix == '.xml':
            ET.fromstring(source.read_bytes())
    return sorted(files)


def normalized_bytes(path):
    # Git on Windows can use CRLF; deployed Bash/Python sources use LF.
    return path.read_bytes().replace(b'\r\n', b'\n')


def make_patch(root, output):
    files = collect_sources(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, 'w:gz') as archive:
        for name in files:
            data = normalized_bytes(root / name)
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o755 if name.endswith('.sh') else 0o644
            archive.addfile(info, io.BytesIO(data))
    with tarfile.open(output, 'r:gz') as archive:
        assert archive.getnames() == files
        for name in files:
            assert archive.extractfile(name).read() == normalized_bytes(root / name), name
    return files


def verify_extracted_build(archive_path):
    with tempfile.TemporaryDirectory(prefix='halo-live-package-') as folder:
        extracted = Path(folder)
        with tarfile.open(archive_path, 'r:gz') as archive:
            archive.extractall(extracted, filter='data')
        validate_package(extracted)
        package = extracted / 'src/rover_perception'
        dist = extracted / 'verified-wheel'
        command = [sys.executable, 'setup.py', 'bdist_wheel', '--dist-dir', str(dist)]
        result = subprocess.run(command, cwd=package, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError('Extracted package build failed:\n' + result.stdout + result.stderr)
        wheels = list(dist.glob('*.whl'))
        assert len(wheels) == 1
        with zipfile.ZipFile(wheels[0]) as wheel:
            names = set(wheel.namelist())
            for name in ('__init__.py', 'detector_node.py', 'live_core.py', 'live_bridge.py',
                         'live_dashboard.py', 'live_dashboard.html', 'drive_mode.py', 'gazebo_drive.py'):
                assert 'rover_perception/' + name in names, name
            assert any(name.endswith('/share/ament_index/resource_index/packages/rover_perception')
                       for name in names), 'ROS package resource marker was not installed'
            entry_file = next(name for name in names if name.endswith('.dist-info/entry_points.txt'))
            entries = wheel.read(entry_file).decode()
            assert 'live_bridge = rover_perception.live_bridge:main' in entries
            assert 'rock_detector = rover_perception.detector_node:main' in entries
            assert 'gazebo_drive = rover_perception.gazebo_drive:main' in entries


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=root.parent / 'halo-gazebo-controls-20261013.tar.gz')
    parser.add_argument('--verify-build', action='store_true')
    args = parser.parse_args(argv)
    files = make_patch(root, args.output)
    if args.verify_build:
        verify_extracted_build(args.output)
    print(f'Verified {len(files)} source files: {args.output} ({args.output.stat().st_size} bytes)')
    if args.verify_build:
        print('Build from the extracted deployment archive passed; namespace, marker, executable and UI are present.')


if __name__ == '__main__':
    main()
