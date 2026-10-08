"""Orchestrate a bounded dataset run; one isolated Gazebo process group per scene."""
import argparse
import json
import os
import signal
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

from rover_training.geometry import CLASS_NAMES
from .common import read_config, write_json
from .generate import generate_bundle


def stop_group(process):
    if process.poll() is None:
        # Let ROS launch stop its own children before closing the X display.
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL)
            process.wait(timeout=5)


@contextmanager
def simulator_process(cmd,log):
    """Headless Gazebo (EGL rendering) kept off every other simulator's network."""
    env = os.environ.copy()
    env.setdefault('LIBGL_ALWAYS_SOFTWARE','1')
    # Separate Gazebo transport partition and container-local ROS discovery, so a
    # simulator in another container cannot inject its clock or robot description.
    env['GZ_PARTITION'] = f'mars_capture_{os.getpid()}_{time.monotonic_ns()}'
    env.setdefault('ROS_AUTOMATIC_DISCOVERY_RANGE','LOCALHOST')
    process = None
    try:
        process = subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        yield process
    finally:
        if process is not None:
            stop_group(process)


def _remove_scene_files(output, ident):
    for folder in ('metadata', 'images', 'labels', 'previews', 'depth', 'diagnostics'):
        for path in (output/folder).glob(f'**/{ident}_*'):
            if path.is_file():
                path.unlink()


def capture_bundle(bundle,output,resume=False,scene_ids=None):
    bundle,output = Path(bundle).resolve(),Path(output).resolve()
    plan = json.loads((bundle/'plan.json').read_text())
    output.mkdir(parents=True,exist_ok=True)
    stored = output/'plan.json'
    if stored.exists() and json.loads(stored.read_text()) != plan:
        raise ValueError('Dataset belongs to a different scene plan; use another output directory')
    write_json(stored,plan)
    write_json(output/'splits.json',json.loads((bundle/'splits.json').read_text()))
    (output/'logs').mkdir(exist_ok=True)
    (output/'scenes').mkdir(exist_ok=True)
    names = '\n'.join(f'  {index}: {name}' for index, name in enumerate(CLASS_NAMES))
    (output/'data.yaml').write_text(
        f'path: {output.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n')
    (output/'classes.txt').write_text('\n'.join(CLASS_NAMES) + '\n')
    for split in ('train','val','test'):
        (output/'images'/split).mkdir(parents=True,exist_ok=True)
        (output/'labels'/split).mkdir(parents=True,exist_ok=True)
    begin = time.monotonic()
    completed = []
    for entry in plan['scenes']:
        ident = entry['id']
        if scene_ids and ident not in scene_ids:
            continue
        scene_path = bundle/'scenes'/ident/'scene.json'
        scene = json.loads(scene_path.read_text())
        write_json(output/'scenes'/f'{ident}.json',scene)
        report_path = output/'reports'/f'{ident}.json'
        if report_path.exists():
            report = json.loads(report_path.read_text())
            if resume and report['scene_sha256']==entry['content_sha256'] and report['frames']==entry['views']:
                if not all((output/p).is_file() for p in report['records']):
                    raise ValueError(f'Incomplete metadata for {ident}')
                print(f'Resuming: {ident} already complete',flush=True)
                continue
            raise FileExistsError(f'{ident} already captured; pass --resume or choose a fresh output')
        partial = list((output/'metadata').glob(f'**/{ident}_*.json'))
        if partial and not resume:
            raise ValueError(f'{ident} has a partial capture; rerun with --resume to replace it')
        cmd = ['ros2','launch','mars_training_environment','mars_world.launch.py',
               f'scene:={scene_path}',f'output:={output}','capture:=true','gui:=false']
        # A stalled camera or slow software rendering can still fail a scene.
        # A fresh Gazebo process usually gets the whole scene.
        report = None
        start = time.monotonic()
        for attempt in range(1, 4):
            _remove_scene_files(output, ident)
            print(f'Capturing {ident} ({entry["split"]}, {entry["views"]} frames), attempt {attempt}', flush=True)
            log_path = output/'logs'/f'{ident}.log'
            try:
                with log_path.open('w') as log, simulator_process(cmd, log) as process:
                    code = process.wait(timeout=scene['config']['capture']['timeout_wall_sec']*(entry['views']+2))
            except subprocess.TimeoutExpired:
                code = 'timeout'
            if not report_path.exists():
                print(f'{ident} attempt {attempt} failed with exit {code}', flush=True)
                continue
            report = json.loads(report_path.read_text())
            if report['frames']==entry['views'] and report['scene_sha256']==entry['content_sha256']:
                break
            print(f'{ident} attempt {attempt} wrote an incomplete report', flush=True)
            report_path.unlink()
            report = None
        if report is None:
            raise RuntimeError(f'{ident} failed after 3 attempts; inspect {output/"logs"/(ident+".log")}')
        report['launch_wall_seconds'] = time.monotonic()-start
        write_json(report_path, report)
        completed.append(ident)
        print(f'Finished {ident} in {report["launch_wall_seconds"]:.1f}s', flush=True)
    from .validate import validate_dataset
    result = validate_dataset(output,bundle,require_complete=scene_ids is None)
    result['this_run_wall_seconds'] = time.monotonic()-begin
    result['this_run_scenes'] = completed
    write_json(output/'validation.json',result)
    print(json.dumps(result,indent=2),flush=True)
    if result['errors']:
        raise RuntimeError('Dataset validation failed')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--generate',action='store_true',help='Generate configured scenes first')
    parser.add_argument('--scene',action='append',help='Capture just these scene IDs (smoke test)')
    parser.add_argument('--asset-cache',type=Path,default=None,help='Prepared asset cache, used with --generate')
    args = parser.parse_args(argv)
    if args.generate:
        from .asset_cache import default_cache
        generate_bundle(args.bundle,read_config(args.config),cache=args.asset_cache or default_cache())
    capture_bundle(args.bundle,args.output,args.resume,args.scene)


if __name__ == '__main__':
    main()
