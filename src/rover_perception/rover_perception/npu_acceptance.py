"""Promote only measured NPU deployments; preserve explicit GPU launch commands."""
import argparse
import json
import math
import os
from pathlib import Path
import sys

from .npu_tools import (model_identity, read_json, sha256, summarize_live_samples,
                        utc_now, write_json)


PROJECT = Path(__file__).resolve().parents[3]
ACTIVE = PROJECT / 'experiments/npu/active_npu.json'
CONTROL_CHECKS = ('held_wasd', 'simultaneous_steering', 'speed_sliders', 'release',
                  'space', 'focus_loss', 'disconnect_stop', 'expiry_under_0_5s',
                  'stale_camera_stop', 'backend_error_stop')


def source_identity():
    directory = Path(__file__).parent
    return {name: sha256(directory / name) for name in (
        'detectors.py', 'live_dashboard.py', 'live_web.py', 'live_core.py',
        'npu_tools.py', 'npu_acceptance.py')}


def accuracy_gates(baseline, candidate, split='test'):
    """Test reports confirm a frozen candidate; they never select/tune one."""
    for report in (baseline, candidate):
        if report.get('kind') != 'detector_evaluation' or report['dataset']['split'] != split:
            raise ValueError(f'Expected complete {split} evaluation reports')
        if len(report.get('frames', [])) != 100:
            raise ValueError(f'Expected all 100 {split} images')
        if report.get('prediction_confidence') != .001:
            raise ValueError('AP evaluation must use confidence 0.001')
        if report['metrics'].get('fixed_confidence') != .25 or report['metrics'].get('fixed_iou') != .5:
            raise ValueError('Recall must use confidence 0.25 and IoU 0.5')
    if (baseline['dataset']['input_sha256'] != candidate['dataset']['input_sha256']
            or baseline['dataset']['class_names'] != candidate['dataset']['class_names']):
        raise ValueError('Evaluation images, labels or classes differ')
    def drop(left, right):
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (left, right)):
            raise ValueError('Accuracy report has missing/non-finite metrics')
        return left - right
    ap_drop = drop(baseline['metrics']['map50_95'], candidate['metrics']['map50_95'])
    recalls = {name: drop(baseline['metrics']['classes'][name]['recall_at_025'],
                          candidate['metrics']['classes'][name]['recall_at_025'])
               for name in baseline['dataset']['class_names']}
    return {'passed': ap_drop <= .02 + 1e-9 and all(v <= .05 + 1e-9 for v in recalls.values()),
            'map50_95_drop': ap_drop, 'class_recall_drops': recalls}


def promotion_record(manifest, comparison, gpu_test, npu_test, live, controls):
    paths = {key: Path(value).resolve(strict=True) for key, value in dict(
        manifest=manifest, comparison=comparison, gpu_test=gpu_test,
        npu_test=npu_test, live=live, controls=controls).items()}
    reports = {key: read_json(path) for key, path in paths.items()}
    chosen = reports['comparison'].get('selected_detector')
    if reports['comparison'].get('status') != 'qualified_pending_live_and_test' or not chosen:
        raise ValueError('Validation/replay comparison did not qualify an NPU detector')
    if chosen.get('backend') != 'npu' or chosen.get('proof_state') != 'npu_executed':
        raise ValueError('Actual NPU execution is required')
    qualified = [r for r in reports['comparison'].get('candidates', [])
                 if r.get('qualified') and model_identity(r['detector']) == model_identity(chosen)]
    if len(qualified) != 1 or not chosen.get('compile_evidence', {}).get('verified'):
        raise ValueError('Missing qualified candidate or verified compute-offload proof')
    for key, expected in (('gpu_test', reports['comparison']['baseline_detector']), ('npu_test', chosen)):
        if model_identity(reports[key]['detector']) != model_identity(expected):
            raise ValueError('Test detector differs from the frozen validation selection')
    test_gate = accuracy_gates(reports['gpu_test'], reports['npu_test'])
    if not test_gate['passed']:
        raise ValueError('Held-out accuracy gate failed; retain GPU and do not tune on test results')
    observed = reports['live']
    checked = summarize_live_samples(observed.get('samples', []), observed.get('errors', []), observed['seconds'])
    if not observed.get('telemetry_gate_passed') or not checked['telemetry_gate_passed']:
        raise ValueError('Five-minute live telemetry gate failed')
    if any(model_identity(row['status'].get('detector', {})) != model_identity(chosen)
           for row in observed['samples']):
        raise ValueError('Live session did not use the frozen NPU detector')
    review = reports['controls']
    if (review.get('kind') != 'manual_controls_review' or not review.get('reviewed_by')
            or review.get('processing_session_id') != checked['processing_session_id']
            or any(review.get('checks', {}).get(name) is not True for name in CONTROL_CHECKS)):
        raise ValueError('Manual drive/control-stop review is incomplete or belongs to another session')
    compile_path = paths['manifest'].parent / 'compile_report.json'
    compile_report = read_json(compile_path)
    if (not compile_report.get('verified')
            or not compile_report.get('candidate_inference_completed')
            or not compile_report.get('compute_offload', {}).get('verified')
            or compile_report.get('manifest_sha256') != sha256(paths['manifest'])
            or compile_report.get('model_sha256') != reports['manifest']['model_sha256']
            or chosen['compile_evidence'].get('sha256') != sha256(compile_path)
            or compile_report.get('backend', {}).get('runtime_fingerprint') != chosen['runtime_fingerprint']
            or compile_report.get('backend', {}).get('cache_key') != chosen['cache_key']
            or Path(chosen.get('manifest', '')).resolve() != paths['manifest']):
        raise ValueError('Compilation proof does not match the selected manifest/model')
    model_path = (paths['manifest'].parent / reports['manifest']['model_file']).resolve(strict=True)
    if sha256(model_path) != reports['manifest']['model_sha256']:
        raise ValueError('Compiled model content changed')
    paths.update(compile_report=compile_path, model=model_path)
    return {'schema_version': 1, 'kind': 'accepted_npu_deployment', 'status': 'accepted',
            'created_utc': utc_now(), 'manifest': str(paths['manifest']),
            'runtime_fingerprint': chosen['runtime_fingerprint'], 'source_identity': source_identity(),
            'held_out_gate': test_gate,
            'artifacts': {key: {'path': str(path), 'sha256': sha256(path)} for key, path in paths.items()}}


def active_manifest(path=ACTIVE):
    from .detectors import runtime_fingerprint
    import onnxruntime as ort
    record = read_json(path)
    if record.get('kind') != 'accepted_npu_deployment' or record.get('status') != 'accepted':
        raise ValueError('No accepted NPU deployment is configured')
    if record.get('source_identity') != source_identity():
        raise ValueError('Detector/dashboard source changed; repeat acceptance before default NPU use')
    for item in record['artifacts'].values():
        if sha256(item['path']) != item['sha256']:
            raise ValueError('Accepted NPU artifact/evidence changed; repeat acceptance')
    if record.get('runtime_fingerprint') != runtime_fingerprint(ort):
        raise ValueError('NPU runtime/kernel changed; regenerate execution proof and acceptance')
    manifest = Path(record['manifest']).resolve(strict=True)
    if str(manifest) != record['artifacts']['manifest']['path']:
        raise ValueError('Accepted manifest path does not match its evidence')
    return manifest


def launch_arguments(argv, active=ACTIVE):
    if '-h' in argv or '--help' in argv:
        return argv
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--backend', choices=('torch', 'npu'))
    parser.add_argument('--model')
    parser.add_argument('--model-manifest')
    selected, remaining = parser.parse_known_args(argv)
    backend = selected.backend or ('torch' if selected.model else 'npu' if selected.model_manifest else None)
    if backend is None:
        selected.model_manifest = str(active_manifest(active))
        backend = 'npu'
    result = ['--backend', backend]
    if selected.model:
        result += ['--model', selected.model]
    if selected.model_manifest:
        result += ['--model-manifest', selected.model_manifest]
    return result + remaining


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'launch':
        try:
            arguments = launch_arguments(argv[1:])
        except (ValueError, OSError, KeyError, ImportError) as error:
            raise SystemExit(f'NPU default unavailable: {error}. Source halo_npu_env.sh and complete acceptance, '
                             'or explicitly launch --backend torch --model /path/to/best.pt.')
        os.execv(sys.executable, [sys.executable, '-m', 'rover_perception.live_dashboard', *arguments])
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    promote = sub.add_parser('promote')
    for name in ('manifest', 'comparison', 'gpu-test', 'npu-test', 'live', 'controls'):
        promote.add_argument('--' + name, required=True)
    promote.add_argument('--output', type=Path, default=ACTIVE)
    confirm = sub.add_parser('confirm-controls', help='Explicit human attestation after the real drive and stop checks')
    confirm.add_argument('--live', required=True)
    confirm.add_argument('--reviewer', required=True)
    confirm.add_argument('--confirmed', action='store_true', required=True,
                         help='Attest that every documented drive and stop check was performed successfully')
    confirm.add_argument('--notes', default='')
    confirm.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'confirm-controls':
            live = read_json(args.live)
            report = {'kind': 'manual_controls_review', 'created_utc': utc_now(),
                      'processing_session_id': live['processing_session_id'], 'reviewed_by': args.reviewer,
                      'checks': {name: True for name in CONTROL_CHECKS}, 'notes': args.notes}
        else:
            report = promotion_record(args.manifest, args.comparison, args.gpu_test, args.npu_test, args.live, args.controls)
        output = Path(args.output)
        temporary = output.with_name(output.name + '.tmp')
        write_json(temporary, report)
        temporary.replace(output)
        print(json.dumps({'report': str(output.resolve()), 'status': report.get('status', 'reviewed')}))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f'{type(error).__name__}: {error}\n')


if __name__ == '__main__':
    main()
