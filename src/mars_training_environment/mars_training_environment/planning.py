"""Class quotas and an offline audit of primary-target scene plans."""
import argparse
import json
from collections import Counter
from pathlib import Path

from rover_training.geometry import CLASS_NAMES, class_id_for


class ViewPlanningError(ValueError):
    """A sampled layout cannot satisfy the requested target framing."""


def validate_class_quotas(capture):
    quotas = capture.get('target_class_views', {})
    if not isinstance(quotas, dict) or set(quotas)-set(CLASS_NAMES):
        raise ValueError('target_class_views must map canonical class names to view counts')
    if any(type(value) is not int or value < 0 for value in quotas.values()):
        raise ValueError('target_class_views counts must be nonnegative integers')
    if quotas and sum(quotas.values()) != capture['views_per_scene']:
        raise ValueError('target_class_views counts must sum to views_per_scene')
    attempts = capture.get('class_quota_scene_attempts', 8)
    if type(attempts) is not int or not 1 <= attempts <= 20:
        raise ValueError('class_quota_scene_attempts must be an integer from 1 to 20')


def class_view_schedule(capture, rng, empty=False, rocks_only=False):
    """Quotas apply to mixed scenes; retain explicit rock-only/background scenes."""
    validate_class_quotas(capture)
    count = capture['views_per_scene']
    quotas = capture.get('target_class_views', {})
    if empty or not quotas:
        return [None]*count
    if rocks_only:
        return [0]*count
    classes = [cid for cid, name in enumerate(CLASS_NAMES) for _ in range(quotas.get(name, 0))]
    rng.shuffle(classes)
    return classes


def targets_for_class(targets, class_id):
    return [target for target in targets if class_id_for(target.get('class_name', 'rock')) == class_id]


def class_coverage(scenes):
    """Audit quotas against target identities, not just declared class IDs."""
    frames = Counter()
    backgrounds = Counter()
    actual = {split:Counter() for split in ('train', 'val', 'test')}
    expected = {split:Counter() for split in actual}
    errors = []
    for scene in scenes:
        ident, split = scene['id'], scene['split']
        if scene.get('class_names') != list(CLASS_NAMES) or split not in actual:
            raise ValueError(f'{ident}: invalid scene label schema or split')
        capture = scene['config']['capture']
        validate_class_quotas(capture)
        quotas = capture.get('target_class_views', {})
        by_name = {obj['name']:obj for obj in scene['objects']+scene.get('regions', [])}
        scene_counts = Counter()
        frames[split] += len(scene['views'])
        if len(scene['views']) != capture['views_per_scene']:
            errors.append(f'{ident}: unexpected number of views')
        for view in scene['views']:
            name = view.get('target_name')
            if not name:
                backgrounds[split] += 1
                if not scene['empty']:
                    errors.append(f'{ident}/{view["id"]}: populated view has no target')
                continue
            obj = by_name.get(name)
            cid = None if obj is None else class_id_for(obj.get('class_name', 'rock'))
            if cid is None or cid != view.get('target_class_id'):
                errors.append(f'{ident}/{view["id"]}: target identity/class mismatch')
                continue
            if scene['empty']:
                errors.append(f'{ident}/{view["id"]}: background scene has a target')
            if quotas and view.get('requested_target_class_id') != cid:
                errors.append(f'{ident}/{view["id"]}: requested target class was not retained')
            scene_counts[CLASS_NAMES[cid]] += 1
            actual[split][CLASS_NAMES[cid]] += 1
        if quotas:
            rocks_only = scene['seed'] in scene['config']['dataset'].get('rocks_only_seeds', [])
            wanted = (Counter() if scene['empty'] else
                      Counter({'rock':capture['views_per_scene']}) if rocks_only else Counter(quotas))
            if scene_counts != wanted:
                errors.append(f'{ident}: target quota mismatch: expected {dict(wanted)}, got {dict(scene_counts)}')
            expected[split].update(wanted)
    return {'frames':sum(frames.values()), 'split_counts':dict(frames),
            'background_views':dict(backgrounds),
            'primary_target_counts':{s:{name:c[name] for name in CLASS_NAMES} for s,c in actual.items()},
            'expected_target_counts':{s:{name:c[name] for name in CLASS_NAMES} for s,c in expected.items()},
            'errors':errors}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--require-quotas',action='store_true')
    parser.add_argument('--expected-frames',type=int)
    args = parser.parse_args(argv)
    plan = json.loads((args.bundle/'plan.json').read_text())
    if args.require_quotas and not plan['config']['capture'].get('target_class_views'):
        raise SystemExit('This bundle has no class quotas; generate a fresh balanced bundle')
    scenes = [json.loads((args.bundle/'scenes'/entry['id']/'scene.json').read_text()) for entry in plan['scenes']]
    report = class_coverage(scenes)
    if args.expected_frames is not None and report['frames'] != args.expected_frames:
        report['errors'].append(f'Expected {args.expected_frames} frames, found {report["frames"]}')
    print(json.dumps(report,indent=2))
    raise SystemExit(bool(report['errors']))


if __name__ == '__main__':
    main()
