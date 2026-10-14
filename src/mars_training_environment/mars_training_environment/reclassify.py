"""Create a separate, reviewable mudstone dataset from an immutable reviewed export.

Images, edited box coordinates and split assignments are reused. Object identity
comes from capture metadata, never detector predictions or filename guesses.
"""
import argparse
import copy
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from rover_training.geometry import CLASS_NAMES, FOUR_CLASS_NAMES
from rover_training.yolo_export import dataset_class_names, write_dataset_index, write_label_file, yolo_line
from .common import digest
from .review import ReviewStore, _atomic_json, _inside, _label_boxes, review_index, write_review_page


SLAB_APPEARANCES = {'cheyava_rock', 'veined_lookalike'}
SPLITS = ('train', 'val', 'test')
# Pilot coverage guidance, not statistical evidence of real-world accuracy.
MINIMUM_COVERAGE = {'train': {'instances': 30, 'images': 10, 'scene_groups': 3},
                    'val': {'instances': 10, 'images': 5, 'scene_groups': 2},
                    'test': {'instances': 10, 'images': 5, 'scene_groups': 2}}


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def coverage_report(root):
    index = review_index(root)
    names = index['class_names']
    frames = [frame for frame in index['frames'] if frame['accepted'] and not frame['excluded']]
    counts, warnings = {}, []
    for split in SPLITS:
        selected = [frame for frame in frames if frame['split'] == split]
        counts[split] = {}
        for cid, name in enumerate(names):
            positive = [frame for frame in selected if any(label['class_id'] == cid for label in frame['labels'])]
            values = {'instances': sum(label['class_id'] == cid for frame in selected for label in frame['labels']),
                      'images': len(positive), 'scene_groups': len({frame['scene_group'] for frame in positive})}
            counts[split][name] = values
            for measure, minimum in MINIMUM_COVERAGE[split].items():
                if values[measure] < minimum:
                    warnings.append(f'{split}/{name}: {values[measure]} {measure}; pilot guidance is at least {minimum}')
    pending = [frame['id'] for frame in frames if frame.get('class_review_required') and not frame['reviewed']]
    return {'class_names': names, 'frames': len(frames), 'split_counts': dict(Counter(f['split'] for f in frames)),
            'class_coverage': counts, 'pending_class_reviews': pending, 'coverage_warnings': warnings,
            'coverage_guidance': MINIMUM_COVERAGE}


def _source_plan(source, reviewed):
    source, reviewed = Path(source).resolve(), Path(reviewed).resolve()
    if dataset_class_names(source) != CLASS_NAMES:
        raise ValueError('The source must be the original three-class capture')
    validation_path = source / 'validation.json'
    validation = json.loads(validation_path.read_text())
    if validation.get('errors') or not validation.get('frames'):
        raise ValueError('Source capture validation is incomplete or contains errors')
    audit_path = reviewed / 'review_export.json'
    audit = json.loads(audit_path.read_text())
    if dataset_class_names(reviewed) != CLASS_NAMES or audit.get('class_names', list(CLASS_NAMES)) != list(CLASS_NAMES):
        raise ValueError('Use a three-class reviewed export as the source')
    entries = audit['audit']
    if not entries or audit.get('frames') != len(entries):
        raise ValueError('Source reviewed export frame count does not match its audit')
    prepared, seen, groups = [], set(), {}
    for item in entries:
        split, stem = item['split'], item['id']
        if split not in SPLITS or stem in seen or Path(stem).name != stem:
            raise ValueError('Invalid or duplicate reviewed frame identity')
        seen.add(stem)
        source_image = _inside(source, item['source_image'], {'images'})
        if source_image.parent.name != split or source_image.stem != stem:
            raise ValueError('Source image does not match its persisted split/identity')
        image = _inside(reviewed, f'images/{split}/{source_image.name}', {'images'})
        label_path = _inside(reviewed, f'labels/{split}/{stem}.txt', {'labels'})
        if sha256(image) != item['image_sha256'] or sha256(source_image) != item['image_sha256']:
            raise ValueError(f'{stem}: reviewed/source image changed after export')
        if sha256(label_path) != item['label_sha256']:
            raise ValueError(f'{stem}: reviewed training labels changed after export')
        metadata_path = _inside(source, f'metadata/{split}/{stem}.json', {'metadata'})
        record = json.loads(metadata_path.read_text())
        if record['split'] != split or record['image'] != item['source_image']:
            raise ValueError(f'{stem}: capture metadata identity mismatch')
        raw_label = _inside(source, record['label'], {'labels'})
        revision = digest({'metadata': record, 'yolo': raw_label.read_text(),
                           'image_size': source_image.stat().st_size,
                           'image_mtime': source_image.stat().st_mtime_ns})
        if revision != item['source_revision']:
            raise ValueError(f'{stem}: source capture revision changed; export its current review before relabeling')
        group = record.get('scene_group', record['scene_id'])
        if group in groups and groups[group] != split:
            raise ValueError(f'{group}: scene group crosses dataset splits')
        groups[group] = split
        width, height = (int(record['intrinsics'][key]) for key in ('width', 'height'))
        labels = _label_boxes(item['labels'], width, height)
        expected = '\n'.join(yolo_line(label, width, height) for label in labels)
        if label_path.read_text().strip() != expected:
            raise ValueError(f'{stem}: export audit boxes do not match the training label file')
        original = {obj['name']: obj for obj in record.get('objects', []) + record['labels']}
        changed, issues, converted = [], [], []
        for label in labels:
            evidence = original.get(label['name'])
            appearance = evidence.get('appearance_name') if evidence else None
            slab = bool(evidence and (appearance in SLAB_APPEARANCES or evidence.get('asset_id') == 'cand_slab'))
            result = dict(evidence or {})
            result.update(label)  # approved coordinates take precedence over original geometry
            if 'bbox_xyxy' in result:
                result['bbox_xyxy'] = [result[key] for key in ('xmin', 'ymin', 'xmax', 'ymax')]
            if slab and label['class_id'] == 0:
                result.update(class_id=3, class_name='mudstone_slab')
                changed.append(label['name'])
            elif slab:
                issues.append(f'{label["name"]}: slab metadata conflicts with reviewed {label["class_name"]}; inspect its class')
            elif label['class_id'] == 0 and (evidence is None or not appearance):
                issues.append(f'{label["name"]}: no original appearance identity; check rock versus mudstone_slab manually')
            converted.append(result)
        updated = copy.deepcopy(record)
        updated.update(image=f'images/{split}/{source_image.name}', label=f'labels/{split}/{stem}.txt', labels=converted)
        updated['four_class_relabel'] = {'source_metadata_sha256': sha256(metadata_path),
                                        'source_review_revision': item['review_revision'],
                                        'changed_objects': changed, 'issues': issues,
                                        'review_required': bool(changed or issues),
                                        'capture_checks_inherited': True}
        note = item.get('note', '')
        if changed:
            note += '\nReclassified as mudstone_slab: ' + ', '.join(changed)
        if issues:
            note += '\n' + '\n'.join(issues)
        prepared.append({'id': stem, 'split': split, 'image': image, 'record': updated,
                         'reviewed': bool(item.get('reviewed')) and not (changed or issues), 'note': note.strip()})
    if {entry['split'] for entry in prepared} != set(SPLITS):
        raise ValueError('Keep at least one reviewed image in every split')
    return prepared, {'source_dataset': str(source), 'source_review_export': str(reviewed),
                      'source_review_export_sha256': sha256(audit_path),
                      'source_validation_sha256': sha256(validation_path),
                      'source_excluded': audit.get('excluded', []), 'source_capture_validation': validation}


def prepare_dataset(source, reviewed, output):
    output = Path(output).resolve()
    for origin in (Path(source).resolve(), Path(reviewed).resolve()):
        if output.is_relative_to(origin) or origin.is_relative_to(output):
            raise ValueError('Choose a separate output directory outside the source capture and reviewed export')
    if output.exists():
        raise ValueError('Output already exists; reuse it for review, or choose a fresh version')
    prepared, provenance = _source_plan(source, reviewed)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + '.staging-', dir=output.parent))
    try:
        write_dataset_index(staging, FOUR_CLASS_NAMES)
        for item in prepared:
            record, split, stem = item['record'], item['split'], item['id']
            destination = staging / record['image']
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item['image'], destination)
            write_label_file(staging / record['label'], record['labels'],
                             record['intrinsics']['width'], record['intrinsics']['height'], FOUR_CLASS_NAMES)
            _atomic_json(staging / f'metadata/{split}/{stem}.json', record)
        _atomic_json(staging / 'validation.json', {'errors': [], 'frames': len(prepared),
                     'validation_kind': 'relabel_of_validated_capture',
                     'source_validation_sha256': provenance['source_validation_sha256']})
        initial = {frame['id']: frame for frame in review_index(staging)['frames']}
        for item in prepared:
            frame = initial[item['id']]
            _atomic_json(staging / f'review/edits/{item["id"]}.json',
                         {'source_revision': frame['source_revision'], 'labels': frame['labels'],
                          'excluded': False, 'reviewed': item['reviewed'], 'note': item['note'],
                          'updated_at': datetime.now(timezone.utc).isoformat()})
        report = coverage_report(staging)
        _atomic_json(staging / 'relabel_report.json', dict(report, **provenance,
                     changed_boxes=sum(len(item['record']['four_class_relabel']['changed_objects']) for item in prepared),
                     uncertain_frames=[item['id'] for item in prepared if item['record']['four_class_relabel']['issues']]))
        write_review_page(staging)
        staging.rename(output)
    except BaseException:
        # Only this function's verified temporary sibling is removed on failure.
        if staging.resolve().parent == output.parent and staging.name.startswith(output.name + '.staging-'):
            shutil.rmtree(staging)
        raise
    write_dataset_index(output, FOUR_CLASS_NAMES)  # replace the temporary absolute YAML path
    return dict(report, dataset=str(output), review_page=str(output / 'review/index.html'))


def approve_dataset(root, note):
    """Record the user's completed class review in reversible review sidecars."""
    root = Path(root).resolve()
    if dataset_class_names(root) != FOUR_CLASS_NAMES:
        raise ValueError('Expected the prepared four-class dataset')
    return dict(ReviewStore(root).approve_class_reviews(note), dataset=str(root))


def export_dataset(root, require_coverage=False):
    root = Path(root).resolve()
    if dataset_class_names(root) != FOUR_CLASS_NAMES:
        raise ValueError('Expected the prepared four-class dataset')
    report = coverage_report(root)
    if report['pending_class_reviews']:
        raise ValueError(f'Review {len(report["pending_class_reviews"])} changed/uncertain frames in the browser first')
    if require_coverage and report['coverage_warnings']:
        raise ValueError('Coverage needs additional examples before training:\n' + '\n'.join(report['coverage_warnings']))
    exported = ReviewStore(root).export()
    result = dict(report, data_yaml=exported['data_yaml'], revision=exported['revision'])
    _atomic_json(root / 'training_export.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare', help='Copy an audited three-class export and relabel its slabs')
    prepare.add_argument('--source', type=Path, required=True)
    prepare.add_argument('--review-export', type=Path, required=True)
    prepare.add_argument('--output', type=Path, required=True)
    audit = commands.add_parser('audit', help='Count current examples, scene groups and outstanding reviews')
    audit.add_argument('--dataset', type=Path, required=True)
    approve = commands.add_parser('approve', help='Record a completed human review of all pending class changes')
    approve.add_argument('--dataset', type=Path, required=True)
    approve.add_argument('--note', required=True, help='Who reviewed and approved these labels; stored in review history')
    export = commands.add_parser('export', help='Export current reviewed labels for training')
    export.add_argument('--dataset', type=Path, required=True)
    export.add_argument('--require-coverage', action='store_true')
    export.add_argument('--print-data-path', action='store_true')
    export.add_argument('--run-manifest', type=Path, help='Record this exact export for a fresh training run')
    args = parser.parse_args(argv)
    try:
        if args.command == 'prepare':
            result = prepare_dataset(args.source, args.review_export, args.output)
        elif args.command == 'audit':
            result = coverage_report(args.dataset)
        elif args.command == 'approve':
            result = approve_dataset(args.dataset, args.note)
        else:
            result = export_dataset(args.dataset, args.require_coverage)
            if args.run_manifest:
                args.run_manifest.parent.mkdir(parents=True, exist_ok=True)
                with args.run_manifest.open('x', encoding='utf-8') as stream:
                    json.dump(result, stream, indent=2)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f'{error}\n')
    if args.command == 'export' and args.print_data_path:
        print(result['data_yaml'])
    else:
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
