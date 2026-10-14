"""Verify a taxonomy migration preserves reviewed evidence and requires review."""
import copy
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mars_training_environment.common import png, write_json
from mars_training_environment.reclassify import approve_dataset, coverage_report, export_dataset, main, prepare_dataset
from mars_training_environment.review import ReviewStore, review_index
from rover_training.geometry import CLASS_NAMES, FOUR_CLASS_NAMES
from rover_training.yolo_export import read_label_file, write_dataset_index, write_label_file


def capture_fixture(root):
    write_dataset_index(root)
    appearances = [(0, 'rock', 'scan'), (1, 'silica_soil', 'silica_soil'),
                   (2, 'sulfur_rock', 'cand_sulfur'), (0, 'veined_lookalike', 'cand_slab'),
                   (0, 'cheyava_rock', 'cand_slab'), (0, 'veined_lookalike', 'cand_slab')]
    for split_index, split in enumerate(('train', 'val', 'test')):
        for index in range(2):
            scene = f'scene_{100 + split_index * 10 + index:06d}'
            stem = scene + '_view_000'
            labels = []
            for ordinal, (cid, appearance, asset) in enumerate(appearances):
                labels.append(dict(name=f'object_{ordinal}', class_id=cid, appearance_name=appearance,
                                   asset_id=asset, class_name=CLASS_NAMES[cid], xmin=ordinal * 12 + 1,
                                   ymin=10, xmax=ordinal * 12 + 10, ymax=24))
            image, label = f'images/{split}/{stem}.png', f'labels/{split}/{stem}.txt'
            (root / image).parent.mkdir(parents=True, exist_ok=True)
            png(root / image, np.full((60, 80, 3), 73, np.uint8))
            write_label_file(root / label, labels, 80, 60)
            write_json(root / f'metadata/{split}/{stem}.json',
                       dict(scene_id=scene, scene_group=scene, split=split, view='view_000', image=image, label=label,
                            intrinsics={'width': 80, 'height': 60}, labels=labels, objects=labels,
                            depth_quality={'visibility_check': {'checked': True}}, capture_checks={'passed': True}))
    write_json(root / 'validation.json', {'errors': [], 'frames': 6})
    store = ReviewStore(root)
    for frame in review_index(root)['frames']:
        kept = copy.deepcopy(frame['labels'][:-1])  # reviewer removed the last slab
        kept[3]['xmax'] -= 1  # reviewer corrected a slab box
        store.save(frame['id'], frame['revision'], kept, reviewed=True, note='Approved extents')
    return Path(store.export()['data_yaml']).parent


class RelabelTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.base = Path(self.folder.name)
        self.source = self.base / 'original'
        self.source.mkdir()
        self.export = capture_fixture(self.source)
        self.output = self.base / 'fourclass'

    def tearDown(self):
        self.folder.cleanup()

    def approve(self):
        store = ReviewStore(self.output)
        for frame in review_index(self.output)['frames']:
            store.save(frame['id'], frame['revision'], frame['labels'], reviewed=True, note=frame['note'])

    def test_conversion_preserves_images_reviewed_boxes_removed_labels_splits_and_source(self):
        before = {path.relative_to(self.source): path.read_bytes() for path in self.source.rglob('*') if path.is_file()}
        report = prepare_dataset(self.source, self.export, self.output)
        self.assertEqual(report['frames'], 6)
        self.assertEqual(report['split_counts'], {'train': 2, 'val': 2, 'test': 2})
        self.assertEqual(len(report['pending_class_reviews']), 6)
        self.assertEqual((self.output / 'classes.txt').read_text().splitlines(), list(FOUR_CLASS_NAMES))
        for frame in review_index(self.output)['frames']:
            self.assertEqual([label['class_id'] for label in frame['labels']], [0, 1, 2, 3, 3])
            self.assertEqual(frame['labels'][3]['xmax'], 45)
            self.assertEqual((self.output / frame['image']).read_bytes(), (self.source / frame['image']).read_bytes())
            self.assertIn('class_review_required', frame['flags'])
        self.assertEqual(before, {p.relative_to(self.source): p.read_bytes() for p in self.source.rglob('*') if p.is_file()})
        self.assertNotIn('.staging-', (self.output / 'data.yaml').read_text())

    def test_export_requires_new_class_review_and_supports_four_class_editor(self):
        prepare_dataset(self.source, self.export, self.output)
        with self.assertRaisesRegex(ValueError, 'Review'):
            ReviewStore(self.output).export()
        self.approve()
        report = export_dataset(self.output)
        root = Path(report['data_yaml']).parent
        self.assertIn('3: mudstone_slab', (root / 'data.yaml').read_text())
        label = next((root / 'labels').glob('*/*.txt'))
        self.assertEqual([item['class_id'] for item in read_label_file(label, 80, 60, FOUR_CLASS_NAMES)], [0, 1, 2, 3, 3])
        self.assertEqual(report['pending_class_reviews'], [])
        with self.assertRaisesRegex(ValueError, 'additional examples'):
            export_dataset(self.output, require_coverage=True)

    def test_coverage_counts_instances_images_and_independent_scene_groups(self):
        prepare_dataset(self.source, self.export, self.output)
        report = coverage_report(self.output)
        self.assertEqual(report['class_coverage']['train']['mudstone_slab'],
                         {'instances': 4, 'images': 2, 'scene_groups': 2})
        self.assertTrue(report['coverage_warnings'])

    def test_bulk_approval_preserves_labels_notes_exclusions_and_previous_approvals(self):
        prepare_dataset(self.source, self.export, self.output)
        frames = review_index(self.output)['frames']
        store = ReviewStore(self.output)
        store.save(frames[0]['id'], frames[0]['revision'], frames[0]['labels'],
                   excluded=True, reviewed=False, note='Exclude this uncertain frame')
        labels = copy.deepcopy(frames[1]['labels'])
        labels[3]['xmin'] += 1
        store.save(frames[1]['id'], frames[1]['revision'], labels, reviewed=True, note='Already approved edit')
        frames = review_index(self.output)['frames']
        untouched_edits = {frame['id']: (self.output / f'review/edits/{frame["id"]}.json').read_bytes()
                           for frame in frames[:2]}
        before = {path.relative_to(self.output): path.read_bytes() for folder in ('images', 'labels', 'metadata')
                  for path in (self.output / folder).rglob('*') if path.is_file()}
        report = approve_dataset(self.output, 'Reviewed and approved by Ian')
        self.assertEqual(report['approved_frames'], 4)
        self.assertEqual(report['pending_class_reviews'], [])
        current = review_index(self.output)
        self.assertEqual(current['split_counts'], {'test': 1, 'train': 2, 'val': 2})
        self.assertEqual([frame['labels'] for frame in frames], [frame['labels'] for frame in current['frames']])
        self.assertEqual([frame['note'] for frame in frames], [frame['note'] for frame in current['frames']])
        self.assertTrue(all(frame['reviewed'] for frame in current['frames'] if not frame['excluded']))
        for stem, content in untouched_edits.items():
            self.assertEqual((self.output / f'review/edits/{stem}.json').read_bytes(), content)
        self.assertEqual(before, {path.relative_to(self.output): path.read_bytes() for folder in ('images', 'labels', 'metadata')
                                  for path in (self.output / folder).rglob('*') if path.is_file()})
        history = (self.output / 'review/history.jsonl').read_bytes()
        approvals = [json.loads(line) for line in history.decode().splitlines()
                     if json.loads(line).get('action') == 'approve_class_review']
        self.assertEqual(len(approvals), 4)
        for approval in approvals:
            self.assertEqual(approval['approval_note'], 'Reviewed and approved by Ian')
            self.assertEqual(approval['before']['labels'], approval['after']['labels'])
            self.assertFalse(approval['before']['reviewed'])
            self.assertTrue(approval['after']['reviewed'])
        self.assertEqual(approve_dataset(self.output, 'Reviewed and approved by Ian')['approved_frames'], 0)
        self.assertEqual((self.output / 'review/history.jsonl').read_bytes(), history)
        self.assertEqual(export_dataset(self.output)['frames'], 5)

    def test_bulk_approval_rejects_stale_or_incomplete_data_before_any_edit(self):
        prepare_dataset(self.source, self.export, self.output)
        metadata = next((self.output / 'metadata').glob('*/*.json'))
        original = metadata.read_bytes()
        record = json.loads(original)
        record['timestamp_sim_sec'] = 1000
        write_json(metadata, record)
        edits = {path.name: path.read_bytes() for path in (self.output / 'review/edits').glob('*.json')}
        with self.assertRaisesRegex(ValueError, 'stale edits'):
            approve_dataset(self.output, 'Reviewed and approved by Ian')
        self.assertEqual(edits, {path.name: path.read_bytes() for path in (self.output / 'review/edits').glob('*.json')})
        self.assertFalse((self.output / 'review/history.jsonl').exists())
        metadata.write_bytes(original)
        write_json(self.output / 'validation.json', {'errors': [], 'frames': 7})
        with self.assertRaisesRegex(ValueError, 'validate the capture'):
            approve_dataset(self.output, 'Reviewed and approved by Ian')
        self.assertEqual(edits, {path.name: path.read_bytes() for path in (self.output / 'review/edits').glob('*.json')})
        self.assertFalse((self.output / 'review/history.jsonl').exists())

    def test_approval_cli_records_completed_review_and_allows_export(self):
        prepare_dataset(self.source, self.export, self.output)
        with contextlib.redirect_stdout(io.StringIO()) as text:
            main(['approve', '--dataset', str(self.output), '--note', 'Reviewed and approved by Ian'])
        report = json.loads(text.getvalue())
        self.assertEqual(report['approved_frames'], 6)
        self.assertEqual(report['pending_class_reviews'], [])
        self.assertEqual(export_dataset(self.output)['frames'], 6)

    def test_bulk_approval_preflights_concurrent_edits_before_writing(self):
        prepare_dataset(self.source, self.export, self.output)
        opened = review_index(self.output)
        path = self.output / f'review/edits/{opened["frames"][-1]["id"]}.json'
        edit = json.loads(path.read_text())
        edit['note'] = 'A newer reviewer edit'
        write_json(path, edit)
        before = {file.name: file.read_bytes() for file in (self.output / 'review/edits').glob('*.json')}
        with patch('mars_training_environment.review.review_index', return_value=opened):
            with self.assertRaisesRegex(ValueError, 'Review changed during approval'):
                approve_dataset(self.output, 'Reviewed and approved by Ian')
        self.assertEqual(before, {file.name: file.read_bytes() for file in (self.output / 'review/edits').glob('*.json')})
        self.assertFalse((self.output / 'review/history.jsonl').exists())

    def test_bulk_approval_requires_four_class_data_and_an_explicit_note(self):
        with self.assertRaisesRegex(ValueError, 'four-class'):
            approve_dataset(self.source, 'Reviewed and approved by Ian')
        prepare_dataset(self.source, self.export, self.output)
        with self.assertRaisesRegex(ValueError, 'approval note'):
            approve_dataset(self.output, '  ')
        self.assertEqual(len(coverage_report(self.output)['pending_class_reviews']), 6)

    def test_training_run_keeps_its_export_when_later_reviews_create_another_revision(self):
        prepare_dataset(self.source, self.export, self.output)
        self.approve()
        manifest = self.base / 'run_dataset.json'
        args = ['export', '--dataset', str(self.output), '--run-manifest', str(manifest), '--print-data-path']
        with contextlib.redirect_stdout(io.StringIO()) as text:
            main(args)
        frozen = manifest.read_bytes()
        saved = json.loads(frozen)
        self.assertEqual(text.getvalue().strip(), saved['data_yaml'])
        frame = review_index(self.output)['frames'][0]
        labels = copy.deepcopy(frame['labels'])
        labels[0]['class_id'] = 1
        ReviewStore(self.output).save(frame['id'], frame['revision'], labels, reviewed=True)
        later = export_dataset(self.output)
        self.assertNotEqual(saved['revision'], later['revision'])
        self.assertEqual(manifest.read_bytes(), frozen)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(args)
        self.assertEqual(manifest.read_bytes(), frozen)

    def test_manual_renaming_is_flagged_without_guessing_from_box_overlap(self):
        store = ReviewStore(self.source)
        frame = review_index(self.source)['frames'][0]
        labels = copy.deepcopy(frame['labels'])
        labels[3]['name'] = 'manual_slab'
        store.save(frame['id'], frame['revision'], labels, reviewed=True)
        reviewed = Path(store.export()['data_yaml']).parent
        prepare_dataset(self.source, reviewed, self.output)
        changed = next(f for f in review_index(self.output)['frames'] if f['id'] == frame['id'])
        self.assertEqual(changed['labels'][3]['class_id'], 0)
        self.assertIn('no original appearance identity', changed['note'])
        self.assertFalse(changed['reviewed'])

    def test_conflicting_reviewed_class_is_preserved_and_flagged(self):
        store = ReviewStore(self.source)
        frame = review_index(self.source)['frames'][0]
        labels = copy.deepcopy(frame['labels'])
        labels[3]['class_id'] = 1
        store.save(frame['id'], frame['revision'], labels, reviewed=True)
        prepare_dataset(self.source, Path(store.export()['data_yaml']).parent, self.output)
        changed = next(f for f in review_index(self.output)['frames'] if f['id'] == frame['id'])
        self.assertEqual(changed['labels'][3]['class_id'], 1)
        self.assertIn('conflicts with reviewed', changed['note'])

    def test_edited_export_or_stale_metadata_is_rejected_before_copying(self):
        label = next((self.export / 'labels').glob('*/*.txt'))
        text = label.read_text()
        label.write_text('')
        with self.assertRaisesRegex(ValueError, 'labels changed'):
            prepare_dataset(self.source, self.export, self.output)
        self.assertFalse(self.output.exists())
        label.write_text(text)
        metadata = next((self.source / 'metadata').glob('*/*.json'))
        record = json.loads(metadata.read_text())
        record['timestamp_sim_sec'] = 1000
        write_json(metadata, record)
        with self.assertRaisesRegex(ValueError, 'revision changed'):
            prepare_dataset(self.source, self.export, self.output)
        self.assertFalse(self.output.exists())

    def test_existing_or_nested_destination_cannot_replace_source_data(self):
        with self.assertRaisesRegex(ValueError, 'separate output'):
            prepare_dataset(self.source, self.export, self.source / 'new')
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError, 'already exists'):
            prepare_dataset(self.source, self.export, self.output)

    def test_source_frame_exclusions_are_preserved(self):
        store = ReviewStore(self.source)
        frame = review_index(self.source)['frames'][0]
        store.save(frame['id'], frame['revision'], frame['labels'], excluded=True)
        prepare_dataset(self.source, Path(store.export()['data_yaml']).parent, self.output)
        self.assertEqual(len(review_index(self.output)['frames']), 5)
        self.assertFalse(any(f['id'] == frame['id'] for f in review_index(self.output)['frames']))


if __name__ == '__main__':
    unittest.main()
