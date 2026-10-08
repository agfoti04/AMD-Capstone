"""Validate saved scenes, labels, synchronization, split isolation and throughput."""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from .common import digest, split_for_seed, write_json
from .terrain import Terrain


def validate_scene(scene,assets):
    errors = []
    ident = scene['id']
    document = dict(scene)
    expected = document.pop('content_sha256')
    if digest(document)!=expected:
        errors.append(f'{ident}: scene content hash mismatch')
    if digest(scene['config'])!=scene['config_sha256']:
        errors.append(f'{ident}: config hash mismatch')
    if scene['split']!=split_for_seed(scene['seed']):
        errors.append(f'{ident}: seed split assignment changed')
    # Slope limit applies to the driveable surface. Ripples live only on the
    # visual mesh, which is what `heights` stores for the labeler.
    driveable = scene['terrain'].get('base_heights', scene['terrain']['heights'])
    field = Terrain(driveable, scene['terrain']['size_m'])
    if math.degrees(field.maximum_slope())>scene['config']['terrain']['max_slope_deg']+.15:
        errors.append(f'{ident}: terrain exceeds slope limit')
    if abs(float(field.height(0.,0.)))>1e-7:
        errors.append(f'{ident}: spawn pad is not at zero')
    by_id = {a['id']:a for a in assets}
    for index,obj in enumerate(scene['objects']):
        if obj['asset_id'] not in by_id:
            errors.append(f'{ident}: unknown asset')
            continue
        if by_id[obj['asset_id']]['held_out'] != (scene['split']=='test'):
            errors.append(f'{ident}: reserved shape crosses split boundary')
        if math.hypot(*obj['xyz'][:2])-obj['radius_m']<scene['config']['terrain']['spawn_clearance_m']:
            errors.append(f'{ident}: rock encroaches on spawn clearance')
        for other in scene['objects'][:index]:
            separation = math.hypot(obj['xyz'][0]-other['xyz'][0],obj['xyz'][1]-other['xyz'][1])
            if separation < obj['radius_m']+other['radius_m']+scene['config']['rocks']['clearance_m']-1e-8:
                errors.append(f'{ident}: intersecting rock footprints')
    if scene['empty'] and scene['objects']:
        errors.append(f'{ident}: empty scene has rocks')
    return errors


def contact_sheet(root,records):
    try:
        import cv2
    except ImportError:
        return None
    choices = []
    # One frame per scene plus the most occluded and clipped examples.
    seen = set()
    for record in records:
        if record['scene_id'] not in seen:
            choices.append(record)
            seen.add(record['scene_id'])
    extras = sorted(records,key=lambda r:sum(o['rejected_reason']=='fully_occluded' for o in r['objects']),reverse=True)[:2]
    choices += extras
    width,height = 320,222
    canvas = np.full((math.ceil(len(choices)/4)*height,4*width,3),35,dtype=np.uint8)
    for i,r in enumerate(choices):
        image_path = root/'previews'/r['split']/Path(r['image']).name
        if not image_path.exists():
            image_path = root/r['image']
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        image = cv2.resize(image,(width,200))
        x,y = (i%4)*width,(i//4)*height
        canvas[y:y+200,x:x+width] = image
        cv2.putText(canvas,f'{r["scene_id"]} {r["split"]} boxes={len(r["labels"])}',(x+4,y+216),
                    cv2.FONT_HERSHEY_SIMPLEX,.38,(230,230,230),1,cv2.LINE_AA)
    path = root/'previews'/'contact_sheet.jpg'
    path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(path),canvas)
    return str(path)


def validate_dataset(root,bundle,require_complete=True):
    root,bundle = Path(root),Path(bundle)
    plan = json.loads((bundle/'plan.json').read_text())
    assignments = json.loads((bundle/'splits.json').read_text())
    assets = json.loads((bundle/'assets/catalog.json').read_text())
    errors = []
    if digest(assets) != json.loads((bundle/'scenes'/plan['scenes'][0]['id']/'scene.json').read_text())['assets_sha256']:
        errors.append('Asset catalog changed after generation')
    for asset in assets:
        geometry = json.loads((bundle/asset['geometry']).read_text())
        if digest(geometry)!=asset['sha256']:
            errors.append(f'{asset["id"]}: geometry hash mismatch')
    scenes = {s['id']:json.loads((bundle/'scenes'/s['id']/'scene.json').read_text()) for s in plan['scenes']}
    for s in scenes.values():
        errors.extend(validate_scene(s,assets))
        if assignments[s['scene_group']]!=s['split']:
            errors.append(f'{s["id"]}: persisted split mismatch')
    records = []
    counts,scene_counts = Counter(),Counter()
    seeds,splits_seen = {},{}
    empty = clipped = hidden = tiny = 0
    image_hashes = {}
    bbox_checks,bbox_disagreements = [],[]
    for file in sorted((root/'metadata').glob('*/*.json')):
        r = json.loads(file.read_text())
        records.append(r)
        s = scenes.get(r['scene_id'])
        if s is None:
            errors.append(f'{file.name}: unknown scene')
            continue
        if r['scene_sha256']!=s['content_sha256'] or r['config_sha256']!=s['config_sha256']:
            errors.append(f'{file.name}: stale scene/config')
        if r['split']!=s['split'] or file.parent.name!=r['split']:
            errors.append(f'{file.name}: scene split mismatch')
        prior = seeds.setdefault(r['seed'],r['split'])
        if prior!=r['split']:
            errors.append('A seed occurs in multiple splits')
        prior = splits_seen.setdefault(r['scene_group'],r['split'])
        if prior!=r['split']:
            errors.append('A scene/trajectory occurs in multiple splits')
        image,label = root/r['image'],root/r['label']
        if not image.is_file() or not label.is_file():
            errors.append(f'{file.name}: missing image or label')
            continue
        sha = hashlib.sha256(image.read_bytes()).hexdigest()
        prior = image_hashes.setdefault(sha,r['split'])
        if prior!=r['split']:
            errors.append(f'{file.name}: identical RGB appears across splits')
        text = label.read_text().strip()
        lines = text.splitlines() if text else []
        if len(lines)!=len(r['labels']):
            errors.append(f'{file.name}: metadata/label count mismatch')
        intrinsics = r['intrinsics']
        w,h = intrinsics['width'],intrinsics['height']
        for line,index in zip(lines,range(len(lines))):
            parts = line.split()
            if len(parts)!=5 or parts[0]!='0':
                errors.append(f'{file.name}: invalid YOLO row')
                continue
            xc,yc,bw,bh = map(float,parts[1:])
            if not all(math.isfinite(v) for v in (xc,yc,bw,bh)) or min(bw,bh)<=0:
                errors.append(f'{file.name}: nonfinite or empty box')
            if min(xc-bw/2,yc-bh/2)<-1e-6 or max(xc+bw/2,yc+bh/2)>1+1e-6:
                errors.append(f'{file.name}: box outside image')
            if min(bw*w,bh*h)<s['config']['labels']['min_box_side_px']-.002:
                errors.append(f'{file.name}: tiny object policy mismatch')
        if s['empty'] and lines:
            errors.append(f'{file.name}: empty scene has labels')
        if not lines:
            empty += 1
        if r['pose_source']!='interpolated_ground_truth_odom':
            errors.append(f'{file.name}: ground-truth pose missing')
        a,b = r['pose_bracket_sim_sec']
        ts = r['timestamp_sim_sec']
        if not a-1e-7<=ts<=b+1e-7 or max(ts-a,b-ts)>s['config']['capture']['max_pose_gap_sec']+1e-7:
            errors.append(f'{file.name}: RGB/pose timestamp mismatch')
        if not r.get('clean_rgb'):
            errors.append(f'{file.name}: missing clean-RGB marker')
        clipped += sum(o['truncated'] for o in r['labels'])
        hidden += sum(o['rejected_reason']=='fully_occluded' for o in r['objects'])
        tiny += sum(o['rejected_reason'] in ('too_small','too_few_visible_pixels') for o in r['objects'])
        if r.get('depth_quality',{}).get('rock_depth_median_error_m',0)>.08:
            errors.append(f'{file.name}: geometric depth mismatch')
        check = r.get('bbox_check')
        if check is None:
            if s['config']['capture'].get('require_bbox_check'):
                errors.append(f'{file.name}: Gazebo bounding-box check missing')
        else:
            bbox_checks.append(check)
            if not check['agrees']:
                bbox_disagreements.append(r['image'])
        counts[r['split']] += 1
        scene_counts[r['scene_id']] += 1
    if require_complete:
        for entry in plan['scenes']:
            if scene_counts[entry['id']]!=entry['views']:
                errors.append(f'{entry["id"]}: expected {entry["views"]} frames, found {scene_counts[entry["id"]]}')
    if not records:
        errors.append('No captured frames')
    # Detect orphan images/labels as well as missing pairs.
    expected_images = {r['image'] for r in records}
    actual_images = {p.relative_to(root).as_posix() for p in (root/'images').glob('*/*') if p.is_file()}
    if expected_images!=actual_images:
        errors.append('Orphan or missing clean images')
    reports = [json.loads(p.read_text()) for p in (root/'reports').glob('scene_*.json')]
    wall = sum(r.get('launch_wall_seconds',r['wall_seconds']) for r in reports)
    ann = [r['annotation_seconds'] for r in records]
    quality = [r['depth_quality']['rock_depth_median_error_m'] for r in records if 'rock_depth_median_error_m' in r['depth_quality']]
    preview = contact_sheet(root,records) if records else None
    ious = [c['median_iou'] for c in bbox_checks if c['median_iou'] is not None]
    gazebo_check = {'frames_checked':len(bbox_checks),'frames_agreeing':len(bbox_checks)-len(bbox_disagreements),
                    'labels_matched':sum(c['matched'] for c in bbox_checks),
                    'labels_without_gazebo_match':sum(len(c['labels_without_gazebo_match']) for c in bbox_checks),
                    'gazebo_boxes_for_rejected_rocks':sum(c['gazebo_boxes_for_rejected_rocks'] for c in bbox_checks),
                    'gazebo_boxes_below_policy':sum(c.get('gazebo_boxes_below_policy',0) for c in bbox_checks),
                    'gazebo_boxes_unexplained':sum(c['gazebo_boxes_unexplained'] for c in bbox_checks),
                    'median_iou':float(np.median(ious)) if ious else None,
                    'disagreeing_images':bbox_disagreements}
    return {'errors':errors,'frames':len(records),'split_counts':dict(counts),'scene_counts':dict(scene_counts),
            'gazebo_bbox_check':gazebo_check,
            'empty_frames':empty,'clipped_labels':clipped,'fully_occluded_objects_rejected':hidden,'tiny_objects_rejected':tiny,
            'reserved_meshes':plan['reserved_meshes'],'capture_wall_seconds_including_launches':wall,
            'images_per_wall_second':len(records)/wall if wall else None,
            'median_annotation_seconds':float(np.median(ann)) if ann else None,
            'median_rock_depth_error_m':float(np.median(quality)) if quality else None,
            'median_scene_real_time_factor':float(np.median([r['real_time_factor'] for r in reports])) if reports else None,
            'contact_sheet':preview}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--allow-partial',action='store_true')
    args = parser.parse_args(argv)
    report = validate_dataset(args.dataset,args.bundle,not args.allow_partial)
    write_json(args.dataset/'validation.json',report)
    print(json.dumps(report,indent=2))
    raise SystemExit(bool(report['errors']))


if __name__ == '__main__':
    main()
