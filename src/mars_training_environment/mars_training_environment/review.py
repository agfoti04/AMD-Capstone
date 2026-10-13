"""Automatically populated capture review, reversible edits, and curated export.

No ROS imports. The server can run in the Halo's ordinary Python environment.
Capture images, labels, and metadata remain immutable; edits are sidecars.
"""
import argparse
import hashlib
import json
import math
import mimetypes
import secrets
import shutil
import tempfile
import threading
from collections import Counter
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from rover_training.geometry import CLASS_NAMES
from rover_training.yolo_export import write_dataset_index, write_label_file
from .annotation import iou
from .common import digest


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,
                                     prefix=path.name+'.',suffix='.tmp',delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(text)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path, document):
    _atomic_text(path,json.dumps(document,indent=2)+'\n')


def _inside(root, relative, prefixes=None):
    path = (root/relative).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError('Path is outside the dataset')
    if prefixes and path.relative_to(root.resolve()).parts[0] not in prefixes:
        raise ValueError('File is outside the permitted dataset folders')
    return path


def _label_boxes(labels, width, height):
    if not isinstance(labels, list) or len(labels) > 1000:
        raise ValueError('Expected a list of at most 1000 labels')
    result = []
    for index, label in enumerate(labels):
        cid = label.get('class_id')
        if type(cid) is not int or not 0 <= cid < len(CLASS_NAMES):
            raise ValueError('Invalid class ID')
        coords = [label.get(k) for k in ('xmin', 'ymin', 'xmax', 'ymax')]
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in coords):
            raise ValueError('Box coordinates must be finite numbers')
        x0,y0,x1,y1 = coords
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError('Box must have positive area and lie inside the image')
        name = label.get('name', f'manual_{index}')
        if not isinstance(name, str) or len(name)>120:
            raise ValueError('Invalid object name')
        result.append({'name':name, 'class_id':cid, 'class_name':CLASS_NAMES[cid],
                       'xmin':x0, 'ymin':y0, 'xmax':x1, 'ymax':y1})
    return result


def _predictions(root, split, stem, width, height):
    path = root/'predictions'/split/(stem+'.txt')
    if not path.is_file():
        return []
    predictions = []
    for index, line in enumerate(path.read_text().splitlines()):
        parts = line.split()
        if len(parts) not in (5,6):
            raise ValueError('Prediction rows must be YOLO xywh with optional confidence')
        cid = int(parts[0])
        x,y,w,h = map(float, parts[1:5])
        box = _label_boxes([{'class_id':cid, 'name':f'prediction_{index}',
                             'xmin':max(0,(x-w/2)*width), 'ymin':max(0,(y-h/2)*height),
                             'xmax':min(width,(x+w/2)*width), 'ymax':min(height,(y+h/2)*height)}], width, height)[0]
        confidence = float(parts[5]) if len(parts)==6 else None
        if confidence is not None and (not math.isfinite(confidence) or not 0<=confidence<=1):
            raise ValueError('Invalid prediction confidence')
        box['confidence'] = confidence
        predictions.append(box)
    return predictions


def prediction_errors(labels, predictions):
    """Review hints at IoU .5, not a substitute for dataset mAP evaluation."""
    box = lambda item: [item[k] for k in ('xmin','ymin','xmax','ymax')]
    pairs = sorted(((iou(box(a),box(b)),i,j) for i,a in enumerate(labels)
                    for j,b in enumerate(predictions)), reverse=True)
    used_labels, used_predictions, wrong = set(), set(), 0
    for overlap, i, j in pairs:
        if overlap < .5:
            break
        if i in used_labels or j in used_predictions:
            continue
        used_labels.add(i); used_predictions.add(j)
        wrong += labels[i]['class_id'] != predictions[j]['class_id']
    return {'wrong_class':wrong, 'missed':len(labels)-len(used_labels),
            'extra':len(predictions)-len(used_predictions)}


def review_index(root):
    root = Path(root).resolve()
    names_path = root/'classes.txt'
    if names_path.exists() and names_path.read_text().splitlines() != list(CLASS_NAMES):
        raise ValueError('Review expects the current three-class dataset')
    frames = []
    for path in sorted((root/'metadata').glob('*/*.json')):
        try:
            record = json.loads(path.read_text())
            width,height = (int(record['intrinsics'][k]) for k in ('width','height'))
            stem, split = path.stem, record['split']
            if split not in ('train','val','test') or path.parent.name != split:
                raise ValueError('Invalid persisted split')
            image = _inside(root, record['image'], {'images'})
            label_path = _inside(root, record['label'], {'labels'})
            if not image.is_file() or not label_path.is_file():
                continue  # Capture may still be publishing this record.
            original = _label_boxes(record['labels'], width, height)
            edit_path = root/'review'/'edits'/(stem+'.json')
            edit = json.loads(edit_path.read_text()) if edit_path.is_file() else {}
            base_revision = digest({'metadata':record, 'yolo':label_path.read_text(),
                                    'image_size':image.stat().st_size, 'image_mtime':image.stat().st_mtime_ns})
            stale_edit = bool(edit and edit.get('source_revision') != base_revision)
            current_edit = {} if stale_edit else edit
            labels = _label_boxes(current_edit.get('labels', original), width, height)
            flags = []
            if stale_edit:
                flags.append('stale_review_edits')
            checks = record.get('capture_checks', {})
            if not record.get('depth_quality', {}).get('visibility_check', {}).get('checked'):
                flags.append('visibility_not_audited')
            if checks and not checks.get('passed'):
                flags.append('capture_check_failed')
            if record.get('framing_adjustments'):
                flags.append('target_reframed: '+str(len(record['framing_adjustments'])))
            if (any(label.get('source') == 'ground_region' for label in record['labels'])
                    and record.get('depth_quality',{}).get('ground_region_projection') != 'visible_terrain_surface'):
                flags.append('legacy_ground_region_projection')
            for rejected in record.get('depth_quality', {}).get('visibility_check', {}).get('rejected_objects', []):
                flags.append('renderer_invisible_label_removed: '+rejected['name'])
            if record.get('bbox_check') and not record['bbox_check']['agrees']:
                flags.append('gazebo_box_disagreement')
            if any(l[k] == edge for l in labels for k,edge in [('xmin',0),('ymin',0),('xmax',width),('ymax',height)]):
                flags.append('edge_labels')
            if any(min(l['xmax']-l['xmin'],l['ymax']-l['ymin'])*960/max(width,height)<8 for l in labels):
                flags.append('small_labels')
            predictions = _predictions(root, split, stem, width, height)
            has_predictions = (root/'predictions'/split/(stem+'.txt')).is_file()
            pred_errors = prediction_errors(labels, predictions) if has_predictions else None
            if pred_errors and any(pred_errors.values()):
                flags.append('prediction_disagreement')
            frames.append({'id':stem, 'accepted':True, 'split':split, 'scene_id':record['scene_id'],
                           'image':record['image'], 'width':width, 'height':height, 'labels':labels,
                           'original_labels':original, 'predictions':predictions, 'prediction_errors':pred_errors,
                           'flags':flags, 'capture_checks':checks, 'target_name':record.get('target_name'),
                           'excluded':current_edit.get('excluded', False), 'reviewed':current_edit.get('reviewed', False),
                           'note':current_edit.get('note','Previous edits belong to a replaced capture; review or restore this frame.' if stale_edit else ''),
                           'source_revision':base_revision,
                           'revision':digest([base_revision, edit])})
        except (json.JSONDecodeError, FileNotFoundError):
            continue  # Atomic capture/review writes will appear at the next refresh.
    for path in sorted((root/'diagnostics').rglob('camera.json')):
        try:
            record = json.loads(path.read_text())
            image = next((p for p in path.parent.glob('rgb.*') if p.suffix in ('.png','.jpg','.bmp')), None)
            if image is None:
                continue
            width,height = (int(record['intrinsics'][k]) for k in ('width','height'))
            diagnostic_parts = path.relative_to(root/'diagnostics').parts[:-1]
            frames.append({'id':'diagnostic_'+'_'.join(diagnostic_parts),
                           'accepted':False, 'split':'diagnostic', 'scene_id':record.get('scene_id',diagnostic_parts[0]),
                           'image':image.relative_to(root).as_posix(), 'width':width, 'height':height,
                           'labels':_label_boxes(record.get('labels',[]), width, height), 'predictions':[],
                           'flags':record.get('checks',{}).get('errors',['capture_rejected']),
                           'capture_checks':record.get('checks',{}), 'excluded':True, 'reviewed':False,
                           'target_name':record.get('view',{}).get('target_name'), 'note':'Rejected capture; diagnostic only',
                           'revision':digest(record)})
        except (json.JSONDecodeError, FileNotFoundError):
            continue
    counts = Counter(f['split'] for f in frames if f['accepted'] and not f['excluded'])
    classes = Counter(l['class_id'] for f in frames if f['accepted'] and not f['excluded'] for l in f['labels'])
    return {'class_names':list(CLASS_NAMES), 'frames':frames, 'split_counts':dict(counts),
            'class_counts':{name:classes[i] for i,name in enumerate(CLASS_NAMES)},
            'accepted_frames':sum(f['accepted'] for f in frames),
            'excluded_frames':sum(f['accepted'] and f['excluded'] for f in frames)}


def write_review_page(root):
    root = Path(root).resolve()
    index = review_index(root)
    folder = root/'review'
    _atomic_json(folder/'index.json', index)
    template = Path(__file__).with_name('review_ui.html').read_text(encoding='utf-8')
    embedded = json.dumps(index).replace('<', '\\u003c')
    # Embedded data lets the same page work without a server as a read-only review.
    html = template.replace('__REVIEW_DATA__', embedded)
    _atomic_text(folder/'index.html',html)
    return folder/'index.html'


class ReviewStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.lock = threading.RLock()

    def save(self, frame_id, revision, labels, excluded=False, reviewed=True, note=''):
        with self.lock:
            frame = next((f for f in review_index(self.root)['frames'] if f['id']==frame_id), None)
            if frame is None or not frame['accepted']:
                raise ValueError('Only accepted capture frames can be edited')
            if revision != frame['revision']:
                raise ValueError('Frame changed since it was opened; reload before saving')
            labels = _label_boxes(labels, frame['width'], frame['height'])
            if type(excluded) is not bool or type(reviewed) is not bool or not isinstance(note,str) or len(note)>2000:
                raise ValueError('Invalid review state')
            edit_path = self.root/'review'/'edits'/(frame_id+'.json')
            previous = json.loads(edit_path.read_text()) if edit_path.exists() else None
            edit = {'source_revision':frame['source_revision'], 'labels':labels, 'excluded':excluded,
                    'reviewed':reviewed, 'note':note, 'updated_at':datetime.now(timezone.utc).isoformat()}
            history = self.root/'review'/'history.jsonl'
            history.parent.mkdir(parents=True, exist_ok=True)
            with history.open('a',encoding='utf-8') as stream:
                stream.write(json.dumps({'frame':frame_id,'before':previous,'after':edit})+'\n')
            _atomic_json(edit_path, edit)
            write_review_page(self.root)
            return next(f for f in review_index(self.root)['frames'] if f['id']==frame_id)

    def restore(self, frame_id, revision):
        frame = next((f for f in review_index(self.root)['frames'] if f['id']==frame_id), None)
        if frame is None or not frame['accepted']:
            raise ValueError('Unknown accepted frame')
        return self.save(frame_id, revision, frame['original_labels'], False, False, 'Restored original capture')

    def export(self):
        with self.lock:
            validation_path = self.root/'validation.json'
            if not validation_path.is_file():
                raise ValueError('Complete and validate the capture before exporting training data')
            validation = json.loads(validation_path.read_text())
            index = review_index(self.root)
            if validation.get('errors') or validation.get('frames') != index['accepted_frames']:
                raise ValueError('Capture validation is incomplete or contains errors')
            if any('stale_review_edits' in f['flags'] for f in index['frames']):
                raise ValueError('Review or restore frames with stale edits before exporting')
            frames = [f for f in index['frames'] if f['accepted'] and not f['excluded']]
            if any(not any(f['split']==split for f in frames) for split in ('train','val','test')):
                raise ValueError('Keep at least one image in every split')
            revision = digest([(f['id'],f['revision'],f['excluded']) for f in index['frames'] if f['accepted']])[:16]
            output = self.root/'review'/'exports'/revision
            audit_path = output/'review_export.json'
            if audit_path.is_file():
                report = json.loads(audit_path.read_text())
                for item in report['audit']:
                    image = output/'images'/item['split']/Path(item['source_image']).name
                    label = output/'labels'/item['split']/(image.stem+'.txt')
                    if (not image.is_file() or not label.is_file()
                            or hashlib.sha256(image.read_bytes()).hexdigest()!=item['image_sha256']
                            or hashlib.sha256(label.read_bytes()).hexdigest()!=item['label_sha256']):
                        raise ValueError('Existing reviewed export was modified; restore it or create a new review revision')
                # The same dataset may move between /work and its Halo host path.
                write_dataset_index(output)
                report['data_yaml'] = str(output/'data.yaml')
                _atomic_json(audit_path,report)
                return report
            output.mkdir(parents=True,exist_ok=True)
            write_dataset_index(output)
            audits = []
            for frame in frames:
                source = _inside(self.root,frame['image'], {'images'})
                dest = output/'images'/frame['split']/source.name
                dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(source,dest)
                label = output/'labels'/frame['split']/(source.stem+'.txt')
                write_label_file(label,frame['labels'],frame['width'],frame['height'])
                audits.append({'id':frame['id'], 'split':frame['split'], 'source_image':frame['image'],
                               'source_revision':frame['source_revision'], 'review_revision':frame['revision'],
                               'image_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                               'label_sha256':hashlib.sha256(label.read_bytes()).hexdigest(),
                               'labels':frame['labels'], 'reviewed':frame['reviewed'], 'note':frame['note']})
            report = {'revision':revision,'frames':len(frames),'split_counts':index['split_counts'],
                      'class_counts':index['class_counts'], 'data_yaml':str(output/'data.yaml'),
                      'excluded':[f['id'] for f in index['frames'] if f['accepted'] and f['excluded']],
                      'audit':audits}
            _atomic_json(audit_path,report)
            return report


def make_server(root, port=8765, host='127.0.0.1'):
    store = ReviewStore(root)
    token = secrets.token_urlsafe(32)
    write_review_page(store.root)

    class Handler(BaseHTTPRequestHandler):
        def json_response(self, value, status=200):
            body = json.dumps(value).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type','application/json')
            self.send_header('Cache-Control','no-store')
            self.send_header('Content-Length',str(len(body)))
            self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            try:
                route = unquote(urlsplit(self.path).path)
                if route == '/api/index':
                    self.json_response(dict(review_index(store.root),token=token))
                    return
                if route == '/':
                    path = store.root/'review'/'index.html'
                elif route.startswith('/file/'):
                    path = _inside(store.root,route[6:], {'images','diagnostics'})
                else:
                    self.send_error(404); return
                body = path.read_bytes()
                self.send_response(200)
                self.send_header('Content-Type',mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
                self.send_header('Content-Length',str(len(body)))
                self.send_header('Cache-Control','no-store')
                self.end_headers(); self.wfile.write(body)
            except (ValueError,FileNotFoundError):
                self.send_error(404)

        def do_POST(self):
            if self.headers.get('X-Review-Token') != token:
                self.json_response({'error':'Refresh the review page before saving'},403); return
            try:
                length = int(self.headers.get('Content-Length','0'))
                if not 0 < length <= 1000000:
                    raise ValueError('Invalid request size')
                payload = json.loads(self.rfile.read(length))
                if self.path == '/api/save':
                    result = store.save(payload['id'],payload['revision'],payload['labels'],
                                        payload.get('excluded',False),payload.get('reviewed',True),payload.get('note',''))
                elif self.path == '/api/restore':
                    result = store.restore(payload['id'],payload['revision'])
                elif self.path == '/api/export':
                    result = store.export()
                else:
                    self.send_error(404); return
                self.json_response(result)
            except (ValueError,KeyError,TypeError,json.JSONDecodeError) as exc:
                self.json_response({'error':str(exc)},400)

    server = ThreadingHTTPServer((host,port),Handler)
    return server


def serve(root, port=8765):
    server = make_server(root, port)
    print(f'Review: http://127.0.0.1:{server.server_port} — changes are reversible sidecars',flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--build-only',action='store_true')
    parser.add_argument('--export',action='store_true')
    args = parser.parse_args(argv)
    if args.export:
        print(json.dumps(ReviewStore(args.dataset).export(),indent=2))
    elif args.build_only:
        print(write_review_page(args.dataset))
    else:
        serve(args.dataset,args.port)


if __name__ == '__main__':
    main()
