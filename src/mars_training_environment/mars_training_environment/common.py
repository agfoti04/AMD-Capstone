"""Small deterministic I/O and configuration helpers (no ROS import)."""
import hashlib
import json
import math
import struct
import zlib
from pathlib import Path

import numpy as np

from rover_training.geometry import APPEARANCE_CLASSES, CLASS_NAMES


def require_label_schema(document):
    if document.get('class_names') != list(CLASS_NAMES):
        raise ValueError('Scene/bundle uses another label schema; generate a fresh three-class bundle and dataset')


def config_path():
    local = Path(__file__).resolve().parents[1] / 'config' / 'sample.json'
    if local.exists():
        return local
    from ament_index_python.packages import get_package_share_directory
    return Path(get_package_share_directory('mars_training_environment')) / 'config' / 'sample.json'


def read_config(path=None):
    config = json.loads(config_path().read_text())
    if path is not None:
        supplied = json.loads(Path(path).read_text())
        for section, values in supplied.items():
            if section not in config or not isinstance(values, dict):
                raise ValueError(f'Unknown/non-object configuration section: {section}')
            unknown = set(values) - set(config[section])
            if unknown:
                raise ValueError(f'Unknown {section} keys: {sorted(unknown)}')
            config[section].update(values)
    t, r, c, cap = (config[k] for k in ('terrain', 'rocks', 'camera', 'capture'))
    n = t['grid_size']
    visual_n = t['visual_grid_size']
    if not isinstance(n, int) or n < 17 or (n - 1) & (n - 2):
        raise ValueError('grid_size must be 2^n + 1, at least 17')
    if not isinstance(visual_n, int) or visual_n < n or visual_n > 257 or (visual_n - 1) & (visual_n - 2):
        raise ValueError('visual_grid_size must be 2^n + 1, at least grid_size, and at most 257')
    for key in ('wavelength_m', 'height_m', 'ridge_wavelength_m', 'ridge_height_m'):
        lo, hi = t['ripples'][key]
        if not 0 < lo <= hi:
            raise ValueError(f'Invalid ripple range {key}')
    if t['size_m'] < 6 or not 0 < t['spawn_clearance_m'] < t['size_m'] / 4:
        raise ValueError('Terrain size or spawn clearance is invalid')
    if not 0 < t['max_slope_deg'] < 25:
        raise ValueError('Use a slope limit between 0 and 25 degrees')
    if t['texture_tile_m'] <= 0 or t['ground_layers'] < 1 or t['composite_px'] < 256:
        raise ValueError('Invalid ground texture tiling')
    if not 0 < c['near_m'] < c['far_m'] or not 0.2 < c['horizontal_fov_rad'] < 2.5:
        raise ValueError('Invalid camera clipping planes or field of view')
    if min(c['width'], c['height'], c['rate_hz'], cap['views_per_scene']) <= 0:
        raise ValueError('Camera dimensions, rate and view count must be positive')
    if not 0 <= r['procedural_count'] <= 64:
        raise ValueError('procedural_count must be 0..64')
    if r['procedural_count'] + len(r['scanned_ids']) + (4 if r['include_existing'] else 0) < 4:
        raise ValueError('Use at least four rock shapes')
    for lo, hi in (r['count_range'], r['sampling_size_m'], r['obstacle_size_m'],
                   r['burial_fraction'], cap['distance_m'], t['tint_strength']):
        if not 0 <= lo <= hi:
            raise ValueError('Invalid configuration range')
    if cap['distance_m'][0] <= c['near_m'] + 0.25 or cap['distance_m'][1] >= c['far_m']:
        raise ValueError('View distances must lie safely inside camera operating range')
    if not 0 <= config['labels']['min_visible_fraction'] <= 1 or not 0 < config['labels']['bbox_min_iou'] <= 1:
        raise ValueError('Invalid visible fraction or box IoU threshold')
    for key in ('training_imgsz', 'target_min_side_px', 'difficult_view_every', 'planning_width'):
        if type(cap[key]) is not int or cap[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if cap['planning_width'] < 160 or cap['training_imgsz'] % 32:
        raise ValueError('Use planning_width >= 160 and training_imgsz divisible by 32')
    if not 0 <= cap['target_margin_fraction'] < .4 or not 0 < cap['target_min_visible_fraction'] <= 1:
        raise ValueError('Invalid target framing policy')
    if cap['target_min_side_px'] >= cap['training_imgsz'] * (1-2*cap['target_margin_fraction']):
        raise ValueError('Target minimum size exceeds the usable training image')
    from .planning import validate_class_quotas
    validate_class_quotas(cap)
    for key in ('depth_support_fraction', 'patch_support_fraction'):
        if not 0 < config['labels'][key] <= 1:
            raise ValueError(f'{key} must be in (0, 1]')
    if not 0 < config['labels']['depth_support_tolerance_m'] <= .08:
        raise ValueError('Use a depth support tolerance between 0 and 0.08 m')
    if not isinstance(cap['require_visibility_check'], bool):
        raise ValueError('require_visibility_check must be boolean')
    if cap['require_visibility_check'] and not cap['require_depth_check']:
        raise ValueError('Visibility checks require synchronized sensor depth')
    _validate_candidates(config)
    for key in ('seeds', 'empty_seeds', 'rocks_only_seeds'):
        seeds = config['dataset'][key]
        if (not isinstance(seeds, list) or any(type(s) is not int or s < 0 for s in seeds)
                or len(set(seeds)) != len(seeds)):
            raise ValueError(f'{key} must contain unique nonnegative integer seeds')
    if set(config['dataset']['empty_seeds']) & set(config['dataset']['rocks_only_seeds']):
        raise ValueError('Empty and rocks-only seeds must be disjoint')
    quotas = cap.get('target_class_views', {})
    mixed = (set(config['dataset']['seeds'])-set(config['dataset']['empty_seeds'])
             -set(config['dataset']['rocks_only_seeds']))
    if quotas and mixed:
        candidates = config['candidates']
        available = set(candidates['categories']) if candidates['enabled'] else set()
        for name, choices in [('pale_soil_patch',[('silica_soil','silica_count'),('pale_lookalike','pale_lookalike_count')]),
                              ('yellow_crystal_rock',[('sulfur_rock','sulfur_count')])]:
            if quotas.get(name,0) and not any(category in available and candidates[key][1]>0 for category,key in choices):
                raise ValueError(f'target_class_views requests {name}, but no such candidates are enabled')
    return config


def _validate_candidates(config):
    cand = config.get('candidates')
    if not cand:
        return
    known = set(APPEARANCE_CLASSES)
    if not isinstance(cand.get('enabled'), bool) or not isinstance(cand.get('exaggerated_spots'), bool):
        raise ValueError('Candidate toggles must be booleans')
    categories = cand.get('categories')
    if not isinstance(categories, list) or set(categories) - known:
        raise ValueError('Unsupported three-class appearance category')
    for key in ('silica_count', 'pale_lookalike_count', 'sulfur_count', 'pale_stone_count',
                'cheyava_count', 'veined_lookalike_count'):
        lo, hi = cand[key]
        if not isinstance(lo, int) or not isinstance(hi, int) or not 0 <= lo <= hi <= 16:
            raise ValueError(f'Invalid candidate count {key}')
    for key in ('silica_size_m', 'sulfur_size_m', 'cheyava_size_m'):
        lo, hi = cand[key]
        if not 0 < float(lo) <= float(hi):
            raise ValueError(f'Invalid candidate size {key}')
    if not float(cand['spot_scale']) > 0:
        raise ValueError('spot_scale must be positive')
    if not 0.001 <= float(cand['slab_surface_height_m']) <= 0.05:
        raise ValueError('slab_surface_height_m must be between 0.001 and 0.05 m')
    if not 0 < float(cand['slab_embed_m']) <= 0.1 or not 0 < float(cand['slab_edge_blend_m']) <= 0.2:
        raise ValueError('Invalid mudstone embedding depth or edge blend')


def stable_numbers(value):
    # Sub-nanometre differences between Windows/Linux BLAS must not change IDs.
    if isinstance(value, float):
        rounded = round(value, 10)
        return 0.0 if rounded == 0 else rounded
    if isinstance(value, dict):
        return {k:stable_numbers(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [stable_numbers(v) for v in value]
    return value


def canonical(value):
    return json.dumps(stable_numbers(value), sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stable_numbers(value), indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


def split_for_seed(seed):
    # Independent of scene count/config: adding seeds cannot move an existing group.
    bucket = int(hashlib.sha256(f'mars-scene:{seed}'.encode()).hexdigest()[:8], 16) % 10
    return 'train' if bucket < 6 else ('val' if bucket < 8 else 'test')


def png(path, pixels):
    """Lossless PNG, RGB8 or grayscale8, without an imaging dependency."""
    pixels = np.asarray(pixels, dtype=np.uint8)
    h, w = pixels.shape[:2]
    color = 2 if pixels.ndim == 3 else 0
    def chunk(tag, data):
        return struct.pack('!I', len(data)) + tag + data + struct.pack('!I', zlib.crc32(tag + data) & 0xffffffff)
    data = b''.join(b'\0' + row.tobytes() for row in pixels)
    Path(path).write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', w, h, 8, color, 0, 0, 0))
                           + chunk(b'IDAT', zlib.compress(data, 6)) + chunk(b'IEND', b''))


def rotation(rpy):
    from rover_training.geometry import rpy_to_matrix
    return np.asarray(rpy_to_matrix(*rpy))


def quaternion(rpy):
    r, p, y = (v / 2 for v in rpy)
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return [sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy]
