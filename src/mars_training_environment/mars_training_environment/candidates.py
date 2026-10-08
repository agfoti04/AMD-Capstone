"""Sampling-site appearances synthesized from NASA photograph references.

The files in Mars_Gazebo_Reference_Pack are appearance references. This module
does not copy those pixels onto the ground. Patch sizes, crystal grain, and
spot diameters are simulation choices and are stored on each record.

Assigned categories are ground truth for training. They are not chemical
measurements, and they are not written into the RGB image filenames.
"""
import math
from pathlib import Path

import numpy as np

from .common import png, rotation

# Ground-mask values. The mask is a scene file, not a camera image.
MASK_IDS = {'silica_soil': 1, 'pale_lookalike': 2}
NOTES = [
    'Sampling candidates are visual analogues. Category tags are assigned ground truth, not chemical measurements.',
    'Silica patch size is a simulation choice. The 20 cm wheel track in PIA09491 is not used as the patch size.',
    'Sulfur yellow is limited to a fracture face and is a synthetic analogue, not a yellow soil bed.',
    'Cheyava spots use the configured physical scale unless exaggerated_spots is set, which tags an exaggerated analogue.',
    'PIXL display colors are not used as surface colors.',
    'Pale soil, intact pale stones, and veined mudstone without spots are lookalike controls.',
]
MESH_CLASSES = {
    'sulfur_rock': 'cand_sulfur',
    'pale_stone': 'cand_pale_stone',
    'cheyava_rock': 'cand_slab',
    'veined_lookalike': 'cand_slab',
}


def candidate_shapes():
    """Stable original meshes. Textures are rewritten per scene."""
    sulfur_v, sulfur_f, sulfur_uv = fracture_mesh()
    pale_v, pale_f = _sphere(2)
    chey_v, chey_f, chey_uv = slab_mesh()
    sulfur_note = ('Synthetic analogue of a fractured sulfur-bearing stone. '
                   'References PIA26309, PIA26307, and PIA26210. Not a copy of those photographs.')
    pale_note = ('Intact pale stone. Reference PIA26210. Yellow interior is not visible.')
    slab_note = ('Synthetic mudstone slab. References PIA26368 and PIA26401. '
                 'Rover abrasion patches and the drill hole are not reproduced.')
    return [
        ('cand_sulfur', sulfur_v, sulfur_f, sulfur_uv, None, sulfur_note, 'Apache-2.0', None),
        ('cand_pale_stone', pale_v, pale_f, _planar_uv(pale_v), None, pale_note, 'Apache-2.0', None),
        ('cand_slab', chey_v, chey_f, chey_uv, None, slab_note, 'Apache-2.0', None),
    ]


def fracture_mesh():
    """A stone with one flattened face. That face is the high-U side of the texture."""
    v, f = _sphere(7)
    v = np.array(v, dtype=float)
    v[v[:, 0] > 0.12, 0] = 0.12
    v -= (v.min(axis=0) + v.max(axis=0)) / 2
    v /= np.ptp(v, axis=0).max()
    return v, f, _planar_uv(v)


def slab_mesh(seed=21, aspect=0.62, thickness=0.045):
    """Thin mudstone prism. The longest side is 1 m before the scene scales it."""
    rng = np.random.RandomState(seed)
    count = 14
    angles = np.linspace(0, 2 * math.pi, count, endpoint=False)
    radii = rng.uniform(0.82, 1.0, count)
    poly = np.column_stack([np.cos(angles) * radii, np.sin(angles) * radii * aspect])
    poly -= poly.mean(axis=0)
    n = len(poly)
    top = np.column_stack([poly, np.full(n, thickness / 2)])
    bottom = np.column_stack([poly, np.full(n, -thickness / 2)])
    vertices = np.vstack([top, bottom])
    faces = []
    for i in range(1, n - 1):
        faces.append([0, i, i + 1])
        faces.append([n, n + i + 1, n + i])
    for i in range(n):
        j = (i + 1) % n
        faces.extend([[i, j, n + j], [i, n + j, n + i]])
    vertices[:, :2] /= np.ptp(vertices[:, :2], axis=0).max()
    vertices[:, 2] /= np.ptp(vertices[:, :2], axis=0).max()
    return vertices, np.asarray(faces, dtype=int), _planar_uv(vertices)


def add_candidates(rng, config, field, driveable, assets, root, objects, regions):
    """Append mesh candidates and ground regions. Empty scenes call this never."""
    selected = set(config['candidates']['categories'])
    by_id = {asset['id']: asset for asset in assets}
    occupied = [(np.asarray(obj['xyz'][:2], dtype=float), obj['radius_m']) for obj in objects]
    for category, key, size_key in (
            ('silica_soil', 'silica_count', 'silica_size_m'),
            ('pale_lookalike', 'pale_lookalike_count', 'silica_size_m')):
        for index in range(_count(rng, config, key, category, selected)):
            longest = float(rng.uniform(*config['candidates'][size_key]))
            polygon = _blob(rng, longest)
            radius = float(np.linalg.norm(polygon - polygon.mean(axis=0), axis=1).max())
            xy = _place(rng, field, driveable, config, occupied, radius)
            polygon = polygon + xy
            z = float(field.height(xy[0], xy[1]))
            regions.append({
                'name': f'{category}_{index:03d}',
                'asset_id': category,
                'class_name': category,
                'source': 'ground_region',
                'xyz': [float(xy[0]), float(xy[1]), z],
                'polygon_xy': np.round(polygon, 5).tolist(),
                'longest_m': longest,
                'radius_m': radius,
                'dust_cover': float(rng.uniform(0.15, 0.55)),
                'mask_id': MASK_IDS[category],
                'references': ['PIA09491', 'PIA09403', 'PIA11758'],
                'simulation_choice': 'Patch size and dust cover are chosen for the rover camera, not measured from the 20 cm track.',
            })
            occupied.append((xy, radius))
    burial_lo, burial_hi = config['rocks']['burial_fraction']
    for category, key, size_key in (
            ('sulfur_rock', 'sulfur_count', 'sulfur_size_m'),
            ('pale_stone', 'pale_stone_count', 'sulfur_size_m'),
            ('cheyava_rock', 'cheyava_count', 'cheyava_size_m'),
            ('veined_lookalike', 'veined_lookalike_count', 'cheyava_size_m')):
        asset = by_id[MESH_CLASSES[category]]
        vertices = np.asarray(json_vertices(root, asset), dtype=float)
        for index in range(_count(rng, config, key, category, selected)):
            scale = float(rng.uniform(*config['candidates'][size_key]))
            tilt = math.radians(12 if category in ('cheyava_rock', 'veined_lookalike') else 18)
            rpy = [float(rng.uniform(-tilt, tilt)), float(rng.uniform(-tilt, tilt)), float(rng.uniform(-math.pi, math.pi))]
            radius_guess = scale * 0.75
            xy = _place(rng, field, driveable, config, occupied, radius_guess)
            points = vertices @ rotation(rpy).T * scale
            radius = float(np.linalg.norm(points[:, :2], axis=1).max())
            surface = field.height(points[:, 0] + xy[0], points[:, 1] + xy[1])
            support = float(np.max(surface - points[:, 2]))
            burial = float(rng.uniform(burial_lo, burial_hi))
            z = support - burial * float(np.ptp(points[:, 2]))
            exaggerated = bool(config['candidates']['exaggerated_spots']) and category == 'cheyava_rock'
            if category == 'pale_stone':
                references = ['PIA26210']
            elif category == 'sulfur_rock':
                references = ['PIA26309', 'PIA26307', 'PIA26210']
            else:
                references = ['PIA26368', 'PIA26401']
            obj = {
                'name': f'{category}_{index:03d}',
                'asset_id': asset['id'],
                'class_name': category,
                'source': 'mesh',
                'xyz': [float(xy[0]), float(xy[1]), float(z)],
                'rpy': rpy,
                'scale': scale,
                'size_class': 'candidate',
                'radius_m': radius,
                'burial_fraction': burial,
                'burial_m': float(support - z),
                'color': [1.0, 1.0, 1.0],
                'synthetic_analogue': category in ('sulfur_rock', 'cheyava_rock', 'veined_lookalike'),
                'exaggerated_analogue': exaggerated,
                'references': references,
            }
            objects.append(obj)
            occupied.append((xy, radius))


def write_candidate_materials(bundle, scene_folder, objects, dust_bgr, seed, config):
    """Per-object textures so a shared mesh can still show a fracture or spots."""
    rng = np.random.RandomState(8100 + int(seed))
    dust_rgb = np.asarray(dust_bgr, dtype=np.float32)[::-1]
    scale = float(config['candidates']['spot_scale'])
    dest = Path(scene_folder) / 'materials'
    written = False
    for obj in objects:
        kind = obj.get('class_name')
        if kind not in MESH_CLASSES:
            continue
        if not written:
            dest.mkdir(parents=True, exist_ok=True)
            written = True
        if kind in ('sulfur_rock', 'pale_stone'):
            albedo, rough = stone_texture(rng, 256, fracture=kind == 'sulfur_rock', dust_rgb=dust_rgb)
        else:
            spots = kind == 'cheyava_rock'
            factor = scale * (8.0 if obj.get('exaggerated_analogue') else 1.0)
            albedo, rough = mudstone_texture(rng, 768, spots=spots, spot_scale=factor, span_m=1.0, dust_rgb=dust_rgb)
        stem = obj['name']
        png(dest / f'{stem}_albedo.png', albedo)
        png(dest / f'{stem}_roughness.png', rough)
        png(dest / f'{stem}_normal.png', _normal(albedo.shape[0], rng, rough_bias=kind == 'sulfur_rock'))
        rel = dest.relative_to(bundle).as_posix()
        obj['material_maps'] = {
            'albedo': f'{rel}/{stem}_albedo.png',
            'normal': f'{rel}/{stem}_normal.png',
            'roughness': f'{rel}/{stem}_roughness.png',
        }


def paint_ground_regions(albedo, normal, rough, regions, size_m, rng):
    """Blend pale exposures into the existing ground. Returns a hard category mask."""
    import cv2
    albedo = np.asarray(albedo, dtype=np.float32)
    height, width = albedo.shape[:2]
    mask = np.zeros((height, width), np.uint8)
    if normal is not None:
        normal = np.asarray(normal, dtype=np.float32)
    for region in regions:
        polygon = np.asarray(region['polygon_xy'], dtype=float)
        col, row = world_to_pixels(polygon, size_m, width)
        pts = np.round(np.column_stack([col, row])).astype(np.int32)
        hard = np.zeros((height, width), np.uint8)
        cv2.fillPoly(hard, [pts], 255)
        if not hard.any():
            continue
        sigma = max(1.0, 0.06 / size_m * width)
        soft = cv2.GaussianBlur(hard, (0, 0), sigma * (1.8 if region['class_name'] == 'pale_lookalike' else 0.8))
        alpha = soft.astype(np.float32) / 255
        if region['class_name'] == 'silica_soil':
            chunk = cv2.GaussianBlur(rng.random((height, width)).astype(np.float32), (0, 0), sigma * 2.4)
            alpha = np.clip(alpha * (0.35 + 1.15 * chunk) - 0.28, 0, 1)
            color = np.array([188, 206, 216], np.float32)
        else:
            color = np.array([176, 196, 208], np.float32)
        cover = float(region.get('dust_cover', 0.3))
        color = color * (1 - cover * 0.45) + albedo.mean(axis=(0, 1)) * cover * 0.45
        alpha = alpha[:, :, None] * (1 - cover * 0.35)
        albedo = albedo * (1 - alpha) + color * alpha
        if region['class_name'] == 'silica_soil':
            _embed_stones(rng, albedo, hard, size_m)
            if normal is not None:
                noise = rng.normal(0, 0.18, (height, width, 2)).astype(np.float32)
                normal[:, :, :2] += noise * (hard[:, :, None] / 255)
        mask[hard > 128] = region['mask_id']
    if normal is not None:
        normal /= np.maximum(np.linalg.norm(normal, axis=2, keepdims=True), 1e-6)
    if rough is not None and mask.any():
        rough = np.asarray(rough, dtype=np.float32)
        rough = np.where(mask > 0, np.maximum(rough, 210), rough)
    return albedo, normal, rough, mask


def paint_procedural_ground(folder, regions, size_m, seed):
    """Paint the procedural fallback texture. Scanned composites are painted earlier."""
    import cv2
    folder = Path(folder)
    image = cv2.imread(str(folder / 'ground.png'), cv2.IMREAD_COLOR)
    if image is None or not regions:
        return None
    rng = np.random.RandomState(7000 + int(seed))
    painted, _, _, mask = paint_ground_regions(image, None, None, regions, size_m, rng)
    cv2.imwrite(str(folder / 'ground.png'), np.clip(painted, 0, 255).astype(np.uint8))
    png(folder / 'candidate_mask.png', mask)
    return mask


def world_to_pixels(xy, size_m, pixels):
    """Match terrain UVs. ogre2 samples the albedo with OpenGL's bottom-left v=0.

    u = (x + size/2) / size and v = (y + size/2) / size, as in terrain.dae.
    Image row 0 is the top of the file, so v is flipped into that row.
    """
    xy = np.asarray(xy, dtype=float)
    u = (xy[..., 0] + size_m / 2) / size_m
    v = (xy[..., 1] + size_m / 2) / size_m
    return u * (pixels - 1), (1.0 - v) * (pixels - 1)


def stone_texture(rng, pixels, fracture, dust_rgb):
    """Pale weathered stone. Yellow crystals stay on the high-U fracture face."""
    pale = np.array([176, 166, 152], np.float32) * 0.72 + np.asarray(dust_rgb, dtype=np.float32) * 0.28
    image = np.broadcast_to(pale, (pixels, pixels, 3)).copy()
    image += rng.normal(0, 9, image.shape)
    rough = np.full((pixels, pixels), 232, np.float32)
    if fracture:
        edge = pixels * 0.62
        start = edge + rng.normal(0, pixels * 0.04, pixels)
        cols = np.arange(pixels)
        face = cols[None, :] > start[:, None]
        crystal = rng.randint(0, 3, (pixels, pixels))
        yellows = np.array([[198, 178, 74], [220, 198, 86], [170, 154, 62]], np.float32)
        painted = yellows[crystal] + rng.normal(0, 12, (pixels, pixels, 1))
        crystals = face & (rng.random((pixels, pixels)) > 0.34)
        image[crystals] = np.clip(painted, 0, 255)[crystals]
        rough[crystals] = 190
    return np.clip(image, 0, 255).astype(np.uint8), np.clip(rough, 0, 255).astype(np.uint8)


def mudstone_texture(rng, pixels, spots, spot_scale, span_m, dust_rgb):
    """Rust mudstone, pale veins, and optional pale spots with dark rims."""
    rust = np.array([150, 112, 86], np.float32) * 0.8 + np.asarray(dust_rgb, dtype=np.float32) * 0.2
    image = np.broadcast_to(rust, (pixels, pixels, 3)).copy()
    image += rng.normal(0, 8, image.shape)
    image = np.clip(image, 0, 255).astype(np.uint8)
    vein = np.array([198, 186, 168], np.float32)
    for _ in range(int(rng.randint(5, 9))):
        x, y = float(rng.uniform(0, pixels)), float(rng.uniform(0, pixels))
        angle = float(rng.uniform(0, math.tau))
        points = []
        for _step in range(int(rng.randint(24, 48))):
            angle += float(rng.uniform(-0.55, 0.55))
            step = pixels * float(rng.uniform(0.01, 0.028))
            x = float(np.clip(x + math.cos(angle) * step, 0, pixels - 1))
            y = float(np.clip(y + math.sin(angle) * step, 0, pixels - 1))
            points.append([int(x), int(y)])
        width = max(2, int(rng.uniform(0.006, 0.016) / span_m * pixels))
        _polyline(image, points, vein, width)
    if spots:
        ppm = pixels / span_m
        count = int(rng.randint(18, 32) if spot_scale > 2 else rng.randint(45, 80))
        for _ in range(count):
            radius_m = float(rng.uniform(0.003, 0.007)) * float(spot_scale)
            radius = max(1, int(round(radius_m * ppm)))
            cx, cy = int(rng.randint(radius, pixels - radius)), int(rng.randint(radius, pixels - radius))
            _disk(image, cx, cy, radius, (92, 74, 60))
            _disk(image, cx, cy, max(1, int(round(radius * 0.55))), (232, 226, 214))
    rough = np.full((pixels, pixels), 226, np.uint8)
    return image, rough


def drape_mesh(field, polygon, step=0.05, lift=0.008):
    """A surface just above the visual terrain, used only to label a ground region."""
    poly = np.asarray(polygon, dtype=float)
    minx, miny = poly.min(axis=0) - step
    maxx, maxy = poly.max(axis=0) + step
    xs = np.arange(minx, maxx + step * 0.5, step)
    ys = np.arange(miny, maxy + step * 0.5, step)
    if len(xs) < 2 or len(ys) < 2:
        xs = np.linspace(minx, maxx, 2)
        ys = np.linspace(miny, maxy, 2)
    xx, yy = np.meshgrid(xs, ys)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    inside = _inside(poly, points).reshape(len(ys), len(xs))
    index = -np.ones(inside.shape, dtype=int)
    vertices = []
    ny, nx = inside.shape
    for j in range(ny):
        for i in range(nx):
            neighborhood = inside[max(0, j - 1):j + 2, max(0, i - 1):i + 2]
            if not neighborhood.any():
                continue
            index[j, i] = len(vertices)
            vertices.append([xs[i], ys[j], float(field.height(xs[i], ys[j])) + lift])
    faces = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            center = np.array([[(xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2]])
            if not _inside(poly, center)[0]:
                continue
            corners = [index[j, i], index[j, i + 1], index[j + 1, i + 1], index[j + 1, i]]
            if min(corners) < 0:
                continue
            faces.extend([[corners[0], corners[1], corners[2]], [corners[0], corners[2], corners[3]]])
    if not faces:
        raise ValueError('Ground region produced an empty label mesh')
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)


def _sphere(index):
    from .meshes import sphere_mesh
    return sphere_mesh(index)


def _planar_uv(vertices):
    xy = np.asarray(vertices)[:, :2]
    return (xy - xy.min(axis=0)) / np.maximum(np.ptp(xy, axis=0), 1e-9)


def _count(rng, config, key, category, selected):
    if category not in selected:
        return 0
    lo, hi = config['candidates'][key]
    return int(rng.randint(lo, hi + 1))


def _blob(rng, longest):
    count = int(rng.randint(9, 13))
    angles = np.linspace(0, 2 * math.pi, count, endpoint=False) + rng.uniform(-0.2, 0.2, count)
    radii = rng.uniform(0.55, 1.0, count)
    aspect = float(rng.uniform(0.45, 0.85))
    poly = np.column_stack([np.cos(angles) * radii, np.sin(angles) * radii * aspect])
    yaw = float(rng.uniform(0, math.tau))
    rotation_xy = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    poly = poly @ rotation_xy.T
    current = max(np.linalg.norm(poly[i] - poly, axis=1).max() for i in range(len(poly)))
    return poly / current * longest


def _place(rng, field, driveable, config, occupied, radius):
    size = field.size
    clearance = config['rocks']['clearance_m']
    spawn = config['terrain']['spawn_clearance_m']
    limit = math.tan(math.radians(config['terrain']['max_slope_deg']))
    for _ in range(800):
        xy = rng.uniform(-size * 0.40, size * 0.40, 2)
        if max(abs(xy)) + radius > size / 2 - 0.5:
            continue
        if np.linalg.norm(xy) < spawn + radius + 0.25:
            continue
        if any(np.linalg.norm(xy - other) < radius + other_radius + clearance for other, other_radius in occupied):
            continue
        if np.linalg.norm(driveable.gradient(*xy)) > limit:
            continue
        return xy
    raise ValueError('Placement constraints could not be met; reduce candidate count or sizes')


def json_vertices(root, asset):
    import json
    return json.loads((Path(root) / asset['geometry']).read_text())['vertices']


def _embed_stones(rng, albedo, hard, size_m):
    ys, xs = np.nonzero(hard > 128)
    if len(xs) < 20:
        return
    area = len(xs) * (size_m / hard.shape[1]) ** 2
    picks = rng.choice(len(xs), size=min(len(xs), max(3, int(area * 5))), replace=False)
    stone = np.array([48, 56, 64], np.float32)
    for pick in np.atleast_1d(picks):
        albedo[int(ys[pick]), int(xs[pick])] = stone


def _normal(pixels, rng, rough_bias=False):
    image = np.zeros((pixels, pixels, 3), np.float32)
    image[:, :, 2] = 1
    if rough_bias:
        image[:, :, 0] += rng.normal(0, 0.15, (pixels, pixels))
    image /= np.maximum(np.linalg.norm(image, axis=2, keepdims=True), 1e-6)
    return np.clip((image + 1) * 127.5, 0, 255).astype(np.uint8)


def _polyline(image, points, color, width):
    import cv2
    cv2.polylines(image, [np.asarray(points, np.int32)], False, [int(c) for c in color], width, cv2.LINE_AA)


def _disk(image, x, y, radius, color):
    import cv2
    cv2.circle(image, (int(x), int(y)), int(radius), [int(c) for c in color], -1, cv2.LINE_8)


def _inside(poly, points):
    x = points[:, 0]
    y = points[:, 1]
    x0, y0 = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    crosses = ((y0 > y[:, None]) != (y1 > y[:, None]))
    x_at = x0 + (y[:, None] - y0) * (x1 - x0) / (y1 - y0 + 1e-15)
    return (crosses & (x[:, None] < x_at)).sum(axis=1) % 2 == 1
