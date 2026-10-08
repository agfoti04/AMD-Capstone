"""Scanned Poly Haven rocks, original procedural rocks, and existing Fuel rocks."""
import json
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .common import digest, png, write_json


def sphere_mesh(index, rings=9, sectors=18):
    rng = np.random.RandomState(4000 + index)
    proportions = rng.uniform([0.65, 0.5, 0.35], [1.3, 1.05, 1.05])
    phases = rng.uniform(0, 2 * math.pi, 6)
    vertices = [[0, 0, proportions[2]]]
    for ring in range(1, rings):
        theta = math.pi * ring / rings
        for sector in range(sectors):
            phi = 2 * math.pi * sector / sectors
            radial = (1 + 0.15 * math.sin(3*phi + phases[0]) * math.sin(theta)**2
                      + 0.10 * math.cos(5*theta + 2*phi + phases[1]) * math.sin(theta)
                      + 0.07 * math.sin(7*phi + phases[2]) * math.sin(theta)**3)
            point = np.array([math.sin(theta)*math.cos(phi), math.sin(theta)*math.sin(phi), math.cos(theta)])
            point *= radial * proportions
            # Flat slabs, angular blocks and irregular rounded boulders.
            if index % 4 == 0:
                point[2] = max(-0.45*proportions[2], min(0.55*proportions[2], point[2]))
            if index % 4 == 1:
                point[:2] = np.sign(point[:2]) * np.abs(point[:2])**0.8
            point[0] += 0.12 * point[2] * math.sin(phases[3])
            vertices.append(point.tolist())
    vertices.append([0, 0, -proportions[2]])
    if index % 4 == 0:
        vertices[0][2], vertices[-1][2] = 0.55*proportions[2], -0.45*proportions[2]
    faces = []
    for j in range(sectors):
        faces.append([0, 1+j, 1+(j+1)%sectors])
    for i in range(rings-2):
        for j in range(sectors):
            a, b = 1+i*sectors+j, 1+i*sectors+(j+1)%sectors
            faces.extend([[a, a+sectors, b], [b, a+sectors, b+sectors]])
    end = len(vertices)-1
    start = 1+(rings-2)*sectors
    for j in range(sectors):
        faces.append([start+j, end, start+(j+1)%sectors])
    v = np.asarray(vertices, dtype=float)
    v -= (v.min(axis=0) + v.max(axis=0)) / 2
    v /= np.ptp(v, axis=0).max()
    return v, np.asarray(faces, dtype=int)


def read_collada(path):
    """Read position-indexed triangles/polylist from our existing untransformed Fuel DAEs."""
    ns = {'c': 'http://www.collada.org/2005/11/COLLADASchema'}
    root = ET.parse(path).getroot()
    sources = {}
    for source in root.findall('.//c:mesh/c:source', ns):
        arr = source.find('c:float_array', ns)
        acc = source.find('c:technique_common/c:accessor', ns)
        if arr is not None and acc is not None:
            sources[source.get('id')] = np.fromstring(arr.text, sep=' ').reshape(-1, int(acc.get('stride', '1')))
    vertices_map = {v.get('id'): v.find('c:input[@semantic="POSITION"]', ns).get('source')[1:]
                    for v in root.findall('.//c:mesh/c:vertices', ns)}
    vv, ff = [], []
    for mesh in root.findall('.//c:mesh', ns):
        for primitive in list(mesh):
            kind = primitive.tag.split('}')[-1]
            if kind not in ('triangles', 'polylist'):
                continue
            inputs = primitive.findall('c:input', ns)
            inp = next(x for x in inputs if x.get('semantic') == 'VERTEX')
            data = sources[vertices_map[inp.get('source')[1:]]][:, :3]
            offset = sum(len(v) for v in vv)
            vv.append(data)
            stride = max(int(x.get('offset', '0')) for x in inputs) + 1
            indices = np.fromstring(primitive.find('c:p', ns).text, sep=' ', dtype=int).reshape(-1, stride)[:, int(inp.get('offset', '0'))]
            counts = [3] * (len(indices)//3) if kind == 'triangles' else [int(n) for n in primitive.find('c:vcount', ns).text.split()]
            start = 0
            for count in counts:
                poly = indices[start:start+count]
                ff.extend([[offset+poly[0], offset+poly[j], offset+poly[j+1]] for j in range(1, count-1)])
                start += count
    if not vv or not ff:
        raise ValueError(f'No triangles in {path}')
    v = np.concatenate(vv)
    unit = root.find('c:asset/c:unit', ns)
    v *= float(unit.get('meter', '1')) if unit is not None else 1
    up = root.findtext('c:asset/c:up_axis', 'Y_UP', ns)
    if up == 'Y_UP':
        v = v[:, [0, 2, 1]] * [1, -1, 1]
    elif up != 'Z_UP':
        raise ValueError('Unsupported imported up axis')
    v -= (v.max(axis=0) + v.min(axis=0))/2
    v /= np.ptp(v, axis=0).max()
    return v, np.asarray(ff, dtype=int)


def write_dae(path, vertices, faces, uv=None, texture=None):
    """Metres, Z-up, one untransformed mesh; same triangles feed annotation."""
    v, f = np.asarray(vertices), np.asarray(faces)
    normals = np.zeros_like(v)
    face_n = np.cross(v[f[:, 1]]-v[f[:, 0]], v[f[:, 2]]-v[f[:, 0]])
    for col in range(3):
        np.add.at(normals, f[:, col], face_n)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    if uv is None:
        uv = (v[:, :2] - v[:, :2].min(axis=0)) / np.maximum(np.ptp(v[:, :2], axis=0), 1e-9)
    def source(name, data, params):
        data = np.asarray(data)
        return (f'<source id="{name}"><float_array id="{name}-array" count="{data.size}">'
                + ' '.join(f'{x:.9g}' for x in data.flat) + '</float_array><technique_common>'
                + f'<accessor source="#{name}-array" count="{len(data)}" stride="{data.shape[1]}">'
                + ''.join(f'<param name="{p}" type="float"/>' for p in params)
                + '</accessor></technique_common></source>')
    material = ''
    binding = ''
    if texture:
        material = f'''<library_images><image id="texture"><init_from>{texture}</init_from></image></library_images>
<library_effects><effect id="effect"><profile_COMMON>
<newparam sid="surface"><surface type="2D"><init_from>texture</init_from></surface></newparam>
<newparam sid="sampler"><sampler2D><source>surface</source></sampler2D></newparam>
<technique sid="common"><phong><ambient><color>0.45 0.45 0.45 1</color></ambient>
<diffuse><texture texture="sampler" texcoord="UVMap"/></diffuse><specular><color>0 0 0 1</color></specular></phong></technique>
</profile_COMMON></effect></library_effects><library_materials><material id="material"><instance_effect url="#effect"/></material></library_materials>'''
        binding = '<bind_material><technique_common><instance_material symbol="mat" target="#material"><bind_vertex_input semantic="UVMap" input_semantic="TEXCOORD" input_set="0"/></instance_material></technique_common></bind_material>'
    text = ('<?xml version="1.0"?><COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">'
            '<asset><unit name="meter" meter="1"/><up_axis>Z_UP</up_axis></asset>' + material
            + '<library_geometries><geometry id="geometry"><mesh>'
            + source('positions', v, 'XYZ') + source('normals', normals, 'XYZ') + source('uv', uv, 'ST')
            + '<vertices id="vertices"><input semantic="POSITION" source="#positions"/></vertices>'
            + f'<triangles count="{len(f)}" material="mat"><input semantic="VERTEX" source="#vertices" offset="0"/>'
            + '<input semantic="NORMAL" source="#normals" offset="1"/><input semantic="TEXCOORD" source="#uv" offset="2" set="0"/><p>'
            + ' '.join(str(int(i)) for i in np.repeat(f.reshape(-1),3)) + '</p></triangles></mesh></geometry></library_geometries>'
            + '<library_visual_scenes><visual_scene id="scene"><node id="model"><instance_geometry url="#geometry">'
            + binding + '</instance_geometry></node></visual_scene></library_visual_scenes><scene><instance_visual_scene url="#scene"/></scene></COLLADA>')
    Path(path).write_text(text, encoding='utf-8')


MAPS = ('albedo', 'normal', 'roughness')


def _prepared(cache, kind, ident):
    if cache is None:
        raise ValueError(f'{ident} needs the asset cache; pass --asset-cache')
    folder = Path(cache)/'prepared'/kind/ident
    if not (folder/'meta.json').exists():
        raise FileNotFoundError(f'{ident} is not prepared in {cache}; run: mars_assets all --cache {cache}')
    return folder, json.loads((folder/'meta.json').read_text())


def build_library(directory, config, cache=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    assets = []
    # (id, vertices, faces, uv or None, collision (v, f) or None, provenance, license, texture folder or None)
    shapes = [(f'proc_{i:02d}', *sphere_mesh(i), None, None, 'Original procedural geometry', 'Apache-2.0', None)
              for i in range(config['procedural_count'])]
    if config['include_existing']:
        from rover_training.assets import load_assets, models_dir
        for asset in load_assets():
            v, f = read_collada(models_dir() / asset['id'] / asset['mesh'])
            shapes.append((asset['id'], v, f, None, None, asset['fuel_url'], 'CC-BY-4.0', None))
    for ident in config['scanned_ids']:
        source, meta = _prepared(cache, 'rocks', ident)
        visual = json.loads((source/'visual.json').read_text())
        collision = json.loads((source/'collision.json').read_text())
        shapes.append((ident, np.asarray(visual['vertices'], dtype=float), np.asarray(visual['faces'], dtype=int),
                       np.asarray(visual['uv'], dtype=float),
                       (np.asarray(collision['vertices'], dtype=float), np.asarray(collision['faces'], dtype=int)),
                       meta['page'], meta['license'], source))
    from .candidates import candidate_shapes
    shapes.extend(candidate_shapes())
    for index, (ident, v, f, uv, collision, provenance, license_id, source) in enumerate(shapes):
        folder = directory / ident
        folder.mkdir(exist_ok=True)
        if source is None:
            rng = np.random.RandomState(7300+index)
            yy, xx = np.mgrid[0:128, 0:128]
            base = np.array([[111,88,67], [124,120,112], [151,120,86], [84,78,74]][index % 4])
            grain = rng.normal(0, 8, (128, 128)) + 12*np.sin(xx*.13 + 2*np.sin(yy*.09)) + 6*np.cos(yy*.4)
            png(folder / 'albedo.png', np.clip(base + grain[:, :, None], 0, 255))
            textures = {'albedo': f'assets/{ident}/albedo.png'}
        else:
            for key in MAPS:
                shutil.copyfile(source/f'{key}.jpg', folder/f'{key}.jpg')
            textures = {key: f'assets/{ident}/{key}.jpg' for key in MAPS}
        # The visual triangles are also the label mesh. Procedural and Fuel rocks are
        # a few hundred triangles, so their collision uses the same triangles.
        write_dae(folder / 'visual.dae', v, f, uv=uv)
        cv, cf = collision if collision is not None else (v, f)
        write_dae(folder / 'collision.dae', cv, cf)
        geometry = {'vertices': v.tolist(), 'faces': f.tolist()}
        if uv is not None:
            geometry['uv'] = np.asarray(uv).tolist()
        write_json(folder / 'geometry.json', geometry)
        record = {'id': ident, 'held_out': ident in config['held_out_ids'], 'source': provenance,
                  'license': license_id, 'textures': textures,
                  'mesh': f'assets/{ident}/visual.dae', 'collision': f'assets/{ident}/collision.dae',
                  'geometry': f'assets/{ident}/geometry.json', 'sha256': digest(geometry),
                  'bounds_min': v.min(axis=0).tolist(), 'bounds_max': v.max(axis=0).tolist(),
                  'triangles': len(f), 'collision_triangles': len(cf)}
        if ident.startswith('cand_'):
            # Same shape may appear in every split. Scanned rocks stay split-isolated.
            record['shared_shape'] = True
            record['held_out'] = False
        assets.append(record)
    unknown = set(config['held_out_ids']) - {a['id'] for a in assets}
    if unknown or not any(a['held_out'] for a in assets) or all(a['held_out'] for a in assets):
        raise ValueError(f'Invalid evaluation mesh reservation: {sorted(unknown)}')
    write_json(directory / 'catalog.json', assets)
    return assets


def build_ground(directory, ids, cache=None):
    """Copy prepared ground texture sets; an empty list keeps the procedural ground."""
    directory = Path(directory)
    ground = []
    for ident in ids:
        source, meta = _prepared(cache, 'ground', ident)
        folder = directory/ident
        folder.mkdir(parents=True, exist_ok=True)
        for key in MAPS:
            shutil.copyfile(source/f'{key}.jpg', folder/f'{key}.jpg')
        ground.append({'id': ident, 'size_m': meta['size_m'], 'source': meta['page'], 'license': meta['license'],
                       **{key: f'assets/ground/{ident}/{key}.jpg' for key in MAPS}})
    if ground:
        write_json(directory/'catalog.json', ground)
    return ground
