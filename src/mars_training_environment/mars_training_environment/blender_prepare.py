"""Run inside Blender: blender -b --python blender_prepare.py -- SRC VISUAL COLLISION VIS_TRIS COL_TRIS

Imports one glTF rock, joins its meshes, and writes two simplified copies as
JSON. Both share one normalization: centered on the visual bounding box and
scaled so its longest side is 1. Coordinates are Blender's Z-up metres.
The visual copy keeps UVs, split at seams, and is also the label mesh.

A scan that is several separate stones is prepared with a leading "split"
argument: split SRC OUTDIR VIS_TRIS COL_TRIS MIN_SHARE MAX_PIECES. Each kept
piece is written to OUTDIR/piece_XX/{visual,collision}.json.
"""
import json
import sys
from pathlib import Path

import bmesh
import bpy


def triangle_count(obj):
    return sum(len(p.vertices)-2 for p in obj.data.polygons)


def decimated(source, target, weld):
    copy = source.copy()
    copy.data = source.data.copy()
    bpy.context.collection.objects.link(copy)
    if weld:
        copy.modifiers.new('weld', 'WELD').merge_threshold = 1e-5
    tris = triangle_count(copy)
    decimate = copy.modifiers.new('decimate', 'DECIMATE')
    decimate.use_collapse_triangulate = True
    decimate.ratio = min(1.0, target/tris)
    copy.modifiers.new('triangulate', 'TRIANGULATE')
    mesh = None
    for _ in range(6):
        if mesh is not None:
            bpy.data.meshes.remove(mesh)
        bpy.context.view_layer.update()
        evaluated = copy.evaluated_get(bpy.context.evaluated_depsgraph_get())
        mesh = bpy.data.meshes.new_from_object(evaluated)
        mesh.calc_loop_triangles()
        got = len(mesh.loop_triangles)
        if got <= target*1.05 or decimate.ratio < 1e-4:
            break
        decimate.ratio *= target/got
    bpy.data.objects.remove(copy)
    return mesh, got


def simplified(source, target):
    # Some scans ship as unwelded patches that stall collapse decimation;
    # welding others creates seams that stall it instead. Keep whichever works.
    mesh, got = decimated(source, target, weld=False)
    if got > target*1.05:
        welded, welded_got = decimated(source, target, weld=True)
        if welded_got < got:
            bpy.data.meshes.remove(mesh)
            mesh = welded
        else:
            bpy.data.meshes.remove(welded)
    return mesh


def extract(mesh, with_uv):
    uv_layer = mesh.uv_layers.active if with_uv else None
    index, vertices, uvs, faces = {}, [], [], []
    for tri in mesh.loop_triangles:
        face = []
        for loop, vert in zip(tri.loops, tri.vertices):
            uv = tuple(round(c, 6) for c in uv_layer.data[loop].uv) if uv_layer else ()
            key = (vert, uv)
            if key not in index:
                index[key] = len(vertices)
                vertices.append(list(mesh.vertices[vert].co))
                uvs.append(list(uv))
            face.append(index[key])
        faces.append(face)
    return vertices, uvs, faces


def load_rock(src):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=src)
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if not meshes:
        raise RuntimeError(f'No mesh in {src}')
    bpy.ops.object.select_all(action='DESELECT')
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    if len(meshes) > 1:
        bpy.ops.object.join()
    rock = bpy.context.view_layer.objects.active
    if not rock.data.uv_layers:
        raise RuntimeError(f'{src} has no UV map')
    return rock


def write_normalized(rock, visual_target, collision_target, visual_path, collision_path):
    visual_mesh = simplified(rock, int(visual_target))
    vertices, uvs, faces = extract(visual_mesh, True)
    lo = [min(v[i] for v in vertices) for i in range(3)]
    hi = [max(v[i] for v in vertices) for i in range(3)]
    center = [(a+b)/2 for a, b in zip(lo, hi)]
    scale = 1/max(b-a for a, b in zip(lo, hi))

    def normalize(points):
        return [[round((p[i]-center[i])*scale, 7) for i in range(3)] for p in points]

    with open(visual_path, 'w') as handle:
        json.dump({'vertices': normalize(vertices), 'uv': uvs, 'faces': faces}, handle)
    collision_mesh = simplified(rock, int(collision_target))
    vertices, _, faces = extract(collision_mesh, False)
    with open(collision_path, 'w') as handle:
        json.dump({'vertices': normalize(vertices), 'faces': faces}, handle)


def connected_pieces(rock, min_share, max_pieces):
    """Largest separate stones, joined by vertex position so UV seams stay whole."""
    mesh = rock.data
    parent = list(range(len(mesh.vertices)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    by_position = {}
    for vertex in mesh.vertices:
        other = by_position.setdefault(tuple(round(c, 5) for c in vertex.co), vertex.index)
        parent[find(vertex.index)] = find(other)
    for polygon in mesh.polygons:
        first = find(polygon.vertices[0])
        for index in polygon.vertices[1:]:
            parent[find(index)] = first
    area = {}
    for polygon in mesh.polygons:
        root = find(polygon.vertices[0])
        area[root] = area.get(root, 0.0)+polygon.area
    total = sum(area.values())
    chosen = [root for root, amount in sorted(area.items(), key=lambda item: -item[1])
              if amount >= min_share*total][:max_pieces]
    if not chosen:
        raise RuntimeError(f'No piece reaches {min_share:.0%} of the surface')
    pieces = []
    for root in chosen:
        patch = bmesh.new()
        patch.from_mesh(mesh)
        patch.verts.ensure_lookup_table()
        drop = [face for face in patch.faces if find(face.verts[0].index) != root]
        bmesh.ops.delete(patch, geom=drop, context='FACES_ONLY')
        bmesh.ops.delete(patch, geom=[vertex for vertex in patch.verts if not vertex.link_faces], context='VERTS')
        piece_mesh = bpy.data.meshes.new('piece')
        patch.to_mesh(piece_mesh)
        patch.free()
        obj = bpy.data.objects.new('piece', piece_mesh)
        bpy.context.collection.objects.link(obj)
        pieces.append((obj, area[root]/total))
    return pieces


def main():
    args = sys.argv[sys.argv.index('--')+1:]
    if args[0] == 'split':
        _, src, outdir, visual_target, collision_target, min_share, max_pieces = args
        rock = load_rock(src)
        outdir = Path(outdir)
        pieces = connected_pieces(rock, float(min_share), int(max_pieces))
        print(f'{len(pieces)} pieces kept: ' + ', '.join(f'{share:.1%}' for _, share in pieces), flush=True)
        for index, (piece, _) in enumerate(pieces):
            folder = outdir/f'piece_{index:02d}'
            folder.mkdir(parents=True, exist_ok=True)
            write_normalized(piece, visual_target, collision_target, folder/'visual.json', folder/'collision.json')
        return
    src, visual_path, collision_path, visual_target, collision_target = args
    rock = load_rock(src)
    write_normalized(rock, visual_target, collision_target, visual_path, collision_path)


main()
