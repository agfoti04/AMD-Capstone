"""Download pinned Poly Haven assets and prepare them for the Mars generator.

lock     Rebuild config/assets.lock.json from the Poly Haven API. Only needed
         when config/assets.json changes.
fetch    Download every file in the lock into the cache and check its MD5.
prepare  Simplify rock meshes with headless Blender and stage the textures.
all      fetch, then prepare.

Downloads stay in the cache (default ./assets_cache, or $MARS_ASSET_CACHE),
which is not committed.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path

from .common import write_json

API = 'https://api.polyhaven.com'
USER_AGENT = 'rover-mars-training/0.1'


def config_file(name):
    local = Path(__file__).resolve().parents[1]/'config'/name
    if local.exists():
        return local
    from ament_index_python.packages import get_package_share_directory
    return Path(get_package_share_directory('mars_training_environment'))/'config'/name


def default_cache():
    return Path(os.environ.get('MARS_ASSET_CACHE', 'assets_cache')).resolve()


def _request(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': USER_AGENT}), timeout=120)


def _get_json(url):
    with _request(url) as response:
        return json.load(response)


def _entry(item):
    return {'url': item['url'], 'md5': item['md5'], 'size': item['size']}


def build_lock(sources):
    rocks, ground = {}, {}
    res = sources['rocks']['resolution']
    idents = list(sources['rocks']['ids'])+[item['id'] for item in sources['rocks'].get('split', [])]
    for ident in idents:
        files = _get_json(f'{API}/files/{ident}')
        gltf = files['gltf'][res]['gltf']
        entries = {Path(gltf['url']).name: _entry(gltf)}
        entries.update({path: _entry(item) for path, item in gltf['include'].items()})
        rough = files['Rough'][res]['jpg']
        entries['textures/'+Path(rough['url']).name] = _entry(rough)
        def find(tag):
            return next(p for p in entries if p.startswith('textures/') and f'_{tag}_' in p)
        rocks[ident] = {'page': f'https://polyhaven.com/a/{ident}', 'gltf': Path(gltf['url']).name,
                        'albedo': find('diff'), 'normal': find('nor_gl'), 'roughness': find('rough'),
                        'files': entries}
    res = sources['ground']['resolution']
    for ident in sources['ground']['ids']:
        files = _get_json(f'{API}/files/{ident}')
        info = _get_json(f'{API}/info/{ident}')
        maps = {'albedo': files['Diffuse'][res]['jpg'], 'normal': files['nor_gl'][res]['jpg'],
                'roughness': files['Rough'][res]['jpg']}
        width_mm = (info.get('dimensions') or [2000, 2000])[0]
        ground[ident] = {'page': f'https://polyhaven.com/a/{ident}', 'size_m': width_mm/1000,
                         **{k: Path(v['url']).name for k, v in maps.items()},
                         'files': {Path(v['url']).name: _entry(v) for v in maps.values()}}
    return {'provider': sources['provider'], 'license': sources['license'], 'rocks': rocks, 'ground': ground}


def _md5(path):
    digest = hashlib.md5()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def fetch(cache, lock):
    downloaded = reused = 0
    for kind in ('rocks', 'ground'):
        for ident, asset in lock[kind].items():
            for relative, item in asset['files'].items():
                dest = cache/'raw'/kind/ident/relative
                if dest.exists() and _md5(dest) == item['md5']:
                    reused += 1
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                partial = dest.with_suffix(dest.suffix+'.part')
                with _request(item['url']) as response, open(partial, 'wb') as handle:
                    shutil.copyfileobj(response, handle)
                if partial.stat().st_size != item['size'] or _md5(partial) != item['md5']:
                    partial.unlink()
                    raise ValueError(f'{item["url"]} does not match the pinned checksum')
                partial.replace(dest)
                downloaded += 1
    print(f'Fetched {downloaded} files, {reused} already cached, into {cache/"raw"}', flush=True)


def piece_areas(vertices, faces):
    """Surface area of each connected piece, largest first.

    Vertices are joined by position, so UV seams do not split a piece.
    """
    import numpy as np
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=int)
    _, welded = np.unique(np.round(v, 5), axis=0, return_inverse=True)
    parent = np.arange(welded.max()+1)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b, c in welded.ravel()[f]:
        ra = find(a)
        parent[find(b)] = ra
        parent[find(c)] = ra
    roots = np.array([find(i) for i in welded.ravel()[f[:, 0]]])
    area = np.linalg.norm(np.cross(v[f[:, 1]]-v[f[:, 0]], v[f[:, 2]]-v[f[:, 0]]), axis=1)/2
    _, piece = np.unique(roots, return_inverse=True)
    return sorted(np.bincount(piece, weights=area).tolist(), reverse=True)


def _piece_ok(meta_path, meta):
    if not meta_path.exists():
        return False
    existing = json.loads(meta_path.read_text())
    return {key: existing.get(key) for key in meta} == meta


def _stage_rock(raw, asset, lock, out, visual, collision, targets):
    for key in ('albedo', 'normal', 'roughness'):
        shutil.copyfile(raw/asset[key], out/f'{key}.jpg')
    areas = piece_areas(visual['vertices'], visual['faces'])
    if sum(amount >= .05*sum(areas) for amount in areas) > 1:
        raise ValueError(f'{out.name} is still several separate stones ({len(areas)} pieces)')
    meta = {'id': out.name, 'page': asset['page'], 'license': lock['license'],
            'visual_target': targets['visual_triangles'], 'collision_target': targets['collision_triangles'],
            'source_md5': {key: value['md5'] for key, value in asset['files'].items()},
            'triangles': len(visual['faces']), 'collision_triangles': len(collision['faces'])}
    return meta


def _prepare_split(cache, lock, ident, asset, spec, targets, script, blender, force):
    """Turn one multi-stone scan into several single-rock assets."""
    raw = cache/'raw'/'rocks'/ident
    count = int(spec['pieces'])
    staged = cache/'prepared'/'rocks'
    fingerprint = {'source': ident, 'min_area_share': spec['min_area_share'], 'pieces': count,
                   'source_md5': {key: value['md5'] for key, value in asset['files'].items()},
                   'visual_target': targets['visual_triangles'], 'collision_target': targets['collision_triangles']}
    if not force and all(_piece_ok(staged/f'{ident}_p{i:02d}'/'meta.json', fingerprint) for i in range(count)):
        return
    work = cache/'prepared'/'_split'/ident
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    subprocess.run([blender, '-b', '--factory-startup', '--python-exit-code', '1', '--python', str(script), '--',
                    'split', str(raw/asset['gltf']), str(work), str(targets['visual_triangles']),
                    str(targets['collision_triangles']), str(spec['min_area_share']), str(spec['max_pieces'])],
                   check=True)
    folders = sorted(path for path in work.iterdir() if path.is_dir())
    if len(folders) != count:
        raise ValueError(f'{ident} kept {len(folders)} pieces; assets.json asks for {count}')
    for index, folder in enumerate(folders):
        dest = staged/f'{ident}_p{index:02d}'
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        shutil.copyfile(folder/'visual.json', dest/'visual.json')
        shutil.copyfile(folder/'collision.json', dest/'collision.json')
        visual = json.loads((dest/'visual.json').read_text())
        collision = json.loads((dest/'collision.json').read_text())
        meta = _stage_rock(raw, asset, lock, dest, visual, collision, targets)
        meta.update(fingerprint)
        write_json(dest/'meta.json', meta)
        print(f'Prepared rock {dest.name} from {ident}: {meta["triangles"]} visual / '
              f'{meta["collision_triangles"]} collision triangles', flush=True)


def prepare(cache, lock, sources, blender='blender', force=False):
    script = Path(__file__).with_name('blender_prepare.py')
    targets = sources['rocks']
    splits = {item['id']: item for item in sources['rocks'].get('split', [])}
    for ident, asset in lock['rocks'].items():
        if ident in splits:
            _prepare_split(cache, lock, ident, asset, splits[ident], targets, script, blender, force)
            continue
        raw, out = cache/'raw'/'rocks'/ident, cache/'prepared'/'rocks'/ident
        meta_path = out/'meta.json'
        meta = {'id': ident, 'page': asset['page'], 'license': lock['license'],
                'visual_target': targets['visual_triangles'], 'collision_target': targets['collision_triangles'],
                'source_md5': {k: v['md5'] for k, v in asset['files'].items()}}
        if not force and meta_path.exists():
            existing = json.loads(meta_path.read_text())
            if {k: existing.get(k) for k in meta} == meta:
                continue
        out.mkdir(parents=True, exist_ok=True)
        subprocess.run([blender, '-b', '--factory-startup', '--python-exit-code', '1', '--python', str(script), '--',
                        str(raw/asset['gltf']), str(out/'visual.json'), str(out/'collision.json'),
                        str(targets['visual_triangles']), str(targets['collision_triangles'])],
                       check=True)
        visual = json.loads((out/'visual.json').read_text())
        collision = json.loads((out/'collision.json').read_text())
        # One asset must be one rock so it gets one label box; a scanned pile of
        # touching stones would teach the detector to box clusters.
        meta = _stage_rock(raw, asset, lock, out, visual, collision, targets)
        write_json(meta_path, meta)
        print(f'Prepared rock {ident}: {meta["triangles"]} visual / {meta["collision_triangles"]} collision triangles',
              flush=True)
    for ident, asset in lock['ground'].items():
        raw, out = cache/'raw'/'ground'/ident, cache/'prepared'/'ground'/ident
        out.mkdir(parents=True, exist_ok=True)
        for key in ('albedo', 'normal', 'roughness'):
            shutil.copyfile(raw/asset[key], out/f'{key}.jpg')
        write_json(out/'meta.json', {'id': ident, 'page': asset['page'], 'license': lock['license'],
                                     'size_m': asset['size_m']})
    print(f'Prepared assets in {cache/"prepared"}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['lock', 'fetch', 'prepare', 'all'])
    parser.add_argument('--cache', type=Path, default=None)
    parser.add_argument('--lock', type=Path, default=None, help='Lock file to read, or to write with "lock"')
    parser.add_argument('--blender', default='blender')
    parser.add_argument('--force', action='store_true', help='Re-run Blender for every rock')
    args = parser.parse_args(argv)
    sources = json.loads(config_file('assets.json').read_text())
    if args.command == 'lock':
        target = args.lock or config_file('assets.lock.json')
        write_json(target, build_lock(sources))
        print(f'Wrote {target}')
        return
    lock = json.loads((args.lock or config_file('assets.lock.json')).read_text())
    expected = set(sources['rocks']['ids'])|{item['id'] for item in sources['rocks'].get('split', [])}
    if set(lock['rocks']) != expected or set(lock['ground']) != set(sources['ground']['ids']):
        raise SystemExit('assets.lock.json does not match assets.json; run "lock" first')
    cache = (args.cache or default_cache()).resolve()
    if args.command in ('fetch', 'all'):
        fetch(cache, lock)
    if args.command in ('prepare', 'all'):
        prepare(cache, lock, sources, args.blender, args.force)


if __name__ == '__main__':
    main()
