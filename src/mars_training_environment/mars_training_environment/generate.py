"""Generate portable, seeded world bundles without importing ROS."""
import argparse
import json
import math
import shutil
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from .common import digest, read_config, rotation, split_for_seed, write_json
from .meshes import build_library
from .terrain import Terrain, generate_terrain


def make_scene(root, config, seed, assets):
    root = Path(root).resolve()
    ident = f'scene_{seed:06d}'
    folder = root/'scenes'/ident
    manifest_path = folder/'scene.json'
    config_hash = digest(config)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing['config_sha256'] != config_hash:
            raise ValueError(f'{ident} already exists with another config; use a new output directory')
    rng = np.random.RandomState(seed)
    field, terrain = generate_terrain(seed, config['terrain'], folder)
    split = split_for_seed(seed)
    pool = [a for a in assets if a['held_out'] == (split == 'test')]
    if not pool:
        raise ValueError('No rock assets in this split')
    rc = config['rocks']
    empty = seed in config['dataset']['empty_seeds']
    count = 0 if empty else int(rng.randint(rc['count_range'][0], rc['count_range'][1]+1))
    objects = []
    size = field.size
    clusters = rng.uniform(-size*.32, size*.32, (5,2))
    geometry = {a['id']: np.asarray(json.loads((root/a['geometry']).read_text())['vertices']) for a in pool}
    for index in range(count):
        asset = pool[int(rng.randint(len(pool)))]
        sampling = rng.random_sample() < rc['sampling_probability']
        diameter = float(rng.uniform(*(rc['sampling_size_m'] if sampling else rc['obstacle_size_m'])))
        tilt = math.radians(rc['max_tilt_deg'])
        rpy = [float(rng.uniform(-tilt,tilt)), float(rng.uniform(-tilt,tilt)), float(rng.uniform(-math.pi,math.pi))]
        points = geometry[asset['id']] @ rotation(rpy).T * diameter
        radius = float(np.linalg.norm(points[:, :2], axis=1).max())
        xy = None
        for _ in range(600):
            candidate = (clusters[int(rng.randint(len(clusters)))] + rng.normal(0, .7, 2)
                         if rng.random_sample() < rc['cluster_probability'] else rng.uniform(-size*.43, size*.43, 2))
            if max(abs(candidate)) + radius > size/2-.4:
                continue
            if np.linalg.norm(candidate) < config['terrain']['spawn_clearance_m'] + radius + .3:
                continue
            if any(np.linalg.norm(candidate-np.asarray(o['xyz'][:2])) < radius+o['radius_m']+rc['clearance_m'] for o in objects):
                continue
            if np.linalg.norm(field.gradient(*candidate)) > math.tan(math.radians(config['terrain']['max_slope_deg'])):
                continue
            xy = candidate
            break
        if xy is None:
            raise ValueError('Placement constraints could not be met; reduce rock count or sizes')
        # Lowest support contact against the full sampled footprint. Burial is
        # the only intentional rock/terrain intersection; pairwise rocks stay apart.
        ground = field.height(points[:,0]+xy[0], points[:,1]+xy[1])
        support = float(np.max(ground-points[:,2]))
        burial = float(rng.uniform(*rc['burial_fraction']))
        z = support - burial*float(np.ptp(points[:,2]))
        shade = rng.uniform(.72,1.0)
        color = (np.array([1., rng.uniform(.82,1.), rng.uniform(.7,.95)])*shade).tolist()
        objects.append({'name': f'rock_{index:03d}', 'asset_id': asset['id'], 'xyz': [float(xy[0]),float(xy[1]),z],
                        'rpy': rpy, 'scale': diameter, 'size_class': 'sampling' if sampling else 'obstacle',
                        'radius_m': radius, 'burial_fraction': burial, 'burial_m': support-z, 'color': color})
    views = []
    for index in range(config['capture']['views_per_scene']):
        found = None
        for attempt in range(1000):
            if objects:
                target = objects[(index+attempt) % len(objects)]
                tx,ty,_ = target['xyz']
                # distance_m is how far the lens is from the target. The lens is
                # ahead of base_link, and a large rock can reach the near plane.
                lens = float(config['camera']['mount_xyz'][0])
                near = float(config['camera']['near_m'])
                lo, hi = config['capture']['distance_m']
                min_gap = target['radius_m'] + near + 0.25
                if min_gap >= hi:
                    continue
                gap = float(rng.uniform(max(lo, min_gap), hi))
                angle = float(rng.uniform(-math.pi, math.pi))
                x,y = tx-(gap+lens)*math.cos(angle), ty-(gap+lens)*math.sin(angle)
                # Last two views intentionally place targets near the image border.
                offset = rng.uniform(-.12,.12) if index % 5 < 3 else rng.choice([-1,1])*rng.uniform(.35,.58)
                yaw = angle+offset
            else:
                x,y = rng.uniform(-size*.3,size*.3,2)
                yaw = rng.uniform(-math.pi,math.pi)
            if max(abs(x),abs(y)) > size/2-1.2:
                continue
            if any(math.hypot(x-o['xyz'][0],y-o['xyz'][1]) < .32+o['radius_m']+.1 for o in objects):
                continue
            if objects:
                lens = float(config['camera']['mount_xyz'][0])
                near = float(config['camera']['near_m'])
                cx, cy = x+lens*math.cos(yaw), y+lens*math.sin(yaw)
                if any(math.hypot(cx-o['xyz'][0], cy-o['xyz'][1]) < o['radius_m']+near+0.08 for o in objects):
                    continue
            gx,gy = field.gradient(x,y)
            if math.hypot(gx,gy) > math.tan(math.radians(config['terrain']['max_slope_deg'])):
                continue
            roll = math.atan(-gx*math.sin(yaw)+gy*math.cos(yaw))
            pitch = -math.atan(gx*math.cos(yaw)+gy*math.sin(yaw))
            found = {'id': f'view_{index:03d}', 'xyz':[float(x),float(y),float(field.height(x,y))+.09],
                     'rpy':[roll,pitch,float(yaw)]}
            break
        if found is None:
            raise ValueError('Could not place a collision-free camera viewpoint')
        views.append(found)
    scene = {'schema_version':1, 'id':ident, 'seed':seed, 'scene_group':f'seed_{seed}', 'split':split,
             'config_sha256':config_hash, 'config':config, 'empty':empty, 'terrain':terrain,
             'objects':objects, 'views':views, 'gravity':[0,0,-9.81],
             'lighting':{'azimuth':float(rng.uniform(0,math.tau)), 'elevation':float(rng.uniform(.35,1.2)),
                         'intensity':float(rng.uniform(.65,1.1)), 'ambient':float(rng.uniform(.25,.45))},
             'assets_sha256':digest(assets)}
    scene['content_sha256'] = digest(scene)
    write_json(manifest_path, scene)
    write_world(root, scene, assets)
    return scene


def write_world(root, scene, assets):
    """Write host-specific URIs only here; manifests and hashes stay portable."""
    root = Path(root).resolve()
    folder = root/'scenes'/scene['id']
    by_id = {a['id']:a for a in assets}
    def uri(path):
        return escape(Path(path).resolve().as_uri())
    material_folder = folder/'materials'
    material_folder.mkdir(exist_ok=True)
    t = scene['terrain']
    terrain_material = ''
    if t['backend'] == 'heightmap':
        height = (f'<uri>{uri(folder/"heightmap.png")}</uri><size>{t["size_m"]} {t["size_m"]} {t["heightmap_range_m"]}</size>'
                  f'<pos>0 0 {t["heightmap_min_m"]}</pos><sampling>1</sampling>')
        collision = '<heightmap>'+height+'</heightmap>'
        visual = ('<heightmap>'+height+'<use_terrain_paging>false</use_terrain_paging><texture>'
                  f'<diffuse>{uri(folder/"ground.png")}</diffuse><normal>{uri(folder/"normal.png")}</normal>'
                  f'<size>{t["size_m"]}</size></texture></heightmap>')
    else:
        mesh = f'<mesh><uri>{uri(folder/"terrain.dae")}</uri></mesh>'
        collision = visual = mesh
        terrain_material = ('<material><ambient>0.55 0.36 0.24 1</ambient><diffuse>0.75 0.48 0.32 1</diffuse>'
                            f'<script><uri>{uri(material_folder)}</uri><name>Mars/ground</name></script></material>')
    rocks = []
    materials = []
    shutil.copyfile(folder/'ground.png', material_folder/'ground.png')
    materials.append('''material Mars/ground
{
  technique { pass {
    ambient 0.55 0.36 0.24 1
    diffuse 0.75 0.48 0.32 1
    specular 0.02 0.02 0.02 1 2
    cull_hardware none
    cull_software none
    texture_unit { texture ground.png
      filtering trilinear
    }
  } }
}''')
    for obj in scene['objects']:
        asset = by_id[obj['asset_id']]
        pose = ' '.join(f'{n:.12g}' for n in obj['xyz']+obj['rpy'])
        scale = ' '.join([str(obj['scale'])]*3)
        color = ' '.join(str(c) for c in obj['color'])+' 1'
        material_name = f'Mars/{scene["id"]}/{obj["name"]}'
        texture_name = f"{asset['id']}.png"
        shutil.copyfile(root/'assets'/asset['id']/'surface.png', material_folder/texture_name)
        materials.append(f'''material {material_name}
{{
  technique {{ pass {{
    ambient {color}
    diffuse {color}
    specular 0 0 0 1 1
    cull_hardware none
    cull_software none
    texture_unit {{ texture {texture_name}
      filtering anisotropic
      max_anisotropy 4
    }}
  }} }}
}}''')
        rocks.append(f'''<model name="{obj['name']}"><static>true</static><pose>{pose}</pose><link name="link">
<collision name="collision"><geometry><mesh><uri>{uri(root/asset['collision'])}</uri><scale>{scale}</scale></mesh></geometry>
<surface><friction><ode><mu>0.9</mu><mu2>0.8</mu2></ode></friction></surface></collision>
<visual name="visual"><geometry><mesh><uri>{uri(root/asset['mesh'])}</uri><scale>{scale}</scale></mesh></geometry>
<material><script><uri>{uri(material_folder)}</uri><name>{material_name}</name></script></material><cast_shadows>true</cast_shadows></visual>
</link></model>''')
    light = scene['lighting']
    a,e,b = light['azimuth'],light['elevation'],light['intensity']
    ambient = light['ambient']
    text = f'''<?xml version="1.0"?><sdf version="1.6"><world name="mars_training">
<gravity>0 0 -9.81</gravity><physics name="default" type="ode"><max_step_size>0.002</max_step_size><real_time_update_rate>500</real_time_update_rate></physics>
<scene><ambient>{ambient} {ambient} {ambient} 1</ambient><background>0.55 0.40 0.30 1</background><shadows>true</shadows></scene>
<light name="sun" type="directional"><pose>0 0 20 0 0 0</pose><cast_shadows>true</cast_shadows>
<diffuse>{b} {b*.94} {b*.86} 1</diffuse><specular>0.05 0.05 0.05 1</specular>
<direction>{math.cos(a)*math.cos(e)} {math.sin(a)*math.cos(e)} {-math.sin(e)}</direction></light>
<plugin name="gazebo_ros_state" filename="libgazebo_ros_state.so"><ros><namespace>/gazebo</namespace></ros><update_rate>50</update_rate></plugin>
<model name="mars_terrain"><static>true</static><link name="ground">
<collision name="terrain_collision"><geometry>{collision}</geometry><surface><friction><ode><mu>0.9</mu><mu2>0.8</mu2></ode></friction></surface></collision>
<visual name="terrain_visual"><geometry>{visual}</geometry>{terrain_material}<cast_shadows>true</cast_shadows></visual>
</link></model>{''.join(rocks)}</world></sdf>'''
    (folder/'world.sdf').write_text(text, encoding='utf-8')
    (material_folder/'rocks.material').write_text('\n'.join(materials),encoding='utf-8')


def generate_bundle(output, config, seeds=None):
    output = Path(output).resolve()
    seeds = config['dataset']['seeds'] if seeds is None else seeds
    output.mkdir(parents=True, exist_ok=True)
    config_hash = digest(config)
    plan_path = output/'plan.json'
    if plan_path.exists():
        existing = json.loads(plan_path.read_text())
        if existing['config_sha256'] != config_hash:
            raise ValueError('Output contains another configuration; use a fresh directory')
    assets = build_library(output/'assets', config['rocks'])
    scenes = [make_scene(output, config, int(seed), assets) for seed in seeds]
    plan = {'schema_version':1, 'config_sha256':config_hash, 'config':config,
            'scenes':[{'id':s['id'],'seed':s['seed'],'split':s['split'], 'content_sha256':s['content_sha256'],
                       'views':len(s['views'])} for s in scenes],
            'reserved_meshes':[a['id'] for a in assets if a['held_out']]}
    write_json(plan_path, plan)
    write_json(output/'splits.json', {s['scene_group']:s['split'] for s in scenes})
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, action='append', help='Repeat for selected scenes; default is all configured seeds')
    args = parser.parse_args(argv)
    plan = generate_bundle(args.output, read_config(args.config), args.seed)
    print(f'Generated {len(plan["scenes"])} scenes / {sum(s["views"] for s in plan["scenes"])} planned images in {args.output}')


if __name__ == '__main__':
    main()
