"""Generate portable, seeded Gazebo Harmonic world bundles without importing ROS."""
import argparse
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from rover_training.geometry import CLASS_NAMES, class_id_for, nominal_intrinsics

from .asset_cache import default_cache
from .annotation import Annotator
from .candidates import NOTES, add_candidates, paint_procedural_ground, write_candidate_materials
from .common import digest, quaternion, read_config, require_label_schema, rotation, split_for_seed, write_json
from .dust import bake_scene_dust
from .ground import composite_ground
from .framing import configured_extrinsics, planning_size_guard, projected_target_fits, target_framing
from .meshes import build_ground, build_library, write_dae
from .terrain import Terrain, generate_terrain

WORLD_NAME = 'mars_training'
ROCK_LABEL = 1


def make_scene(root, config, seed, assets, ground=()):
    root = Path(root).resolve()
    ident = f'scene_{seed:06d}'
    folder = root/'scenes'/ident
    manifest_path = folder/'scene.json'
    config_hash = digest(config)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        require_label_schema(existing)
        if existing['config_sha256'] != config_hash:
            raise ValueError(f'{ident} already exists with another config; use a new output directory')
    rng = np.random.RandomState(seed)
    field, driveable, terrain = generate_terrain(seed, config['terrain'], folder)
    split = split_for_seed(seed)
    # Candidate meshes are placed explicitly below with their appearance name.
    # Do not randomly instantiate a slab or crystal mesh as an ordinary rock.
    pool = [a for a in assets if not a.get('shared_shape') and a['held_out'] == (split == 'test')]
    if not pool:
        raise ValueError('No rock assets in this split')
    rc = config['rocks']
    empty = seed in config['dataset']['empty_seeds']
    rocks_only = seed in config['dataset'].get('rocks_only_seeds', [])
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
            if np.linalg.norm(driveable.gradient(*candidate)) > math.tan(math.radians(config['terrain']['max_slope_deg'])):
                continue
            xy = candidate
            break
        if xy is None:
            raise ValueError('Placement constraints could not be met; reduce rock count or sizes')
        # Lowest support contact against the full sampled footprint. Burial is
        # the only intentional rock/terrain intersection; pairwise rocks stay apart.
        surface = field.height(points[:,0]+xy[0], points[:,1]+xy[1])
        support = float(np.max(surface-points[:,2]))
        burial = float(rng.uniform(*rc['burial_fraction']))
        z = support - burial*float(np.ptp(points[:,2]))
        shade = rng.uniform(.72,1.0)
        color = (np.array([1., rng.uniform(.82,1.), rng.uniform(.7,.95)])*shade).tolist()
        objects.append({'name': f'rock_{index:03d}', 'asset_id': asset['id'], 'xyz': [float(xy[0]),float(xy[1]),z],
                        'rpy': rpy, 'scale': diameter, 'size_class': 'sampling' if sampling else 'obstacle',
                        'radius_m': radius, 'burial_fraction': burial, 'burial_m': support-z, 'color': color})
    regions = []
    if not empty and config.get('candidates', {}).get('enabled'):
        add_candidates(rng, config, field, driveable, assets, root, objects, regions, folder,
                       allowed_categories={'pale_stone'} if rocks_only else None)
    views = []
    targets = list(objects) + list(regions)
    priority = [item for item in targets if class_id_for(item.get('class_name', 'rock')) not in (None, 0)]
    planner_config = json.loads(json.dumps(config))
    pw = config['capture']['planning_width']
    ph = round(pw * config['camera']['height']/config['camera']['width'])
    planning_k = nominal_intrinsics(pw, ph, config['camera']['horizontal_fov_rad'])
    ratio = pw/config['camera']['width']
    planner_config['labels']['min_box_side_px'] = max(1, config['labels']['min_box_side_px']*ratio)
    planner_config['labels']['min_visible_pixels'] = max(1, config['labels']['min_visible_pixels']*ratio*ratio)
    planner = Annotator(root, {'id':ident, 'class_names':list(CLASS_NAMES), 'config':planner_config,
                              'terrain':terrain, 'objects':objects, 'regions':regions})
    full_k = nominal_intrinsics(config['camera']['width'], config['camera']['height'],
                                config['camera']['horizontal_fov_rad'])
    full_planner = Annotator(root, {'id':ident, 'class_names':list(CLASS_NAMES), 'config':config,
                                   'terrain':terrain, 'objects':objects, 'regions':regions})
    target_vertices = {o['name']:v for o,v,_f in planner.meshes}
    target_vertices.update({region['name']:mesh[0] for region,mesh in zip(regions,planner.region_meshes)})
    for index in range(config['capture']['views_per_scene']):
        found = None
        difficult = bool(targets) and (index+1) % config['capture']['difficult_view_every'] == 0
        for attempt in range(1000):
            if targets:
                if priority and index % 2 == 0:
                    target = priority[(index // 2 + attempt) % len(priority)]
                else:
                    target = targets[(index + attempt) % len(targets)]
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
                # One difficult view per group, with a target still required in frame.
                offset = (rng.choice([-1,1])*rng.uniform(.25,.48) if difficult else rng.uniform(-.08,.08))
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
            gx,gy = driveable.gradient(x,y)
            if math.hypot(gx,gy) > math.tan(math.radians(config['terrain']['max_slope_deg'])):
                continue
            roll = math.atan(-gx*math.sin(yaw)+gy*math.cos(yaw))
            pitch = -math.atan(gx*math.cos(yaw)+gy*math.sin(yaw))
            candidate_view = {'id': f'view_{index:03d}', 'xyz':[float(x),float(y),float(driveable.height(x,y))+.09],
                              'rpy':[roll,pitch,float(yaw)], 'difficult':difficult}
            if targets:
                # Predict at wheel-rest height; Gazebo settles the commanded pose.
                settled = dict(candidate_view, xyz=candidate_view['xyz'][:2]+[candidate_view['xyz'][2]-.01])
                if not projected_target_fits(target_vertices[target['name']], settled, config['camera'],
                                             planning_k, config['capture'], difficult):
                    continue
                planned_labels, _records, _quality, _instances = planner.annotate(
                    settled['xyz'], quaternion(settled['rpy']), configured_extrinsics(config['camera']), planning_k)
                framing = target_framing(planned_labels, target['name'], planning_k, config['capture'], difficult,
                                         size_guard_px=planning_size_guard(planning_k, config['capture']))
                if not framing['passed']:
                    continue
                # Coarse raster bounds can overestimate a small visible target.
                # Confirm the final choice at capture resolution, with room for
                # settling. Actual capture retains the unmodified acceptance floor.
                full_labels, _records, _quality, _instances = full_planner.annotate(
                    settled['xyz'], quaternion(settled['rpy']), configured_extrinsics(config['camera']), full_k)
                full_framing = target_framing(full_labels, target['name'], full_k, config['capture'], difficult,
                                              size_guard_px=planning_size_guard(full_k, config['capture']))
                if not full_framing['passed']:
                    continue
                candidate_view.update(target_name=target['name'], target_class_id=class_id_for(target.get('class_name','rock')),
                                      planned_framing=full_framing, coarse_planned_framing=framing,
                                      planning_resolution=[full_k['width'], full_k['height']])
            found = candidate_view
            break
        if found is None:
            raise ValueError(f'{ident}/view_{index:03d}: no collision-free pose satisfies target framing; '
                             'inspect target sizes, camera pose, distances or framing thresholds')
        views.append(found)
    ground_recipe = (composite_ground(root, list(ground), config['terrain'], seed, folder, regions) if ground
                     else {'layers': [], 'procedural': 'ground.png'})
    if regions and not ground:
        paint_procedural_ground(folder, regions, config['terrain']['size_m'], seed)
        # Painted candidate masks use one world-sized image, including the
        # procedural fallback. Repeating its UVs would move/duplicate patches.
        vertices, faces = field.mesh()
        uv = (vertices[:, :2]+field.size/2)/field.size
        write_dae(folder/'terrain.dae',vertices,faces,uv=uv)
        ground_recipe['candidate_mask'] = 'candidate_mask.png'
        ground_recipe['mask_ids'] = {'background': 0, 'silica_soil': 1, 'pale_lookalike': 2}
    dust_bgr = ground_recipe.get('dust_bgr') or [96.0, 128.0, 168.0]
    if any(obj.get('material_maps') is None and obj.get('class_name') in ('sulfur_rock', 'pale_stone', 'cheyava_rock', 'veined_lookalike') for obj in objects):
        write_candidate_materials(root, folder, objects, dust_bgr, seed, config)
    dust_textures = (bake_scene_dust(root, folder, assets, objects, dust_bgr)
                     if ground_recipe.get('dust_bgr') else {})
    # Mars sky: dusty butterscotch by day, varied per scene.
    sky = np.array([.80, .62, .46])*rng.uniform(.85, 1.08)
    kind = 'empty' if empty else ('mixed' if priority else 'rocks_only')
    scene = {'schema_version':5, 'simulator':'gazebo-harmonic', 'id':ident, 'seed':seed,
             'class_names':list(CLASS_NAMES), 'kind':kind,
             'scene_group':f'seed_{seed}', 'split':split,
             'config_sha256':config_hash, 'config':config, 'empty':empty, 'terrain':terrain,
             'ground':ground_recipe, 'dust_textures':dust_textures, 'objects':objects, 'regions':regions,
             'appearance_notes': NOTES if config.get('candidates', {}).get('enabled') else [],
             'views':views,
             'gravity':[0,0,-9.81],
             'lighting':{'azimuth':float(rng.uniform(0,math.tau)), 'elevation':float(rng.uniform(.35,1.2)),
                         'intensity':float(rng.uniform(.75,1.15)), 'ambient':float(rng.uniform(.32,.50)),
                         'sky_rgb':np.clip(sky,0,1).tolist()},
             'assets_sha256':digest(assets), 'ground_sha256':digest(list(ground))}
    scene['content_sha256'] = digest(scene)
    write_json(manifest_path, scene)
    write_world(root, scene, assets)
    return scene


def _material(maps, color, roughness=1.0):
    """Physically based material; the diffuse color tints the albedo map."""
    tags = ''.join(f'<{tag}>{escape(str(path))}</{tag}>' for tag, path in maps.items())
    return (f'<material><ambient>{color}</ambient><diffuse>{color}</diffuse><specular>0.05 0.05 0.05 1</specular>'
            f'<pbr><metal>{tags}<roughness>{roughness}</roughness><metalness>0</metalness></metal></pbr></material>')


def write_world(root, scene, assets):
    """Write host-specific absolute paths only here; manifests and hashes stay portable."""
    root = Path(root).resolve()
    folder = root/'scenes'/scene['id']
    by_id = {a['id']:a for a in assets}
    visual_mesh = f'<mesh><uri>{escape(str(folder/"terrain.dae"))}</uri></mesh>'
    collision_mesh = f'<mesh><uri>{escape(str(folder/"terrain_collision.dae"))}</uri></mesh>'
    if scene['ground']['layers']:
        terrain_maps = {'albedo_map': folder/'ground_albedo.jpg', 'normal_map': folder/'ground_normal.jpg',
                        'roughness_map': folder/'ground_roughness.jpg'}
    else:
        terrain_maps = {'albedo_map': folder/'ground.png'}
    rocks = []
    for obj in scene['objects']:
        asset = by_id[obj['asset_id']]
        mesh_path = root / obj.get('mesh', asset['mesh'])
        collision_path = root / obj.get('collision', asset['collision'])
        pose = ' '.join(f'{n:.12g}' for n in obj['xyz']+obj['rpy'])
        scale = ' '.join([str(obj['scale'])]*3)
        dusted = scene.get('dust_textures', {}).get(obj['asset_id'])
        if obj.get('material_maps'):
            color = '1 1 1 1'
            maps = {f'{key}_map': root/path for key, path in obj['material_maps'].items()}
        elif dusted:
            color = '1 1 1 1'
            maps = {f'{key}_map': root/path for key, path in dusted.items()}
        else:
            color = ' '.join(f'{c:.6g}' for c in obj['color'])+' 1'
            maps = {f'{key}_map': root/path for key, path in asset['textures'].items()}
        rocks.append(f'''
<model name="{obj['name']}"><static>true</static><pose>{pose}</pose><link name="link">
<collision name="collision"><geometry><mesh><uri>{escape(str(collision_path))}</uri><scale>{scale}</scale></mesh></geometry>
<surface><friction><ode><mu>0.9</mu><mu2>0.8</mu2></ode></friction></surface></collision>
<visual name="visual"><geometry><mesh><uri>{escape(str(mesh_path))}</uri><scale>{scale}</scale></mesh></geometry>
{_material(maps, color)}<cast_shadows>true</cast_shadows></visual>
</link><plugin filename="gz-sim-label-system" name="gz::sim::systems::Label"><label>{ROCK_LABEL}</label></plugin></model>''')
    light = scene['lighting']
    a,e,b = light['azimuth'],light['elevation'],light['intensity']
    ambient = light['ambient']
    sky = ' '.join(f'{c:.4f}' for c in light['sky_rgb'])
    text = f'''<?xml version="1.0"?>
<sdf version="1.9"><world name="{WORLD_NAME}">
<physics name="default" type="dart"><max_step_size>0.002</max_step_size><real_time_factor>1.0</real_time_factor></physics>
<plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
<plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
<plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
<plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
<gravity>0 0 -9.81</gravity>
<scene><ambient>{ambient} {ambient*.9:.4f} {ambient*.8:.4f} 1</ambient><background>{sky} 1</background>
<shadows>true</shadows><grid>false</grid></scene>
<light name="sun" type="directional"><pose>0 0 20 0 0 0</pose><cast_shadows>true</cast_shadows>
<diffuse>1 0.93 0.82 1</diffuse><specular>0.1 0.1 0.1 1</specular><intensity>{b}</intensity>
<direction>{math.cos(a)*math.cos(e)} {math.sin(a)*math.cos(e)} {-math.sin(e)}</direction></light>
<model name="mars_terrain"><static>true</static><link name="ground">
<collision name="terrain_collision"><geometry>{collision_mesh}</geometry><surface><friction><ode><mu>0.9</mu><mu2>0.8</mu2></ode></friction></surface></collision>
<visual name="terrain_visual"><geometry>{visual_mesh}</geometry>{_material(terrain_maps, '1 1 1 1')}<cast_shadows>true</cast_shadows></visual>
</link></model>{''.join(rocks)}
</world></sdf>'''
    (folder/'world.sdf').write_text(text, encoding='utf-8')


def generate_bundle(output, config, seeds=None, cache=None):
    output = Path(output).resolve()
    seeds = config['dataset']['seeds'] if seeds is None else seeds
    output.mkdir(parents=True, exist_ok=True)
    config_hash = digest(config)
    plan_path = output/'plan.json'
    if plan_path.exists():
        existing = json.loads(plan_path.read_text())
        require_label_schema(existing)
        if existing['config_sha256'] != config_hash:
            raise ValueError('Output contains another configuration; use a fresh directory')
    assets = build_library(output/'assets', config['rocks'], cache)
    ground = build_ground(output/'assets'/'ground', config['terrain']['ground_textures'], cache)
    scenes = [make_scene(output, config, int(seed), assets, ground) for seed in seeds]
    plan = {'schema_version':3, 'class_names':list(CLASS_NAMES), 'config_sha256':config_hash, 'config':config,
            'scenes':[{'id':s['id'],'seed':s['seed'],'split':s['split'], 'content_sha256':s['content_sha256'],
                       'views':len(s['views']), 'kind':s['kind']} for s in scenes],
            'reserved_meshes':[a['id'] for a in assets if a['held_out']]}
    write_json(plan_path, plan)
    write_json(output/'splits.json', {s['scene_group']:s['split'] for s in scenes})
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, action='append', help='Repeat for selected scenes; default is all configured seeds')
    parser.add_argument('--asset-cache', type=Path, default=None,
                        help='Prepared asset cache from mars_assets (default ./assets_cache or $MARS_ASSET_CACHE)')
    args = parser.parse_args(argv)
    plan = generate_bundle(args.output, read_config(args.config), args.seed, args.asset_cache or default_cache())
    print(f'Generated {len(plan["scenes"])} scenes / {sum(s["views"] for s in plan["scenes"])} planned images in {args.output}')


if __name__ == '__main__':
    main()
