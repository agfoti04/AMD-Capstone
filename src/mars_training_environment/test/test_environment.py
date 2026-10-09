import copy
import json
import math
import tempfile
import unittest
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np

from mars_training_environment.annotation import Annotator, compare_boxes, labels_from_instances, rasterize
from mars_training_environment.candidates import (
    drape_mesh, mudstone_texture, paint_ground_regions, stone_texture, terrain_slab_mesh, world_to_pixels)
from mars_training_environment.common import digest, quaternion, read_config, rotation, split_for_seed, write_json
from mars_training_environment.generate import make_scene
from mars_training_environment.dust import dust_images
from mars_training_environment.ground import broad_field, composite_ground, periodic_field
from mars_training_environment.meshes import build_ground, build_library, read_collada, sphere_mesh, write_dae
from mars_training_environment.robot import sync_mast
from mars_training_environment.terrain import Terrain, generate_terrain
from mars_training_environment.validate import validate_scene


K = {'width':80,'height':60,'fx':50.,'fy':50.,'cx':40.,'cy':30.}
POLICY = {'min_visible_pixels':6,'min_box_side_px':2,'min_visible_fraction':.12}
# Procedural and Fuel rocks only, so tests need no downloaded assets.
OFFLINE = {'rocks':{'procedural_count':16,'include_existing':True,'scanned_ids':[],
                    'held_out_ids':['proc_12','proc_13','proc_14','proc_15','falling_rock4']},
           'terrain':{'ground_textures':[], 'visual_grid_size':65}}


def offline_config(folder):
    path = Path(folder)/'offline.json'
    path.write_text(json.dumps(OFFLINE))
    return read_config(path)


def fake_cache(folder, rocks=(), ground=()):
    """A prepared asset cache with small stand-in meshes and textures."""
    import cv2
    cache = Path(folder)/'cache'
    rng = np.random.RandomState(3)
    for index, ident in enumerate(rocks):
        out = cache/'prepared'/'rocks'/ident
        out.mkdir(parents=True, exist_ok=True)
        v,f = sphere_mesh(index)
        uv = (v[:,:2]-v[:,:2].min(axis=0))/np.ptp(v[:,:2],axis=0)
        write_json(out/'visual.json', {'vertices':v.tolist(),'faces':f.tolist(),'uv':uv.tolist()})
        write_json(out/'collision.json', {'vertices':v.tolist(),'faces':f[::2].tolist()})
        write_json(out/'meta.json', {'id':ident,'page':f'https://polyhaven.com/a/{ident}','license':'CC0-1.0'})
        for key in ('albedo','normal','roughness'):
            cv2.imwrite(str(out/f'{key}.jpg'), rng.randint(0,255,(16,16,3)).astype(np.uint8))
    for ident in ground:
        out = cache/'prepared'/'ground'/ident
        out.mkdir(parents=True, exist_ok=True)
        write_json(out/'meta.json', {'id':ident,'page':'x','license':'CC0-1.0','size_m':2.0})
        cv2.imwrite(str(out/'albedo.jpg'), rng.randint(0,255,(64,64,3)).astype(np.uint8))
        cv2.imwrite(str(out/'normal.jpg'), np.full((64,64,3),(255,128,128),np.uint8))
        cv2.imwrite(str(out/'roughness.jpg'), rng.randint(0,255,(64,64)).astype(np.uint8))
    return cache


def square(x0=-.5,x1=.5,y0=-.5,y1=.5,z=2):
    return np.array([[x0,y0,z],[x1,y0,z],[x1,y1,z],[x0,y1,z]],dtype=float),np.array([[0,1,2],[0,2,3]])


class RasterTests(unittest.TestCase):
    def test_analytic_front_plane_depth_and_extent(self):
        v,f = square()
        d = rasterize(v,f,K,.3,8)
        self.assertAlmostEqual(float(d[30,40]),2.,places=6)
        yy,xx = np.nonzero(np.isfinite(d))
        self.assertLessEqual(abs(xx.min()-27),1)
        self.assertLessEqual(abs(xx.max()-52),1)
        self.assertTrue(np.allclose(d[np.isfinite(d)],2))

    def test_perspective_depth_is_not_linear_screen_depth(self):
        v = np.array([[-1,-1,2],[2,-2,4],[0,2,4.]])
        d = rasterize(v,[[0,1,2]],K,.3,8)
        # Ray/plane intersection gives an independent analytic depth check.
        normal = np.cross(v[1]-v[0],v[2]-v[0])
        ray = np.array([(.5+40-K['cx'])/K['fx'],(.5+30-K['cy'])/K['fy'],1])
        expected = normal.dot(v[0])/normal.dot(ray)
        self.assertAlmostEqual(float(d[30,40]),expected,places=5)

    def test_near_plane_and_image_boundary_clipping(self):
        v,f = square(-3,.5,-.5,.5,1.)
        v[0,2]=.05
        d = rasterize(v,f,K,.3,8)
        self.assertTrue(np.isfinite(d[:,0]).any())
        self.assertTrue((d[np.isfinite(d)]>=.3-1e-6).all())
        self.assertTrue(np.isinf(rasterize(v+np.array([0,0,-10]),f,K,.3,8)).all())

    def test_occluded_and_empty_objects_do_not_make_boxes(self):
        v,f = square()
        front = rasterize(v,f,K,.3,8)
        back = rasterize(v*1.5,f,K,.3,8)
        instances = np.where(np.isfinite(front),1,0)
        objects = [{'name':'front','asset_id':'a'},{'name':'back','asset_id':'b'}]
        labels,records = labels_from_instances(instances,[int(np.isfinite(front).sum()),int(np.isfinite(back).sum())],objects,POLICY)
        self.assertEqual([r['name'] for r in labels],['front'])
        self.assertEqual(records[1]['rejected_reason'],'fully_occluded')
        self.assertEqual(labels_from_instances(np.zeros((60,80),int),[],[],POLICY),([],[]))

    def test_partial_occlusion_burial_and_tiny_policy(self):
        instances = np.zeros((60,80),int)
        instances[20:35,0:10]=1
        instances[10:11,30:40]=2
        objects = [{'name':'buried','asset_id':'a'},{'name':'tiny','asset_id':'b'}]
        labels,records = labels_from_instances(instances,[300,10],objects,POLICY)
        self.assertEqual(len(labels),1)
        self.assertEqual(labels[0]['visible_fraction'],.5)
        self.assertTrue(labels[0]['truncated'])
        self.assertEqual(labels[0]['xmin'],0)
        self.assertEqual(records[1]['rejected_reason'],'too_small')

    def test_terrain_occludes_rock_below_its_surface(self):
        floor_v,f = square(-2,2,-2,2,1.5)
        rock_v,_ = square(-.5,.5,-.5,.5,2.)
        floor = rasterize(floor_v,f,K,.3,8)
        rock = rasterize(rock_v,f,K,.3,8)
        self.assertFalse((rock<floor).any())

    def test_batched_small_triangles_match_the_per_triangle_path(self):
        rng = np.random.RandomState(4)
        centers = np.column_stack([rng.uniform(-2,2,600),rng.uniform(-1.5,1.5,600),rng.uniform(.2,9,600)])
        v = (centers[:,None,:]+rng.normal(0,.04,(600,3,3))*centers[:,None,2:]).reshape(-1,3)
        f = np.arange(len(v)).reshape(-1,3)
        batched = rasterize(v,f,K,.3,8)
        reference = rasterize(v,f,K,.3,8,batch_small=False)
        self.assertGreater(np.isfinite(reference).sum(),200)
        np.testing.assert_array_equal(np.isfinite(batched),np.isfinite(reference))
        np.testing.assert_allclose(batched[np.isfinite(batched)],reference[np.isfinite(reference)],rtol=1e-6)


class GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.config = offline_config(cls.root)
        cls.assets = build_library(cls.root/'assets',cls.config['rocks'])
        cls.scene = make_scene(cls.root,cls.config,101,cls.assets)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_seed_and_config_reproduce_scene(self):
        again = make_scene(self.root,self.config,101,self.assets)
        self.assertEqual(self.scene,again)
        other = make_scene(self.root,self.config,102,self.assets)
        self.assertNotEqual(other['terrain']['heights'],self.scene['terrain']['heights'])

    def test_spawn_flat_slope_bounded_and_placement_nonintersecting(self):
        self.assertEqual(validate_scene(self.scene,self.assets),[])
        t = self.scene['terrain']
        visual = Terrain(t['heights'],t['size_m'])
        driveable = Terrain(t['base_heights'],t['size_m'])
        for x,y in [(0,0),(.4,.4),(-.4,.4)]:
            self.assertAlmostEqual(float(visual.height(x,y)),0.,places=8)
            self.assertAlmostEqual(float(driveable.height(x,y)),0.,places=8)
        self.assertLessEqual(math.degrees(driveable.maximum_slope()),self.config['terrain']['max_slope_deg'])

    def test_surface_contact_uses_rotated_mesh_and_burial(self):
        field = Terrain(self.scene['terrain']['heights'], self.scene['terrain']['size_m'])
        assets = {a['id']:a for a in self.assets}
        for obj in self.scene['objects']:
            path = obj.get('geometry', assets[obj['asset_id']]['geometry'])
            v = np.array(json.loads((self.root/path).read_text())['vertices'])
            v = v @ rotation(obj['rpy']).T*obj['scale']+obj['xyz']
            clearance = v[:,2]-field.height(v[:,0],v[:,1])
            self.assertAlmostEqual(float(clearance.min()),-obj['burial_m'],places=8)

    def test_mudstone_edges_are_embedded_and_rendered_mesh_matches_labels(self):
        field = Terrain(self.scene['terrain']['heights'], self.scene['terrain']['size_m'])
        annotated = {obj['name']: (v, f) for obj, v, f in Annotator(self.root, self.scene).meshes}
        world = ET.parse(self.root/'scenes'/self.scene['id']/'world.sdf')
        slabs = [o for o in self.scene['objects'] if o.get('placement') == 'terrain_following_outcrop']
        self.assertEqual({o['class_name'] for o in slabs}, {'cheyava_rock', 'veined_lookalike'})
        self.assertEqual(len({o['geometry'] for o in slabs}), len(slabs))
        ns = {'c': 'http://www.collada.org/2005/11/COLLADASchema'}
        for obj in slabs:
            model = world.find(f".//model[@name='{obj['name']}']")
            visual = Path(model.findtext('link/visual/geometry/mesh/uri'))
            self.assertEqual(visual, self.root/obj['mesh'])
            self.assertEqual(model.findtext('link/collision/geometry/mesh/uri'), str(visual))
            geometry = json.loads((self.root/obj['geometry']).read_text())
            self.assertEqual(digest(geometry), obj['geometry_sha256'])
            positions = ET.parse(visual).find(".//c:source[@id='positions']/c:float_array", ns).text
            local = np.fromstring(positions, sep=' ').reshape(-1, 3)
            np.testing.assert_allclose(local, geometry['vertices'], atol=1e-8)
            vertices = local @ rotation(obj['rpy']).T * obj['scale'] + obj['xyz']
            np.testing.assert_allclose(annotated[obj['name']][0], vertices, atol=1e-8)
            np.testing.assert_array_equal(annotated[obj['name']][1], geometry['faces'])
            clearance = vertices[:, 2] - field.height(vertices[:, 0], vertices[:, 1])
            count = len(vertices) // 2
            self.assertLessEqual(float(clearance[:count].max()), 0.008 + 1e-8)
            np.testing.assert_allclose(clearance[count:], -0.02, atol=1e-8)

    def test_eval_shapes_and_complete_seed_groups_stay_separate(self):
        held = {a['id'] for a in self.assets if a['held_out']}
        shared = {a['id'] for a in self.assets if a.get('shared_shape')}
        self.assertEqual(len(self.assets),23)
        self.assertEqual(shared, {'cand_sulfur', 'cand_pale_stone', 'cand_slab'})
        self.assertEqual(len(held),5)
        for seed in [100,101,107,114]:
            scene = make_scene(self.root,self.config,seed,self.assets)
            self.assertEqual(validate_scene(scene,self.assets),[])
            ids = {o['asset_id'] for o in scene['objects']} - shared
            self.assertFalse(ids-held if scene['split']=='test' else ids&held)
            self.assertEqual(scene['split'],split_for_seed(seed))
            self.assertEqual(len(scene['views']),10)

    def test_changed_config_cannot_overwrite_scene(self):
        changed = copy.deepcopy(self.config)
        changed['camera']['width']=800
        with self.assertRaisesRegex(ValueError,'another config'):
            make_scene(self.root,changed,101,self.assets)

    def test_standard_collada_indices_round_trip(self):
        v,f = sphere_mesh(3)
        path = self.root/'roundtrip.dae'
        write_dae(path,v,f)
        actual_v,actual_f = read_collada(path)
        np.testing.assert_allclose(actual_v,v,atol=1e-8)
        np.testing.assert_array_equal(actual_f,f)
        ns={'c':'http://www.collada.org/2005/11/COLLADASchema'}
        primitive=ET.parse(path).find('.//c:triangles',ns)
        self.assertEqual([i.get('offset') for i in primitive.findall('c:input',ns)],['0','1','2'])
        self.assertEqual(len(primitive.find('c:p',ns).text.split()),len(f)*9)

    def test_render_and_collision_share_the_terrain_mesh(self):
        world = ET.parse(self.root/'scenes'/self.scene['id']/'world.sdf')
        model = world.find(".//model[@name='mars_terrain']")
        collision = model.findtext('link/collision/geometry/mesh/uri')
        visual = model.findtext('link/visual/geometry/mesh/uri')
        self.assertIn('terrain_collision.dae', collision)
        self.assertIn('terrain.dae', visual)
        self.assertNotEqual(collision, visual)
        self.assertEqual(world.findtext('.//gravity'),'0 0 -9.81')
        self.assertIn('ground.png', model.findtext('link/visual/material/pbr/metal/albedo_map'))
        self.assertNotIn('libgazebo', (self.root/'scenes'/self.scene['id']/'world.sdf').read_text())
        for tag in ('ambient', 'diffuse', 'specular', 'background'):
            for element in world.iter(tag):
                values = [float(v) for v in element.text.split()]
                self.assertTrue(all(0 <= v <= 1 for v in values), f'<{tag}> {values} outside [0, 1]')
        self.assertEqual(float(world.findtext('.//light/intensity')), self.scene['lighting']['intensity'])

    def test_every_rock_is_labelled_for_the_gazebo_box_camera(self):
        world = ET.parse(self.root/'scenes'/self.scene['id']/'world.sdf')
        labeled = [m for m in world.iter('model') if m.find("plugin[@name='gz::sim::systems::Label']") is not None]
        self.assertEqual(len(labeled), len(self.scene['objects']))
        rocks = [m for m in labeled if m.get('name').startswith('rock_')]
        self.assertEqual(len(rocks), sum(1 for obj in self.scene['objects'] if obj['name'].startswith('rock_')))
        for model in labeled:
            plugin = model.find("plugin[@name='gz::sim::systems::Label']")
            self.assertEqual(plugin.findtext('label'), '1')
            self.assertTrue(Path(model.findtext('link/visual/material/pbr/metal/albedo_map')).is_file())
        sensors = world.find(".//plugin[@name='gz::sim::systems::Sensors']")
        self.assertEqual(sensors.findtext('render_engine'), 'ogre2')

    def test_ripples_are_on_the_visual_surface_and_off_the_spawn_pad(self):
        changed = copy.deepcopy(self.config)
        changed['terrain']['grid_size'] = 17
        changed['terrain']['visual_grid_size'] = 33
        fine = make_scene(self.root, changed, 251, self.assets)
        self.assertEqual(len(fine['terrain']['heights']), 33)
        self.assertEqual(len(fine['terrain']['base_heights']), 17)
        visual = Terrain(self.scene['terrain']['heights'], self.scene['terrain']['size_m'])
        base = Terrain(self.scene['terrain']['base_heights'], self.scene['terrain']['size_m'])
        points = [(x, y) for x in np.linspace(-8, 8, 7) for y in np.linspace(-8, 8, 7)
                  if math.hypot(x, y) > 3]
        self.assertGreater(max(abs(float(visual.height(x, y)-base.height(x, y))) for x, y in points), 0.01)
        self.assertAlmostEqual(float(visual.height(0, 0)), 0, places=8)

    def test_terrain_texture_coordinates_tile_at_the_configured_scale(self):
        ns = {'c':'http://www.collada.org/2005/11/COLLADASchema'}
        dae = ET.parse(self.root/'scenes'/self.scene['id']/'terrain.dae')
        uv = np.fromstring(dae.find(".//c:source[@id='uv']/c:float_array",ns).text,sep=' ').reshape(-1,2)
        size, tile = self.config['terrain']['size_m'], self.config['terrain']['texture_tile_m']
        np.testing.assert_allclose(uv.min(axis=0), [-size/2/tile]*2)
        np.testing.assert_allclose(uv.max(axis=0), [size/2/tile]*2)

    def test_pose_quaternion_matches_the_placement_rotation(self):
        rpy = (0.2, -0.35, 1.1)
        x, y, z, w = quaternion(rpy)
        matrix = np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
        ])
        np.testing.assert_allclose(matrix, rotation(rpy), atol=1e-9)

    def test_camera_stays_beyond_the_near_plane(self):
        lens = self.config['camera']['mount_xyz'][0]
        near = self.config['camera']['near_m']
        for view in self.scene['views']:
            x, y = view['xyz'][:2]
            yaw = view['rpy'][2]
            cx, cy = x+lens*math.cos(yaw), y+lens*math.sin(yaw)
            for obj in self.scene['objects']:
                gap = math.hypot(cx-obj['xyz'][0], cy-obj['xyz'][1]) - obj['radius_m']
                self.assertGreater(gap, near + 0.05)

    def test_sampling_sites_mix_into_scenes_and_stay_out_of_empty_ones(self):
        classes = {obj.get('class_name') for obj in self.scene['objects']}
        self.assertTrue({'sulfur_rock', 'pale_stone', 'cheyava_rock', 'veined_lookalike'} <= classes)
        self.assertEqual({region['class_name'] for region in self.scene['regions']}, {'silica_soil', 'pale_lookalike'})
        self.assertTrue((self.root/'scenes'/self.scene['id']/'candidate_mask.png').is_file())
        self.assertTrue(all(view['id'].startswith('view_') for view in self.scene['views']))
        self.assertTrue(self.scene['appearance_notes'])
        self.assertTrue(any('not chemical' in note for note in self.scene['appearance_notes']))
        empty = make_scene(self.root, self.config, 100, self.assets)
        self.assertEqual(empty['regions'], [])
        self.assertFalse(any(obj.get('class_name') for obj in empty['objects']))
        sulfur = next(obj for obj in self.scene['objects'] if obj.get('class_name') == 'sulfur_rock')
        self.assertTrue(sulfur['synthetic_analogue'])
        self.assertFalse(sulfur['exaggerated_analogue'])
        self.assertTrue((self.root/sulfur['material_maps']['albedo']).is_file())

    def test_configured_seeds_cover_every_split(self):
        counts = Counter(split_for_seed(seed) for seed in self.config['dataset']['seeds'])
        self.assertEqual(set(counts), {'train', 'val', 'test'})
        self.assertTrue(set(self.config['dataset']['empty_seeds']) <= set(self.config['dataset']['seeds']))

    def test_default_config_uses_scanned_rocks_and_ground(self):
        config = read_config()
        self.assertEqual(config['rocks']['procedural_count'], 0)
        self.assertTrue(set(config['rocks']['held_out_ids']) < set(config['rocks']['scanned_ids']))
        sources = json.loads((Path(__file__).resolve().parents[1]/'config'/'assets.json').read_text())
        self.assertEqual(set(config['rocks']['scanned_ids']), set(sources['rocks']['ids']))
        self.assertEqual(set(config['terrain']['ground_textures']), set(sources['ground']['ids']))

    def test_sample_camera_is_1080p_and_thirty_inches_up(self):
        from rover_training.geometry import optical_center_above_ground, parse_camera_extrinsics
        from rover_training.randomize import default_xacro
        text = default_xacro().read_text(encoding='utf-8')
        camera = read_config()['camera']
        self.assertEqual((camera['width'], camera['height'], camera['rate_hz']), (1920, 1080, 30.0))
        self.assertEqual(camera['horizontal_fov_status'], 'provisional')
        self.assertAlmostEqual(camera['horizontal_fov_rad'], 1.22, places=6)
        self.assertAlmostEqual(camera['optical_center_height_m'], 0.762, places=6)
        _rotation, translation = parse_camera_extrinsics(text)
        np.testing.assert_allclose(camera['mount_xyz'], translation, atol=1e-9)
        self.assertAlmostEqual(optical_center_above_ground(text), 0.762, places=6)
        self.assertLess(camera['mount_xyz'][2], 0.762)

    def test_mast_follows_the_joint_offset_not_the_ground_height(self):
        root = ET.fromstring('''<robot><link name="base_link">
          <visual name="chassis"><geometry><box size="0.4 0.2 0.1"/></geometry></visual>
          <visual name="camera_mast"><origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry><box size="0.02 0.02 0.1"/></geometry></visual>
        </link></robot>''')
        sync_mast(root, [0.2, 0.0, 0.682])
        origin = root.find("link/visual[@name='camera_mast']/origin").get('xyz').split()
        size = root.find("link/visual[@name='camera_mast']/geometry/box").get('size').split()
        self.assertAlmostEqual(float(origin[0]), 0.2, places=6)
        self.assertAlmostEqual(float(origin[2]), 0.366, places=6)
        self.assertAlmostEqual(float(size[2]), 0.632, places=6)
        self.assertAlmostEqual(float(origin[2]) + float(size[2]) * 0.5, 0.682, places=6)

    def test_invalid_grid_and_operating_range_rejected(self):
        path=self.root/'bad.json'
        path.write_text(json.dumps({'terrain':{'grid_size':64}}))
        with self.assertRaises(ValueError):
            read_config(path)
        path.write_text(json.dumps({'capture':{'distance_m':[.1,4.]}}))
        with self.assertRaises(ValueError):
            read_config(path)


class ScannedAssetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_prepared_rocks_keep_uvs_textures_and_simplified_collision(self):
        cache = fake_cache(self.root, rocks=['scan_a','scan_b','scan_c','scan_d'])
        config = offline_config(self.root)['rocks']
        config.update(procedural_count=0, include_existing=False, scanned_ids=['scan_a','scan_b','scan_c','scan_d'],
                      held_out_ids=['scan_d'])
        assets = build_library(self.root/'assets', config, cache)
        first = assets[0]
        self.assertEqual(first['license'], 'CC0-1.0')
        self.assertEqual(set(first['textures']), {'albedo','normal','roughness'})
        for path in first['textures'].values():
            self.assertTrue((self.root/path).is_file())
        self.assertLess(first['collision_triangles'], first['triangles'])
        ns = {'c':'http://www.collada.org/2005/11/COLLADASchema'}
        dae = ET.parse(self.root/first['mesh'])
        uv = np.fromstring(dae.find(".//c:source[@id='uv']/c:float_array",ns).text,sep=' ').reshape(-1,2)
        expected = json.loads((cache/'prepared/rocks/scan_a/visual.json').read_text())['uv']
        np.testing.assert_allclose(uv, expected, atol=1e-8)
        geometry = json.loads((self.root/first['geometry']).read_text())
        self.assertEqual(len(geometry['faces']), first['triangles'])

    def test_dust_coat_lifts_a_dark_upward_face(self):
        albedo = np.zeros((32, 32, 3), np.uint8)
        roughness = np.full((32, 32), 30, np.uint8)
        vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 1], [1, 0, -1]]
        uvs = [[0, 0], [1, 0], [0, 1], [1, 1]]
        faces = [[0, 1, 2], [0, 1, 3]]
        coated, coated_rough = dust_images(albedo, roughness, vertices, uvs, faces, [40, 90, 150])
        self.assertGreater(float(coated.mean()), 30)
        self.assertGreater(float(coated_rough.mean()), 30)
        # The upward face (brighter coat) and the vertical face land on opposite
        # sides of this test image.
        self.assertGreater(float(coated[21, 8].mean()), float(coated[10, 24].mean()))

    def test_missing_cache_explains_how_to_prepare(self):
        config = offline_config(self.root)['rocks']
        config.update(scanned_ids=['not_downloaded'])
        with self.assertRaisesRegex(FileNotFoundError,'mars_assets'):
            build_library(self.root/'assets', config, self.root/'empty_cache')

    def test_ground_blend_is_seamless_and_reproducible(self):
        field = broad_field(np.random.RandomState(2), 64)
        self.assertGreater(float(np.abs(field[:, 0]-field[:, -1]).max()), 0.05)
        field = periodic_field(np.random.RandomState(1), 128)
        step = np.abs(np.diff(field,axis=1)).max()
        self.assertLessEqual(np.abs(field[:,0]-field[:,-1]).max(), step+1e-5)
        cache = fake_cache(self.root, ground=['g1','g2','g3'])
        catalog = build_ground(self.root/'assets'/'ground', ['g1','g2','g3'], cache)
        terrain = dict(read_config()['terrain'], composite_px=256)
        first = composite_ground(self.root, catalog, terrain, 5, self.root)
        image = (self.root/'ground_albedo.jpg').read_bytes()
        again = composite_ground(self.root, catalog, terrain, 5, self.root)
        self.assertEqual(first, again)
        self.assertEqual((self.root/'ground_albedo.jpg').read_bytes(), image)
        import cv2
        normal = cv2.imread(str(self.root/'ground_normal.jpg')).astype(float)/127.5-1
        self.assertLess(abs(np.median(np.linalg.norm(normal,axis=2))-1), .05)


class GazeboBoxCheckTests(unittest.TestCase):
    LABELS = [{'name':'a','xmin':10,'ymin':10,'xmax':30,'ymax':30},
              {'name':'b','xmin':50,'ymin':10,'xmax':70,'ymax':40}]
    OBJECTS = [{'name':'a','rejected_reason':None,'bbox_xyxy':[10,10,30,30]},
               {'name':'b','rejected_reason':None,'bbox_xyxy':[50,10,70,40]},
               {'name':'c','rejected_reason':'occluded_below_fraction','bbox_xyxy':[0,40,8,48]}]
    POLICY = {'bbox_min_iou':.5,'min_box_side_px':5,'min_visible_pixels':24}

    def test_matching_boxes_agree(self):
        result = compare_boxes(self.LABELS, self.OBJECTS, [[11,10,30,31],[50,11,70,40]], self.POLICY)
        self.assertTrue(result['agrees'])
        self.assertEqual(result['matched'], 2)
        self.assertGreater(result['min_matched_iou'], .85)

    def test_gazebo_box_for_a_policy_rejected_rock_is_expected(self):
        result = compare_boxes(self.LABELS, self.OBJECTS, [[10,10,30,30],[50,10,70,40],[0,40,8,48]], self.POLICY)
        self.assertTrue(result['agrees'])
        self.assertEqual(result['gazebo_boxes_for_rejected_rocks'], 1)

    def test_gazebo_sliver_below_label_policy_is_expected(self):
        result = compare_boxes(self.LABELS, self.OBJECTS, [[10,10,30,30],[50,10,70,40],[90,90,94,91]], self.POLICY)
        self.assertTrue(result['agrees'])
        self.assertEqual(result['gazebo_boxes_below_policy'], 1)

    def test_missing_or_unexplained_boxes_disagree(self):
        missing = compare_boxes(self.LABELS, self.OBJECTS, [[10,10,30,30]], self.POLICY)
        self.assertFalse(missing['agrees'])
        self.assertEqual(missing['labels_without_gazebo_match'], ['b'])
        extra = compare_boxes(self.LABELS, self.OBJECTS, [[10,10,30,30],[50,10,70,40],[90,90,99,99]], self.POLICY)
        self.assertFalse(extra['agrees'])
        self.assertEqual(extra['gazebo_boxes_unexplained'], 1)
        self.assertEqual(extra['unexplained_boxes'][0]['xyxy'], [90.0,90.0,99.0,99.0])
        shifted = compare_boxes(self.LABELS[:1], self.OBJECTS[:1], [[20,20,40,40]], self.POLICY)
        self.assertFalse(shifted['agrees'])

    def test_ground_regions_are_not_required_to_match_a_gazebo_box(self):
        labels = [{'name':'silica_soil_000','xmin':10,'ymin':10,'xmax':40,'ymax':40,'source':'ground_region'}]
        objects = [{'name':'silica_soil_000','rejected_reason':None,'bbox_xyxy':[10,10,40,40],'source':'ground_region'}]
        result = compare_boxes(labels, objects, [], self.POLICY)
        self.assertTrue(result['agrees'])
        self.assertEqual(result['labels_without_gazebo_match'], [])


class CandidateTests(unittest.TestCase):
    def test_mudstone_follows_ripples_without_air_gaps_or_bridging_grid_diagonals(self):
        coords = np.linspace(-1, 1, 17)
        x, y = np.meshgrid(coords, coords)
        field = Terrain(0.12*x + 0.04*y + 0.04*np.cos(3*x)*np.sin(4*y), 2)
        # Concave but star-shaped footprint, including terrain-grid crossings.
        polygon = [[-0.6, -0.45], [0.45, -0.45], [0.25, -0.1], [0.6, 0.4], [-0.45, 0.6]]
        vertices, faces = terrain_slab_mesh(field, polygon)
        count = len(vertices) // 2
        clearance = vertices[:, 2] - field.height(vertices[:, 0], vertices[:, 1])
        self.assertGreaterEqual(float(clearance[:count].min()), 0.001 - 1e-9)
        self.assertLessEqual(float(clearance[:count].max()), 0.008 + 1e-9)
        np.testing.assert_allclose(clearance[count:], -0.02, atol=1e-9)
        top = faces[np.all(faces < count, axis=1)]
        centers = vertices[top].mean(axis=1)
        reveal = centers[:, 2] - field.height(centers[:, 0], centers[:, 1])
        self.assertGreaterEqual(float(reveal.min()), 0.001 - 1e-9)
        self.assertLessEqual(float(reveal.max()), 0.008 + 1e-9)
        area = np.cross(vertices[top[:, 1]] - vertices[top[:, 0]], vertices[top[:, 2]] - vertices[top[:, 0]])[:, 2].sum()/2
        p = np.asarray(polygon)
        expected = abs(np.sum(p[:, 0]*np.roll(p[:, 1], -1) - p[:, 1]*np.roll(p[:, 0], -1)))/2
        self.assertAlmostEqual(float(area), float(expected), places=8)
        edges = Counter(tuple(sorted((int(a), int(b)))) for face in faces
                        for a, b in zip(face, np.roll(face, -1)))
        self.assertEqual(set(edges.values()), {2}, 'Outcrop mesh must be watertight')

    def test_positive_y_maps_toward_the_top_of_the_albedo(self):
        _col, north = world_to_pixels([[0, 5]], 20, 101)
        _col, south = world_to_pixels([[0, -5]], 20, 101)
        self.assertLess(float(np.asarray(north).reshape(-1)[0]), float(np.asarray(south).reshape(-1)[0]))

    def test_pale_controls_stay_near_the_silica_brightness(self):
        image = np.full((180, 180, 3), (70, 110, 160), np.float32)
        regions = [
            {'class_name':'silica_soil','mask_id':1,'dust_cover':0.25,
             'polygon_xy':[[-1.5,-1.2],[1.6,-0.8],[1.2,1.1],[-1.4,0.9]]},
            {'class_name':'pale_lookalike','mask_id':2,'dust_cover':0.25,
             'polygon_xy':[[-1.5,2.0],[1.5,1.8],[1.4,3.6],[-1.6,3.4]]},
        ]
        painted, _normal, _rough, mask = paint_ground_regions(
            image, None, None, regions, 20.0, np.random.RandomState(1))
        silica = painted[mask == 1].mean(axis=1)
        lookalike = painted[mask == 2].mean(axis=1)
        self.assertGreater(len(silica), 20)
        self.assertGreater(len(lookalike), 20)
        self.assertLess(abs(float(silica.mean()) - float(lookalike.mean())) / float(lookalike.mean()), 0.2)
        self.assertLess(float(silica.min()), float(lookalike.min()) - 30)

    def test_sulfur_yellow_stays_on_the_fracture_face(self):
        dust = [150, 110, 80]
        fracture = stone_texture(np.random.RandomState(3), 128, True, dust)[0]
        intact = stone_texture(np.random.RandomState(3), 128, False, dust)[0]

        def yellowness(image):
            return float(image[:, :, 0].mean() + image[:, :, 1].mean() - 2 * image[:, :, 2].mean())

        self.assertGreater(yellowness(fracture[:, 100:]), yellowness(fracture[:, :28]) + 40)
        self.assertLess(abs(yellowness(intact) - yellowness(fracture[:, :28])), 30)
        purple = (fracture[:, :, 0] > 160) & (fracture[:, :, 2] > 160) & (fracture[:, :, 1] < 80)
        self.assertEqual(int(purple.sum()), 0)

    def test_cheyava_spots_are_rimmed_and_exaggeration_makes_them_larger(self):
        dust = [140, 100, 70]
        physical = mudstone_texture(np.random.RandomState(4), 256, True, 1.0, 1.0, dust)[0]
        exaggerated = mudstone_texture(np.random.RandomState(4), 256, True, 8.0, 1.0, dust)[0]
        veins_only = mudstone_texture(np.random.RandomState(4), 256, False, 1.0, 1.0, dust)[0]

        def centers(image):
            return int(((image[:, :, 0] > 220) & (image[:, :, 1] > 210) & (image[:, :, 2] > 200)).sum())

        self.assertEqual(centers(veins_only), 0)
        self.assertGreater(centers(physical), 20)
        self.assertGreater(centers(exaggerated), centers(physical) * 3)
        pixl = (physical[:, :, 2] > 150) & (physical[:, :, 0] > 150) & (physical[:, :, 1] < 80)
        self.assertEqual(int(pixl.sum()), 0)

    def test_a_ground_region_drapes_onto_the_terrain(self):
        field = Terrain(np.zeros((17, 17)), 20)
        vertices, faces = drape_mesh(field, [[-0.3, -0.2], [0.3, -0.2], [0.3, 0.2], [-0.3, 0.2]])
        self.assertGreater(len(faces), 4)
        self.assertAlmostEqual(float(vertices[:, 2].mean()), 0.008, places=3)

    def test_region_labels_keep_their_class_id(self):
        instances = np.zeros((60, 80), int)
        instances[10:30, 10:30] = 1
        objects = [{'name':'silica_soil_000','asset_id':'silica_soil','class_name':'silica_soil','source':'ground_region'}]
        labels, _records = labels_from_instances(instances, [400], objects, POLICY)
        self.assertEqual(labels[0]['class_id'], 1)
        self.assertEqual(labels[0]['class_name'], 'silica_soil')
        self.assertEqual(labels[0]['source'], 'ground_region')


if __name__ == '__main__':
    unittest.main()
