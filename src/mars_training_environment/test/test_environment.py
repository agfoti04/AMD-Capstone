import copy
import json
import math
import tempfile
import unittest
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np

from mars_training_environment.annotation import labels_from_instances, rasterize
from mars_training_environment.common import digest, quaternion, read_config, rotation, split_for_seed
from mars_training_environment.generate import make_scene
from mars_training_environment.meshes import build_library, read_collada, sphere_mesh, write_dae
from mars_training_environment.terrain import Terrain, generate_terrain
from mars_training_environment.validate import validate_scene


K = {'width':80,'height':60,'fx':50.,'fy':50.,'cx':40.,'cy':30.}
POLICY = {'min_visible_pixels':6,'min_box_side_px':2,'min_visible_fraction':.12}


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


class GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.config = read_config()
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
        field = Terrain(t['heights'],t['size_m'])
        for x,y in [(0,0),(.4,.4),(-.4,.4)]:
            self.assertAlmostEqual(float(field.height(x,y)),0.,places=8)
        self.assertLessEqual(math.degrees(field.maximum_slope()),self.config['terrain']['max_slope_deg'])

    def test_surface_contact_uses_rotated_mesh_and_burial(self):
        field = Terrain(self.scene['terrain']['heights'],20)
        assets = {a['id']:a for a in self.assets}
        for obj in self.scene['objects']:
            v = np.array(json.loads((self.root/assets[obj['asset_id']]['geometry']).read_text())['vertices'])
            v = v @ rotation(obj['rpy']).T*obj['scale']+obj['xyz']
            clearance = v[:,2]-field.height(v[:,0],v[:,1])
            self.assertAlmostEqual(float(clearance.min()),-obj['burial_m'],places=8)

    def test_eval_shapes_and_complete_seed_groups_stay_separate(self):
        held = {a['id'] for a in self.assets if a['held_out']}
        self.assertEqual(len(self.assets),20)
        self.assertEqual(len(held),5)
        for seed in [100,101,107,114]:
            scene = make_scene(self.root,self.config,seed,self.assets)
            self.assertEqual(validate_scene(scene,self.assets),[])
            ids = {o['asset_id'] for o in scene['objects']}
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
        self.assertEqual(collision, visual)
        self.assertIn('terrain.dae', collision)
        self.assertEqual(world.findtext('.//gravity'),'0 0 -9.81')
        self.assertIn('Mars/ground', world.getroot().find('.//visual').findtext('material/script/name'))

    def test_heightmap_backend_shares_visual_and_collision(self):
        scene = copy.deepcopy(self.scene)
        scene['terrain']['backend'] = 'heightmap'
        from mars_training_environment.generate import write_world
        write_world(self.root, scene, self.assets)
        world = ET.parse(self.root/'scenes'/scene['id']/'world.sdf')
        model = world.find(".//model[@name='mars_terrain']")
        a = model.find('link/collision/geometry/heightmap')
        b = model.find('link/visual/geometry/heightmap')
        for key in ('uri','size','pos','sampling'):
            self.assertEqual(a.findtext(key), b.findtext(key))
        write_world(self.root, self.scene, self.assets)

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

    def test_configured_seeds_cover_every_split(self):
        counts = Counter(split_for_seed(seed) for seed in self.config['dataset']['seeds'])
        self.assertEqual(set(counts), {'train', 'val', 'test'})
        self.assertTrue(set(self.config['dataset']['empty_seeds']) <= set(self.config['dataset']['seeds']))

    def test_invalid_grid_and_operating_range_rejected(self):
        path=self.root/'bad.json'
        path.write_text(json.dumps({'terrain':{'grid_size':64}}))
        with self.assertRaises(ValueError):
            read_config(path)
        path.write_text(json.dumps({'capture':{'distance_m':[.1,4.]}}))
        with self.assertRaises(ValueError):
            read_config(path)


if __name__ == '__main__':
    unittest.main()
