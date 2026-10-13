import copy
import json
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

from mars_training_environment.annotation import check_sky_rgb, instance_depth_support, discard_confirmed_sky_labels, labels_from_instances, rasterize
from mars_training_environment.common import png, read_config, write_json
from mars_training_environment.dataset import capture_failure_hint
from mars_training_environment.framing import configured_extrinsics, planning_size_guard, target_framing
from mars_training_environment.quality import frame_checks
from mars_training_environment.review import ReviewStore, make_server, review_index, write_review_page
from rover_training.geometry import CLASS_NAMES, parse_camera_extrinsics
from rover_training.randomize import default_xacro
from rover_training.yolo_export import read_label_file, write_dataset_index, write_label_file


def labels():
    return [{'name':'real','class_id':0,'xmin':10,'ymin':10,'xmax':25,'ymax':25},
            {'name':'ghost','class_id':0,'xmin':40,'ymin':10,'xmax':55,'ymax':25}]


class VisibilityTests(unittest.TestCase):
    def setUp(self):
        self.instances = np.zeros((60,80),np.int32)
        self.instances[10:25,10:25] = 1
        self.instances[10:25,40:55] = 2
        self.expected = np.full((60,80),2.,np.float32)
        self.objects = [{'name':'real'},{'name':'ghost'}]
        self.policy = read_config()['labels']

    def test_invalid_depth_object_cannot_hide_behind_good_aggregate_median(self):
        measured = self.expected.copy()
        measured[self.instances==2] = np.inf
        checks = instance_depth_support(self.instances,self.expected,measured,self.objects,labels(),self.policy,.3,8)
        self.assertTrue(checks[0]['passed'])
        self.assertFalse(checks[1]['passed'])
        self.assertEqual(checks[1]['valid_depth_pixels'],0)
        self.assertEqual(checks[1]['support_fraction'],0)

    def test_unrelated_ground_in_box_cannot_support_a_wrong_object_surface(self):
        measured = self.expected.copy()
        measured[self.instances==2] = 4
        checks = instance_depth_support(self.instances,self.expected,measured,self.objects,labels(),self.policy,.3,8)
        self.assertFalse(checks[1]['passed'])
        self.assertEqual(checks[1]['median_error_m'],2)

    def test_missing_depth_is_rejected_and_empty_scene_is_valid(self):
        checks = instance_depth_support(self.instances,self.expected,None,self.objects,labels(),self.policy,.3,8)
        self.assertTrue(all('sensor_depth_missing' in c['reasons'] for c in checks))
        self.assertEqual(instance_depth_support(self.instances*0,self.expected,self.expected,[],[],self.policy,.3,8),[])

    def test_ground_patch_uses_expected_ground_depth(self):
        objects = [{'name':'real','source':'ground_region'},{'name':'ghost'}]
        checks = instance_depth_support(self.instances,self.expected,self.expected,objects,labels(),self.policy,.3,8)
        self.assertTrue(checks[0]['passed'])
        self.assertEqual(checks[0]['source'],'ground_region')

    def test_missing_expected_surface_has_no_nonfinite_error_in_diagnostics(self):
        expected = self.expected.copy()
        expected[self.instances==2] = np.inf
        checks = instance_depth_support(self.instances,expected,self.expected,self.objects,labels(),self.policy,.3,8)
        self.assertFalse(checks[1]['passed'])
        self.assertIsNone(checks[1]['median_error_m'])
        json.dumps(checks,allow_nan=False)

    def test_rgb_only_sky_box_rejected_but_uniform_foreground_not_rejected(self):
        rgb = np.full((60,80,3),[231,206,181],np.uint8)
        rgb[self.instances==1] = [180,140,100]
        checks = instance_depth_support(self.instances,self.expected,self.expected,self.objects,labels(),self.policy,.3,8)
        check_sky_rgb(rgb.tobytes(),self.instances,self.objects,checks)
        self.assertTrue(checks[0]['passed'])
        self.assertFalse(checks[1]['passed'])
        self.assertIn('label_covers_only_sky_rgb',checks[1]['reasons'])

    def test_capture_requires_a_check_for_every_exported_object(self):
        quality = {'visibility_check':{'checked':True,'objects':[]}}
        result = frame_checks({},labels(),quality,{'width':80,'height':60},read_config()['capture'])
        self.assertIn('per_object_visibility_checks_incomplete',result['errors'])


    def sky_quality(self, measured):
        rgb = np.full((60,80,3),[231,206,181],np.uint8)
        rgb[self.instances==1] = [180,140,100]
        checks = instance_depth_support(self.instances,self.expected,measured,self.objects,labels(),self.policy,.3,8)
        check_sky_rgb(rgb.tobytes(),self.instances,self.objects,checks)
        return {'visibility_check':{'checked':measured is not None,'objects':checks}}

    def test_confirmed_invisible_mesh_is_removed_with_failed_evidence_preserved(self):
        measured = self.expected.copy()
        measured[self.instances==2] = np.inf
        quality = self.sky_quality(measured)
        retained = discard_confirmed_sky_labels(labels(),self.objects,quality)
        self.assertEqual([label['name'] for label in retained],['real'])
        self.assertEqual(self.objects[1]['rejected_reason'],'renderer_invisible_sky')
        rejected = quality['visibility_check']['rejected_objects'][0]
        self.assertEqual(rejected['name'],'ghost')
        self.assertFalse(rejected['passed'])
        self.assertEqual(rejected['sensor_valid_depth_pixels'],0)
        self.assertEqual(len(rejected['reasons']),2)
        self.assertTrue(frame_checks({},retained,quality,{'width':80,'height':60},read_config()['capture'])['passed'])
        self.assertIn('target_not_visible',frame_checks({'target_name':'ghost'},retained,quality,
                     {'width':80,'height':60},read_config()['capture'])['errors'])

    def test_finite_depth_mismatch_still_rejects_the_frame(self):
        measured = self.expected.copy()
        measured[self.instances==2] = 4
        quality = self.sky_quality(measured)
        retained = discard_confirmed_sky_labels(labels(),self.objects,quality)
        self.assertEqual(len(retained),2)
        self.assertEqual(quality['visibility_check']['rejected_objects'],[])
        self.assertFalse(frame_checks({},retained,quality,{'width':80,'height':60},read_config()['capture'])['passed'])

    def test_visible_sky_colored_mesh_is_not_silently_removed(self):
        quality = self.sky_quality(self.expected)
        self.assertEqual(len(discard_confirmed_sky_labels(labels(),self.objects,quality)),2)
        self.assertIn('ghost',quality['visibility_check']['failures'])

    def test_missing_sensor_and_invalid_expected_geometry_are_not_filtered(self):
        quality = self.sky_quality(None)
        self.assertEqual(len(discard_confirmed_sky_labels(labels(),self.objects,quality)),2)
        self.expected[self.instances==2] = np.inf
        measured = np.full_like(self.expected,2.)
        quality = self.sky_quality(measured)
        self.assertEqual(len(discard_confirmed_sky_labels(labels(),self.objects,quality)),2)
        self.assertEqual(quality['visibility_check']['objects'][1]['sensor_valid_depth_pixels'],225)

    def test_invalid_depth_without_sky_evidence_is_not_filtered(self):
        measured = self.expected.copy()
        measured[self.instances==2] = np.inf
        checks = instance_depth_support(self.instances,self.expected,measured,self.objects,labels(),self.policy,.3,8)
        quality = {'visibility_check':{'checked':True,'objects':checks}}
        self.assertEqual(len(discard_confirmed_sky_labels(labels(),self.objects,quality)),2)

    def test_ground_appearance_mismatch_is_not_filtered(self):
        self.objects[1]['source'] = 'ground_region'
        measured = self.expected.copy()
        measured[self.instances==2] = np.inf
        quality = self.sky_quality(measured)
        self.assertEqual(len(discard_confirmed_sky_labels(labels(),self.objects,quality)),2)


class CaptureDiagnosticsTests(unittest.TestCase):
    def test_capture_error_is_reported_before_unrelated_shutdown_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'scene.log'
            path.write_text('[mars_capture-7] ValueError: Frame rejected before export: ghost\n'
                            '[twist_stamper-5] RCLError: shutdown already called\n')
            self.assertEqual(capture_failure_hint(path),'ValueError: Frame rejected before export: ghost')

    def test_missing_capture_error_produces_no_misleading_hint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'scene.log'
            self.assertEqual(capture_failure_hint(path),'')
            path.write_text('[twist_stamper-5] RCLError: shutdown already called\n')
            self.assertEqual(capture_failure_hint(path),'')


class FramingTests(unittest.TestCase):
    def setUp(self):
        self.policy = read_config()['capture']
        self.k = {'width':1920,'height':1080}
        self.target = {'name':'target','xmin':300,'ymin':200,'xmax':600,'ymax':500,
                       'visible_fraction':.8,'truncated':False}

    def test_target_can_pass_while_incidental_rock_is_small_or_clipped(self):
        labels = [self.target,dict(self.target,name='incidental',xmin=0,xmax=5,truncated=True)]
        check = target_framing(labels,'target',self.k,self.policy)
        self.assertTrue(check['passed'])
        self.assertEqual(check['side_at_training_px'],150)

    def test_target_must_meet_margin_visibility_and_training_pixel_size(self):
        for changes, reason in [({'xmin':0},'target_outside_frame_margin'),
                                ({'visible_fraction':.5},'target_too_occluded'),
                                ({'xmax':330},'target_too_small_at_training_resolution')]:
            check = target_framing([dict(self.target,**changes)],'target',self.k,self.policy)
            self.assertIn(reason,check['reasons'])

    def test_difficult_view_allows_edge_case_but_not_missing_target(self):
        target = dict(self.target,xmin=0,truncated=True,visible_fraction=.4)
        self.assertTrue(target_framing([target],'target',self.k,self.policy,True)['passed'])
        self.assertFalse(target_framing([],'target',self.k,self.policy,True)['passed'])

    def test_planner_camera_matches_render_camera(self):
        actual = configured_extrinsics(read_config()['camera'])
        expected = parse_camera_extrinsics(default_xacro().read_text())
        for a,b in zip(actual,expected):
            np.testing.assert_allclose(a,b,atol=1e-10)

    def test_coarse_raster_size_cannot_approve_an_undersized_full_resolution_target(self):
        # A 4.4-pixel projected extent can cover 5 coarse pixel centers, but
        # measures 22 pixels at 1920: 12.5 versus 11 pixels at training size 960.
        vertices = np.array([[100.4,80.4,1],[104.8,80.4,1],
                             [104.8,84.8,1],[100.4,84.8,1]])
        vertices[:,:2] /= 100  # Keep the fixture inside the sensor's radial range.
        faces = [[0,1,2],[0,2,3]]
        measured = []
        for scale in (1,5):
            k = {'width':384*scale,'height':216*scale,'fx':100*scale,'fy':100*scale,'cx':0,'cy':0}
            mask = np.isfinite(rasterize(vertices,faces,k,.3,8))
            visible, _ = labels_from_instances(mask.astype(int),[int(mask.sum())],
                                               [{'name':'target','asset_id':'test'}],
                                               {'min_visible_pixels':1,'min_visible_fraction':.3,'min_box_side_px':1})
            check = target_framing(visible,'target',k,self.policy,True)
            measured.append(check['side_at_training_px'])
            if scale == 1:
                self.assertTrue(check['passed'])  # The old planner's false positive.
                guarded = target_framing(visible,'target',k,self.policy,True,
                                          size_guard_px=planning_size_guard(k,self.policy))
                self.assertFalse(guarded['passed'])
            else:
                self.assertFalse(check['passed'])
        self.assertEqual(measured,[12.5,11.])

    def test_invalid_visibility_and_resolution_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'bad.json'
            for supplied in ({'capture':{'require_depth_check':False}},
                             {'capture':{'training_imgsz':950}},
                             {'labels':{'depth_support_fraction':0}}):
                path.write_text(json.dumps(supplied))
                with self.assertRaises(ValueError):
                    read_config(path)


def review_fixture(root):
    write_dataset_index(root)
    for index,split in enumerate(('train','val','test')):
        stem = f'scene_{101+index:06d}_view_000'
        image_path = f'images/{split}/{stem}.png'
        label_path = f'labels/{split}/{stem}.txt'
        label = dict(labels()[0],class_id=index,class_name=CLASS_NAMES[index])
        (root/image_path).parent.mkdir(parents=True,exist_ok=True)
        png(root/image_path,np.full((60,80,3),[150,100,50],np.uint8))
        write_label_file(root/label_path,[label],80,60)
        write_json(root/'metadata'/split/(stem+'.json'),
                   {'scene_id':f'scene_{101+index:06d}','split':split,'view':'view_000',
                    'image':image_path,'label':label_path,'intrinsics':{'width':80,'height':60},
                    'labels':[label],'depth_quality':{'visibility_check':{'checked':True}},
                    'capture_checks':{'passed':True}})
    write_json(root/'validation.json',{'errors':[],'frames':3})


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        review_fixture(self.root)
        self.store = ReviewStore(self.root)
        self.frame = review_index(self.root)['frames'][0]

    def tearDown(self):
        self.tmp.cleanup()

    def test_page_populates_from_metadata_and_refresh_includes_new_frames(self):
        path = write_review_page(self.root)
        self.assertIn(self.frame['id'],path.read_text())
        self.assertNotIn('__REVIEW_DATA__',path.read_text())
        self.assertEqual(json.loads((self.root/'review/index.json').read_text())['accepted_frames'],3)

    def test_removed_invisible_label_is_flagged_in_the_review(self):
        path = self.root/'metadata'/self.frame['split']/(self.frame['id']+'.json')
        record = json.loads(path.read_text())
        record['depth_quality']['visibility_check']['rejected_objects'] = [{'name':'ghost'}]
        write_json(path,record)
        frame = review_index(self.root)['frames'][0]
        self.assertTrue(frame['accepted'])
        self.assertIn('renderer_invisible_label_removed: ghost',frame['flags'])

    def test_reframed_target_is_flagged_in_the_review(self):
        path = self.root/'metadata'/self.frame['split']/(self.frame['id']+'.json')
        record = json.loads(path.read_text())
        record['framing_adjustments'] = [{'reason':'target_too_small_at_training_resolution'}]
        write_json(path,record)
        frame = review_index(self.root)['frames'][0]
        self.assertTrue(frame['accepted'])
        self.assertIn('target_reframed: 1',frame['flags'])

    def test_earlier_ground_projection_is_flagged_until_manually_reviewed(self):
        path = self.root/'metadata'/self.frame['split']/(self.frame['id']+'.json')
        record = json.loads(path.read_text())
        record['labels'][0]['source'] = 'ground_region'
        write_json(path,record)
        frame = review_index(self.root)['frames'][0]
        self.assertIn('legacy_ground_region_projection',frame['flags'])
        record['depth_quality']['ground_region_projection'] = 'visible_terrain_surface'
        write_json(path,record)
        frame = review_index(self.root)['frames'][0]
        self.assertNotIn('legacy_ground_region_projection',frame['flags'])

    def test_reframing_diagnostics_remain_separate_review_entries(self):
        base = self.root/'diagnostics'/'scene_000102'/'view_004'
        for folder in (base,base/'reframe_01'):
            folder.mkdir(parents=True,exist_ok=True)
            png(folder/'rgb.png',np.zeros((60,80,3),np.uint8))
            write_json(folder/'camera.json',{'intrinsics':{'width':80,'height':60},
                       'scene_id':'scene_000102','view':{'target_name':'rock_004'},
                       'checks':{'errors':['target_too_small_at_training_resolution']}})
        frames = [frame for frame in review_index(self.root)['frames'] if not frame['accepted']]
        self.assertEqual(len(frames),2)
        self.assertEqual(len({frame['id'] for frame in frames}),2)
        self.assertTrue(all(frame['scene_id']=='scene_000102' for frame in frames))

    def test_capture_and_editor_can_refresh_the_page_concurrently(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda _index:write_review_page(self.root),range(12)))
        self.assertEqual(json.loads((self.root/'review/index.json').read_text())['accepted_frames'],3)
        self.assertFalse(list((self.root/'review').glob('*.tmp')))

    def test_edit_export_preserves_raw_capture_and_keeps_original_split(self):
        source = self.root/self.frame['image']
        original_image = source.read_bytes()
        original_label = next((self.root/'labels').glob('*/*.txt')).read_text()
        changed = copy.deepcopy(self.frame['labels'])
        changed[0].update(class_id=0,xmax=30)
        self.store.save(self.frame['id'],self.frame['revision'],changed,note='Corrected class and extent')
        report = self.store.export()
        output = Path(report['data_yaml']).parent
        self.assertEqual(source.read_bytes(),original_image)
        self.assertEqual(next((self.root/'labels').glob('*/*.txt')).read_text(),original_label)
        exported = read_label_file(output/'labels'/self.frame['split']/(source.stem+'.txt'),80,60)
        self.assertEqual(exported[0]['class_id'],0)
        self.assertAlmostEqual(exported[0]['xmax'],30,places=3)
        self.assertEqual(report['frames'],3)
        self.assertEqual(report['split_counts'],{'test':1,'train':1,'val':1})
        self.assertTrue((output/'review_export.json').is_file())

    def test_exclusion_is_reversible_and_audited(self):
        edited = self.store.save(self.frame['id'],self.frame['revision'],[],True,note='Bad frame')
        self.assertEqual(review_index(self.root)['excluded_frames'],1)
        self.assertTrue((self.root/self.frame['image']).exists())
        restored = self.store.restore(self.frame['id'],edited['revision'])
        self.assertFalse(restored['excluded'])
        self.assertEqual(restored['labels'],self.frame['original_labels'])
        self.assertEqual(len((self.root/'review/history.jsonl').read_text().splitlines()),2)

    def test_stale_revision_invalid_box_and_unknown_class_cannot_be_saved(self):
        changed = copy.deepcopy(self.frame['labels'])
        for changes in ({'class_id':7},{'xmin':-1},{'xmax':float('nan')}):
            changed[0].update(changes)
            with self.assertRaises(ValueError):
                self.store.save(self.frame['id'],self.frame['revision'],changed)
            changed = copy.deepcopy(self.frame['labels'])
        self.store.save(self.frame['id'],self.frame['revision'],changed)
        with self.assertRaisesRegex(ValueError,'changed since'):
            self.store.save(self.frame['id'],self.frame['revision'],changed)

    def test_recaptured_frame_does_not_apply_old_boxes_or_break_live_capture_review(self):
        changed = copy.deepcopy(self.frame['labels'])
        changed[0]['class_id'] = 0
        self.store.save(self.frame['id'],self.frame['revision'],changed)
        path = self.root/'metadata'/self.frame['split']/(self.frame['id']+'.json')
        metadata = json.loads(path.read_text())
        metadata['timestamp_sim_sec'] = 10
        write_json(path,metadata)
        write_review_page(self.root)
        frame = review_index(self.root)['frames'][0]
        self.assertIn('stale_review_edits',frame['flags'])
        self.assertEqual(frame['labels'],frame['original_labels'])
        with self.assertRaisesRegex(ValueError,'stale edits'):
            self.store.export()
        self.store.restore(frame['id'],frame['revision'])
        self.assertNotIn('stale_review_edits',review_index(self.root)['frames'][0]['flags'])

    def test_export_blocks_incomplete_capture_and_empty_split(self):
        write_json(self.root/'validation.json',{'errors':['capture failed'],'frames':3})
        with self.assertRaisesRegex(ValueError,'validation'):
            self.store.export()
        write_json(self.root/'validation.json',{'errors':[],'frames':3})
        self.store.save(self.frame['id'],self.frame['revision'],self.frame['labels'],True)
        with self.assertRaisesRegex(ValueError,'every split'):
            self.store.export()

    def test_existing_export_is_not_reused_if_training_labels_were_changed(self):
        report = self.store.export()
        label = next(Path(report['data_yaml']).parent.glob('labels/*/*.txt'))
        label.write_text('')
        with self.assertRaisesRegex(ValueError,'export was modified'):
            self.store.export()

    def test_prediction_disagreements_include_zero_detections(self):
        path = self.root/'predictions'/self.frame['split']/(self.frame['id']+'.txt')
        path.parent.mkdir(parents=True)
        path.write_text('')
        frame = review_index(self.root)['frames'][0]
        self.assertEqual(frame['prediction_errors']['missed'],1)
        self.assertIn('prediction_disagreement',frame['flags'])

    def test_prediction_wrong_class_is_reported_without_becoming_a_ground_truth_edit(self):
        path = self.root/'predictions'/self.frame['split']/(self.frame['id']+'.txt')
        write_label_file(path,[dict(self.frame['labels'][0],class_id=0)],80,60)
        frame = review_index(self.root)['frames'][0]
        self.assertEqual(frame['prediction_errors'],{'wrong_class':1,'missed':0,'extra':0})
        self.assertEqual(frame['labels'],self.frame['original_labels'])

    def test_http_editor_requires_token_and_does_not_serve_outside_images(self):
        server = make_server(self.root,0)
        worker = threading.Thread(target=server.serve_forever,daemon=True)
        worker.start()
        base = f'http://127.0.0.1:{server.server_port}'
        try:
            with urlopen(base+'/api/index') as response:
                data = json.load(response)
            payload = json.dumps({'id':self.frame['id'],'revision':self.frame['revision'],
                                  'labels':self.frame['labels'],'excluded':True}).encode()
            with self.assertRaises(HTTPError) as error:
                urlopen(Request(base+'/api/save',data=payload,headers={'Content-Type':'application/json'}))
            self.assertEqual(error.exception.code,403)
            request = Request(base+'/api/save',data=payload,
                              headers={'Content-Type':'application/json','X-Review-Token':data['token']})
            with urlopen(request) as response:
                self.assertTrue(json.load(response)['excluded'])
            with self.assertRaises(HTTPError):
                urlopen(base+'/file/metadata/train/'+self.frame['id']+'.json')
            with self.assertRaises(HTTPError):
                urlopen(base+'/file/images/%2e%2e/%2e%2e/outside.txt')
        finally:
            server.shutdown();server.server_close();worker.join()


if __name__ == '__main__':
    unittest.main()
