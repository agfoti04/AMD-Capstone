import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mars_training_environment.common import read_config, split_for_seed
from mars_training_environment.generate import make_scene
from mars_training_environment.planning import (ViewPlanningError, class_coverage, class_view_schedule,
                                                targets_for_class, validate_class_quotas)
from mars_training_environment.quality import frame_checks
from rover_training.geometry import CLASS_NAMES


QUOTAS = {'rock':4,'pale_soil_patch':3,'yellow_crystal_rock':3}


class ClassQuotaTests(unittest.TestCase):
    def setUp(self):
        self.config = read_config()
        self.config['capture'].update(target_class_views=QUOTAS,class_quota_scene_attempts=3)

    def test_mixed_views_keep_exact_quotas_across_random_schedules(self):
        schedules = []
        for seed in range(8):
            schedule = class_view_schedule(self.config['capture'],np.random.RandomState(seed))
            self.assertEqual(Counter(schedule),{0:4,1:3,2:3})
            schedules.append(schedule)
        self.assertTrue(any(schedule != schedules[0] for schedule in schedules[1:]))

    def test_explicit_empty_and_rock_only_scenes_keep_their_roles(self):
        self.assertEqual(class_view_schedule(self.config['capture'],np.random.RandomState(1),empty=True),[None]*10)
        self.assertEqual(class_view_schedule(self.config['capture'],np.random.RandomState(1),rocks_only=True),[0]*10)
        legacy = dict(self.config['capture'],target_class_views={})
        self.assertEqual(class_view_schedule(legacy,np.random.RandomState(1)),[None]*10)

    def test_crystal_pool_never_falls_back_to_soil_or_rock(self):
        targets = [{'name':'rock'}, {'name':'soil','class_name':'silica_soil'},
                   {'name':'lookalike','class_name':'pale_lookalike'},
                   {'name':'crystal','class_name':'sulfur_rock'}, {'name':'stone','class_name':'pale_stone'}]
        self.assertEqual([target['name'] for target in targets_for_class(targets,2)],['crystal'])
        self.assertEqual([target['name'] for target in targets_for_class(targets,1)],['soil','lookalike'])
        self.assertEqual([target['name'] for target in targets_for_class(targets[:-2],2)],[])

    def test_invalid_quotas_are_rejected(self):
        for quotas in ({'unknown':10}, {'rock':9}, {'rock':True,'pale_soil_patch':9},
                       {'rock':-1,'pale_soil_patch':11}, {'rock':10.0}, None):
            with self.subTest(quotas=quotas), self.assertRaises(ValueError):
                validate_class_quotas(dict(self.config['capture'],target_class_views=quotas))
        for attempts in (0,21,True,1.5):
            with self.subTest(attempts=attempts), self.assertRaises(ValueError):
                validate_class_quotas(dict(self.config['capture'],class_quota_scene_attempts=attempts))

    def test_impossible_candidate_configuration_is_rejected_before_generation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'bad.json'
            for candidates in ({'enabled':False},{'sulfur_count':[0,0]},
                               {'categories':['sulfur_rock','pale_stone']}):
                path.write_text(json.dumps({'capture':{'target_class_views':QUOTAS},'candidates':candidates}))
                with self.subTest(candidates=candidates), self.assertRaises(ValueError):
                    read_config(path)

    def test_retry_changes_layout_seed_but_preserves_scene_seed_and_split(self):
        failure = ViewPlanningError('no crystal pose fits')
        with patch('mars_training_environment.generate._make_scene',side_effect=[failure,{'id':'ok'}]) as make:
            result = make_scene('unused',self.config,101,[])
        self.assertEqual(result,{'id':'ok'})
        first, second = [call.args for call in make.call_args_list]
        self.assertEqual(first[2],second[2])
        self.assertEqual(first[2],101)
        self.assertEqual(split_for_seed(first[2]),split_for_seed(second[2]))
        self.assertNotEqual(first[5],second[5])
        self.assertEqual((first[6],second[6]),(1,2))

    def test_layout_retries_are_bounded_and_io_failures_are_not_swallowed(self):
        with patch('mars_training_environment.generate._make_scene',side_effect=ViewPlanningError('infeasible')) as make:
            with self.assertRaisesRegex(ViewPlanningError,'after 3 layouts'):
                make_scene('unused',self.config,101,[])
            self.assertEqual(make.call_count,3)
        with patch('mars_training_environment.generate._make_scene',side_effect=OSError('disk problem')) as make:
            with self.assertRaises(OSError):
                make_scene('unused',self.config,101,[])
            self.assertEqual(make.call_count,1)

    def scene(self):
        objects = [{'name':'rock','class_name':'rock'},{'name':'soil','class_name':'silica_soil'},
                   {'name':'crystal','class_name':'sulfur_rock'}]
        schedule = [0]*4+[1]*3+[2]*3
        return {'id':'scene_000101','seed':101,'split':'train','empty':False,'class_names':list(CLASS_NAMES),
                'config':self.config,'objects':objects,'regions':[],
                'views':[{'id':f'view_{index:03d}','target_name':objects[cid]['name'],
                          'target_class_id':cid,'requested_target_class_id':cid} for index,cid in enumerate(schedule)]}

    def test_coverage_checks_target_identity_and_per_scene_quota(self):
        scene = self.scene()
        report = class_coverage([scene])
        self.assertEqual(report['errors'],[])
        self.assertEqual(report['primary_target_counts']['train'],QUOTAS)
        self.assertEqual(report['frames'],10)
        corrupted = copy.deepcopy(scene)
        corrupted['views'][-1]['target_name'] = 'soil'
        report = class_coverage([corrupted])
        self.assertTrue(any('identity/class mismatch' in error for error in report['errors']))
        self.assertTrue(any('quota mismatch' in error for error in report['errors']))

    def test_coverage_does_not_accept_cross_class_fallback_with_updated_metadata(self):
        scene = self.scene()
        scene['views'][-1].update(target_name='soil',target_class_id=1)
        report = class_coverage([scene])
        self.assertTrue(any('requested target class' in error for error in report['errors']))
        self.assertTrue(any('quota mismatch' in error for error in report['errors']))

    def test_capture_checks_reject_a_target_of_the_wrong_requested_class(self):
        target = {'name':'target','class_id':0,'xmin':300,'ymin':200,'xmax':600,'ymax':500,
                  'visible_fraction':.8,'truncated':False}
        capture = dict(self.config['capture'],require_visibility_check=False)
        view = {'target_name':'target','requested_target_class_id':2}
        result = frame_checks(view,[target],{}, {'width':1920,'height':1080},capture)
        self.assertIn('target_class_does_not_match_requested_quota',result['errors'])
        self.assertTrue(frame_checks(dict(view,requested_target_class_id=0),[target],{},
                                     {'width':1920,'height':1080},capture)['passed'])

    def test_balanced_profile_retains_expected_split_and_background_counts(self):
        path = Path(__file__).resolve().parents[1]/'config'/'coverage500_balanced.json'
        config = read_config(path)
        self.assertEqual(Counter(split_for_seed(seed) for seed in config['dataset']['seeds']),
                         {'train':30,'val':10,'test':10})
        for key in ('empty_seeds','rocks_only_seeds'):
            self.assertEqual(Counter(split_for_seed(seed) for seed in config['dataset'][key]),
                             {'train':3,'val':1,'test':1})


if __name__ == '__main__':
    unittest.main()
