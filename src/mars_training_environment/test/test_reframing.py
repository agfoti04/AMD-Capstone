import copy
import math
import unittest

import numpy as np

from mars_training_environment.common import read_config
from mars_training_environment.reframing import closer_target_view, MAX_REFRAMES, SIZE_FAILURE


class ReframingTests(unittest.TestCase):
    def setUp(self):
        self.scene = {'config':read_config(), 'terrain':{'size_m':20, 'base_heights':np.zeros((17,17)).tolist()},
                      'objects':[{'name':'target','xyz':[0,0,0],'radius_m':.03}], 'regions':[]}
        self.view = {'id':'view_004','target_name':'target','difficult':True,
                     'xyz':[-2.4,0,.09],'rpy':[0,0,.35]}
        self.checks = {'errors':[SIZE_FAILURE], 'target_framing':{'side_at_training_px':11.}}

    def test_size_failure_moves_closer_with_audited_pose_and_preserved_plan(self):
        original = copy.deepcopy(self.view)
        result = closer_target_view(self.scene,self.view,self.checks,[-2.4,0,.08])
        self.assertIsNotNone(result)
        self.assertGreater(result['xyz'][0],self.view['xyz'][0])
        self.assertEqual(result['rpy'][2],self.view['rpy'][2])
        self.assertEqual(result['target_name'],'target')
        self.assertTrue(result['difficult'])
        self.assertEqual(result['framing_adjustments'][0]['from_measured_xyz'],[-2.4,0,.08])
        self.assertEqual(self.view,original)
        camera_xy = np.array(result['xyz'][:2])+.2*np.array([math.cos(.35),math.sin(.35)])
        self.assertGreaterEqual(np.linalg.norm(camera_xy),1.8)

    def test_missing_target_or_any_visibility_failure_cannot_trigger_a_move(self):
        for errors in (['target_not_visible'], [SIZE_FAILURE,'rock: insufficient_sensor_depth_support'],
                       ['target_too_occluded'], []):
            self.assertIsNone(closer_target_view(self.scene,self.view,dict(self.checks,errors=errors),self.view['xyz']))

    def test_retries_are_bounded_and_range_floor_is_respected(self):
        exhausted = dict(self.view,framing_adjustments=[{}]*MAX_REFRAMES)
        self.assertIsNone(closer_target_view(self.scene,exhausted,self.checks,self.view['xyz']))
        self.assertIsNone(closer_target_view(self.scene,self.view,self.checks,[-2,0,.08]))

    def test_blocked_endpoint_is_rejected(self):
        self.scene['objects'].append({'name':'obstacle','xyz':[-2.05,0,0],'radius_m':.15})
        self.assertIsNone(closer_target_view(self.scene,self.view,self.checks,self.view['xyz']))

    def test_steep_terrain_is_rejected(self):
        row = np.linspace(-10,10,17)*.5
        self.scene['terrain']['base_heights'] = np.tile(row,(17,1)).tolist()
        self.assertIsNone(closer_target_view(self.scene,self.view,self.checks,self.view['xyz']))

    def test_invalid_size_and_unknown_target_are_rejected(self):
        for side in (0,float('nan'),float('inf')):
            checks = dict(self.checks,target_framing={'side_at_training_px':side})
            self.assertIsNone(closer_target_view(self.scene,self.view,checks,self.view['xyz']))
        self.assertIsNone(closer_target_view(self.scene,dict(self.view,target_name='missing'),self.checks,self.view['xyz']))


if __name__ == '__main__':
    unittest.main()
