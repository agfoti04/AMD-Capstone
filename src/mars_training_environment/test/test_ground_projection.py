import unittest

import numpy as np

from mars_training_environment.annotation import ground_region_instances, instance_depth_support, labels_from_instances, rasterize
from mars_training_environment.common import read_config
from mars_training_environment.candidates import world_to_pixels


class GroundProjectionTests(unittest.TestCase):
    def setUp(self):
        self.k = {'width':80,'height':60,'fx':50.,'fy':50.,'cx':40.,'cy':30.}
        self.depth = np.full((60,80),2.,np.float32)
        self.occluded = np.zeros(self.depth.shape,bool)
        self.region = {'name':'patch','asset_id':'silica_soil','class_name':'silica_soil',
                       'source':'ground_region','mask_id':1,
                       'polygon_xy':[[-.4,-.4],[.4,-.4],[.4,.4],[-.4,.4]]}

    def project(self,regions=None,paint_mask=None,r=None,t=None):
        return ground_region_instances(self.depth,self.occluded,regions or [self.region],
                                       np.eye(3) if r is None else r,
                                       np.zeros(3) if t is None else t,self.k,20.,paint_mask)

    def test_labels_follow_actual_surface_and_keep_ground_class(self):
        instances,projected = self.project()
        labels,_ = labels_from_instances(instances,projected,[self.region],read_config()['labels'])
        self.assertEqual(projected,[400])
        self.assertEqual([labels[0][k] for k in ('xmin','ymin','xmax','ymax')],[30,20,50,40])
        self.assertEqual(labels[0]['class_id'],1)
        self.assertEqual(labels[0]['visible_fraction'],1.)
        checks = instance_depth_support(instances,self.depth,self.depth,[self.region],labels,
                                        read_config()['labels'],.3,8.)
        self.assertTrue(checks[0]['passed'])

    def test_sky_pixels_never_receive_ground_labels(self):
        self.depth[:30] = np.inf
        instances,projected = self.project()
        self.assertFalse(instances[:30].any())
        self.assertEqual(projected,[200])
        self.depth[:] = np.inf
        instances,projected = self.project()
        self.assertFalse(instances.any())
        self.assertEqual(projected,[0])

    def test_region_behind_foreground_surface_cannot_project_onto_it(self):
        # These camera rays hit foreground ground at world x/y near zero,
        # while the distant patch is centered at x=4. Its lifted proxy could
        # intersect a screen ray, but the real surface hit is not in the patch.
        distant = dict(self.region,polygon_xy=[[3.6,-.4],[4.4,-.4],[4.4,.4],[3.6,.4]])
        instances,projected = self.project([distant])
        self.assertFalse(instances.any())
        self.assertEqual(projected,[0])

    def test_mesh_occlusion_reduces_visibility_without_changing_projected_area(self):
        self.occluded[:,40:] = True
        instances,projected = self.project()
        self.assertEqual(projected,[400])
        self.assertEqual(int((instances==1).sum()),200)
        self.assertFalse(instances[self.occluded].any())
        labels,_ = labels_from_instances(instances,projected,[self.region],read_config()['labels'])
        self.assertEqual(labels[0]['visible_fraction'],.5)

    def test_painted_mask_clips_geometry_to_the_correct_category(self):
        mask = np.zeros((101,101),np.uint8)
        mask[:51] = 1
        instances,projected = self.project(paint_mask=mask)
        yy,xx = np.nonzero(instances)
        world_xy = np.column_stack([(xx+.5-40)/50*2,(yy+.5-30)/50*2])
        col,row = world_to_pixels(world_xy,20,101)
        self.assertTrue((mask[np.rint(row).astype(int),np.rint(col).astype(int)]==1).all())
        self.assertGreater(projected[0],0)
        self.assertLess(projected[0],400)
        wrong = dict(self.region,mask_id=2)
        instances,projected = self.project([wrong],paint_mask=mask)
        self.assertFalse(instances.any())
        self.assertEqual(projected,[0])

    def test_world_transform_and_same_class_regions_keep_separate_instances(self):
        regions = [dict(self.region,name='north',polygon_xy=[[2.6,4],[3.4,4],[3.4,4.4],[2.6,4.4]]),
                   dict(self.region,name='south',polygon_xy=[[2.6,3.6],[3.4,3.6],[3.4,4],[2.6,4]])]
        r = np.diag([-1.,-1.,1.])
        t = [3.,4.,0.]
        instances,projected = self.project(regions,r=r,t=t)
        self.assertEqual(set(np.unique(instances)),{0,1,2})
        self.assertEqual(projected,[200,200])

    def test_true_sensor_mismatch_and_missing_depth_remain_failures(self):
        instances,projected = self.project()
        labels,_ = labels_from_instances(instances,projected,[self.region],read_config()['labels'])
        for measured in (None,self.depth+1):
            checks = instance_depth_support(instances,self.depth,measured,[self.region],labels,
                                            read_config()['labels'],.3,8.)
            self.assertFalse(checks[0]['passed'])


class SensorRangeTests(unittest.TestCase):
    def setUp(self):
        self.k = {'width':80,'height':60,'fx':50.,'fy':50.,'cx':40.,'cy':30.}
        self.faces = [[0,1,2],[0,2,3]]

    def test_off_axis_surface_with_valid_z_but_excessive_range_is_not_visible(self):
        vertices = [[3,-.4,7.8],[4,-.4,7.8],[4,.4,7.8],[3,.4,7.8]]
        self.assertFalse(np.isfinite(rasterize(vertices,self.faces,self.k,.3,8)).any())

    def test_center_surface_within_range_keeps_optical_axis_depth(self):
        vertices = [[-.4,-.4,7.8],[.4,-.4,7.8],[.4,.4,7.8],[-.4,.4,7.8]]
        depth = rasterize(vertices,self.faces,self.k,.3,8)
        self.assertTrue(np.isfinite(depth).any())
        self.assertTrue(np.allclose(depth[np.isfinite(depth)],7.8))

    def test_crossing_range_boundary_is_clipped_per_pixel(self):
        vertices = [[0,-1,7.5],[4,-1,7.5],[4,1,7.5],[0,1,7.5]]
        depth = rasterize(vertices,self.faces,self.k,.3,8)
        yy,xx = np.nonzero(np.isfinite(depth))
        distance = depth[yy,xx]*np.sqrt(1+((xx+.5-40)/50)**2+((yy+.5-30)/50)**2)
        self.assertTrue((distance<=8-1e-6).all())
        self.assertTrue(np.isfinite(depth[30,40]))
        self.assertTrue(np.isinf(depth[30,65]))

    def test_near_plane_uses_forward_depth_not_total_distance(self):
        vertices = [[.12,-.03,.28],[.18,-.03,.28],[.18,.03,.28],[.12,.03,.28]]
        self.assertTrue((np.linalg.norm(vertices,axis=1)>.3).all())
        self.assertFalse(np.isfinite(rasterize(vertices,self.faces,self.k,.3,8)).any())


if __name__ == '__main__':
    unittest.main()
