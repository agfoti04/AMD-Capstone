"""CPU instance rasterization of the exact visual meshes; no detector input.

Perspective-correct Z buffering resolves rock/rock and rock/terrain occlusion.
The RGB sensor remains Gazebo's existing camera. This code creates only labels.
"""
import json
from pathlib import Path

import numpy as np

from rover_training.geometry import camera_rigid_from_base
from .common import rotation
from .terrain import Terrain


def clip_polygon(poly, z, keep_above):
    out = []
    for start, end in zip(poly, np.roll(poly, -1, axis=0)):
        inside_a = start[2] >= z if keep_above else start[2] <= z
        inside_b = end[2] >= z if keep_above else end[2] <= z
        if inside_a:
            out.append(start)
        if inside_a != inside_b:
            out.append(start+(end-start)*((z-start[2])/(end[2]-start[2])))
    return np.asarray(out)


def rasterize(vertices, faces, intrinsics, near, far):
    """Return optical-axis depth at pixel centers, infinity for empty pixels."""
    h,w = int(intrinsics['height']),int(intrinsics['width'])
    output = np.full((h,w), np.inf, dtype=np.float32)
    vertices = np.asarray(vertices)
    tri = vertices[np.asarray(faces)]
    tri = tri[(tri[:,:,2].max(axis=1)>=near) & (tri[:,:,2].min(axis=1)<=far)]
    if not len(tri):
        return output
    # Reject triangles entirely outside each side of the camera frustum in 3D.
    fx,fy,cx,cy = (float(intrinsics[k]) for k in ('fx','fy','cx','cy'))
    for component, focal, center, extent in ((0,fx,cx,w),(1,fy,cy,h)):
        lower = focal*tri[:,:,component]+center*tri[:,:,2]
        upper = focal*tri[:,:,component]+(center-extent)*tri[:,:,2]
        tri = tri[(lower.max(axis=1)>=0) & (upper.min(axis=1)<=0)]
    for original in tri:
        poly = original
        if poly[:,2].min() < near:
            poly = clip_polygon(poly, near, True)
        if len(poly) < 3:
            continue
        if poly[:,2].max() > far:
            poly = clip_polygon(poly, far, False)
        for k in range(1,len(poly)-1):
            points = np.array([poly[0],poly[k],poly[k+1]])
            px = fx*points[:,0]/points[:,2]+cx
            py = fy*points[:,1]/points[:,2]+cy
            x0,x1 = max(0,int(np.floor(px.min()))), min(w-1,int(np.ceil(px.max())))
            y0,y1 = max(0,int(np.floor(py.min()))), min(h-1,int(np.ceil(py.max())))
            if x0>x1 or y0>y1:
                continue
            denom = (py[1]-py[2])*(px[0]-px[2])+(px[2]-px[1])*(py[0]-py[2])
            if abs(denom)<1e-10:
                continue
            xx = np.arange(x0,x1+1,dtype=float)[None,:]+.5
            yy = np.arange(y0,y1+1,dtype=float)[:,None]+.5
            a = ((py[1]-py[2])*(xx-px[2])+(px[2]-px[1])*(yy-py[2]))/denom
            b = ((py[2]-py[0])*(xx-px[2])+(px[0]-px[2])*(yy-py[2]))/denom
            c = 1-a-b
            inside = (a>=-1e-7)&(b>=-1e-7)&(c>=-1e-7)
            inv_z = a/points[0,2]+b/points[1,2]+c/points[2,2]
            z = np.where(inside & (inv_z>0), 1/np.maximum(inv_z,1e-12), np.inf)
            region = output[y0:y1+1,x0:x1+1]
            np.minimum(region, z, out=region)
    return output


class Annotator:
    def __init__(self, bundle, scene):
        bundle = Path(bundle)
        assets = json.loads((bundle/'assets/catalog.json').read_text())
        by_id = {a['id']:a for a in assets}
        t = scene['terrain']
        self.terrain = Terrain(t['heights'],t['size_m']).mesh()
        self.meshes = []
        cache = {}
        for obj in scene['objects']:
            asset = by_id[obj['asset_id']]
            if asset['id'] not in cache:
                mesh = json.loads((bundle/asset['geometry']).read_text())
                cache[asset['id']] = np.asarray(mesh['vertices']),np.asarray(mesh['faces'],dtype=int)
            v,f = cache[asset['id']]
            transformed = v @ rotation(obj['rpy']).T * obj['scale']+np.asarray(obj['xyz'])
            self.meshes.append((obj,transformed,f))
        self.scene = scene

    def annotate(self, base_xyz, base_quat, extrinsics, intrinsics, observed_depth=None):
        cam_r,cam_t = camera_rigid_from_base(base_xyz,base_quat,extrinsics)
        r,t = np.asarray(cam_r),np.asarray(cam_t)
        camera = self.scene['config']['camera']
        near,far = camera['near_m'],camera['far_m']
        v,f = self.terrain
        zbuffer = rasterize(v @ r.T+t, f, intrinsics, near, far)
        instances = np.zeros(zbuffer.shape,dtype=np.int32)
        projected = []
        for ident,(obj,v,f) in enumerate(self.meshes,1):
            depth = rasterize(v @ r.T+t,f,intrinsics,near,far)
            count = int(np.isfinite(depth).sum())
            closer = depth < zbuffer - 1e-6
            instances[closer] = ident
            zbuffer[closer] = depth[closer]
            projected.append(count)
        labels,objects = labels_from_instances(instances,projected,[m[0] for m in self.meshes],
                                                self.scene['config']['labels'])
        quality = {}
        if observed_depth is not None:
            observed = np.asarray(observed_depth).reshape(zbuffer.shape)
            valid = np.isfinite(observed)&(observed>near)&(observed<far)&np.isfinite(zbuffer)
            for name,mask in [('all',valid),('rock',valid&(instances>0))]:
                errors = np.abs(observed[mask]-zbuffer[mask])
                if len(errors):
                    quality[name+'_depth_median_error_m'] = float(np.median(errors))
                    quality[name+'_depth_p95_error_m'] = float(np.percentile(errors,95))
                    quality[name+'_compared_pixels'] = len(errors)
        return labels,objects,quality,instances


def labels_from_instances(instances, projected, objects, policy):
    """Visible boxes; frame clipping occurs in rasterization, before thresholds."""
    labels,records = [],[]
    h,w = instances.shape
    for ident,(obj,total) in enumerate(zip(objects,projected),1):
        yy,xx = np.nonzero(instances==ident)
        count = len(xx)
        fraction = count/max(total,1)
        reason = None
        if total == 0:
            reason = 'outside_or_subpixel'
        elif count == 0:
            reason = 'fully_occluded'
        elif count < policy['min_visible_pixels']:
            reason = 'too_few_visible_pixels'
        elif fraction < policy['min_visible_fraction']:
            reason = 'occluded_below_fraction'
        bounds = None
        if count:
            bounds = [int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]
            if min(bounds[2]-bounds[0],bounds[3]-bounds[1]) < policy['min_box_side_px']:
                reason = reason or 'too_small'
        record = {'name':obj['name'],'asset_id':obj['asset_id'],'projected_pixels':total,'visible_pixels':count,
                  'visible_fraction':fraction,'rejected_reason':reason,'bbox_xyxy':bounds,
                  'truncated': bool(bounds and (bounds[0]==0 or bounds[1]==0 or bounds[2]==w or bounds[3]==h))}
        records.append(record)
        if reason is None:
            labels.append({'class_id':0,'class_name':'rock','xmin':bounds[0],'ymin':bounds[1],
                           'xmax':bounds[2],'ymax':bounds[3],**record})
    return labels,records
