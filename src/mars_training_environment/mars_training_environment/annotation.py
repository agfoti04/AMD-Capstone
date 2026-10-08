"""CPU instance rasterization of the exact visual meshes; no detector input.

Perspective-correct Z buffering resolves rock/rock and rock/terrain occlusion.
The RGB sensor remains Gazebo's existing camera. This code creates only labels.
"""
import json
from pathlib import Path

import numpy as np

from rover_training.geometry import camera_rigid_from_base, class_id_for
from .candidates import drape_mesh
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


SMALL_TRIANGLE_PX = 8


def _rasterize_small(tri, output, fx, fy, cx, cy):
    """Batch path for unclipped triangles whose pixel bounds fit SMALL_TRIANGLE_PX.

    Same pixel-center barycentric test and perspective-correct depth as the
    per-triangle loop, evaluated for all triangles at once.
    """
    h,w = output.shape
    px = fx*tri[:,:,0]/tri[:,:,2]+cx
    py = fy*tri[:,:,1]/tri[:,:,2]+cy
    x0 = np.maximum(0,np.floor(px.min(axis=1)).astype(int))
    y0 = np.maximum(0,np.floor(py.min(axis=1)).astype(int))
    x1 = np.minimum(w-1,np.ceil(px.max(axis=1)).astype(int))
    y1 = np.minimum(h-1,np.ceil(py.max(axis=1)).astype(int))
    denom = (py[:,1]-py[:,2])*(px[:,0]-px[:,2])+(px[:,2]-px[:,1])*(py[:,0]-py[:,2])
    keep = (x0<=x1)&(y0<=y1)&(np.abs(denom)>=1e-10)
    tri,px,py,x0,y0,x1,y1,denom = (a[keep] for a in (tri,px,py,x0,y0,x1,y1,denom))
    offsets = np.arange(SMALL_TRIANGLE_PX+1)
    xs = (x0[:,None,None]+offsets[None,None,:]).astype(float)
    ys = (y0[:,None,None]+offsets[None,:,None]).astype(float)
    xs,ys = np.broadcast_arrays(xs,ys)
    in_box = (xs<=x1[:,None,None])&(ys<=y1[:,None,None])
    xx,yy = xs+.5,ys+.5
    p = lambda arr,k: arr[:,k,None,None]
    a = ((p(py,1)-p(py,2))*(xx-p(px,2))+(p(px,2)-p(px,1))*(yy-p(py,2)))/denom[:,None,None]
    b = ((p(py,2)-p(py,0))*(xx-p(px,2))+(p(px,0)-p(px,2))*(yy-p(py,2)))/denom[:,None,None]
    c = 1-a-b
    z = tri[:,:,2]
    inv_z = a/z[:,0,None,None]+b/z[:,1,None,None]+c/z[:,2,None,None]
    hit = in_box&(a>=-1e-7)&(b>=-1e-7)&(c>=-1e-7)&(inv_z>0)
    depth = (1/np.maximum(inv_z[hit],1e-12)).astype(np.float32)
    flat = ys[hit].astype(int)*w+xs[hit].astype(int)
    np.minimum.at(output.reshape(-1),flat,depth)


def rasterize(vertices, faces, intrinsics, near, far, batch_small=True):
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
    if batch_small and len(tri):
        unclipped = (tri[:,:,2].min(axis=1)>=near)&(tri[:,:,2].max(axis=1)<=far)
        span = np.zeros(len(tri))
        z = np.where(unclipped[:,None],tri[:,:,2],1.0)
        for component,focal in ((0,fx),(1,fy)):
            screen = focal*tri[:,:,component]/z
            span = np.maximum(span,screen.max(axis=1)-screen.min(axis=1))
        small = unclipped&(span<SMALL_TRIANGLE_PX-1)
        batch = tri[small]
        for start in range(0,len(batch),16384):
            _rasterize_small(batch[start:start+16384],output,fx,fy,cx,cy)
        tri = tri[~small]
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
        self.regions = list(scene.get('regions', []))
        self.field = Terrain(t['heights'], t['size_m'])

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
        if self.regions:
            region_instances = np.zeros(zbuffer.shape, dtype=np.int32)
            region_projected = []
            for ident, region in enumerate(self.regions, 1):
                rv, rf = drape_mesh(self.field, region['polygon_xy'])
                depth = rasterize(rv @ r.T + t, rf, intrinsics, near, far)
                region_projected.append(int(np.isfinite(depth).sum()))
                visible = np.isfinite(depth) & (depth < zbuffer - 1e-4)
                region_instances[visible] = ident
            region_labels, region_records = labels_from_instances(
                region_instances, region_projected, self.regions, self.scene['config']['labels'])
            labels.extend(region_labels)
            objects.extend(region_records)
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


def iou(a, b):
    w = min(a[2],b[2])-max(a[0],b[0])
    h = min(a[3],b[3])-max(a[1],b[1])
    if w <= 0 or h <= 0:
        return 0.
    inter = w*h
    return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)


def compare_boxes(labels, objects, gazebo_boxes, policy):
    """Match our labels to Gazebo's bounding-box camera, one-to-one by IoU.

    A Gazebo box that only matches a rock our policy rejected (too hidden or too
    small) is expected, as is one already below the policy's minimum box size:
    IoU means nothing for a one-pixel sliver. Anything else unmatched on either
    side is a disagreement.
    """
    # Ground patches are painted into the terrain, so Gazebo's box camera has no mesh for them.
    labels = [label for label in labels if label.get('source', 'mesh') != 'ground_region']
    objects = [obj for obj in objects if obj.get('source', 'mesh') != 'ground_region']
    min_iou = policy['bbox_min_iou']
    ours = [(l['xmin'],l['ymin'],l['xmax'],l['ymax']) for l in labels]
    pairs = sorted(((iou(a,b),i,j) for i,a in enumerate(ours) for j,b in enumerate(gazebo_boxes)), reverse=True)
    used_ours,used_theirs,matched = set(),set(),[]
    for score,i,j in pairs:
        if score < min_iou:
            break
        if i in used_ours or j in used_theirs:
            continue
        used_ours.add(i)
        used_theirs.add(j)
        matched.append(score)
    rejected = [o['bbox_xyxy'] for o in objects if o['rejected_reason'] and o['bbox_xyxy']]

    def below_policy(box):
        w, h = box[2]-box[0], box[3]-box[1]
        return min(w, h) < policy['min_box_side_px'] or w*h < policy['min_visible_pixels']

    explained,tiny,unexplained = 0,0,[]
    for j,box in enumerate(gazebo_boxes):
        if j in used_theirs:
            continue
        if any(iou(box,r) >= min_iou for r in rejected):
            explained += 1
            continue
        if below_policy(box):
            tiny += 1
            continue
        scored = [(iou(box,o['bbox_xyxy']),o['name'],o['rejected_reason']) for o in objects if o['bbox_xyxy']]
        best = max(scored, key=lambda s: s[0], default=(0.0,None,None))
        unexplained.append({'xyxy':[float(v) for v in box],'nearest_rock':best[1],'nearest_iou':float(best[0]),
                            'nearest_rejected_reason':best[2]})
    missing = [labels[i]['name'] for i in range(len(ours)) if i not in used_ours]
    return {'gazebo_boxes':len(gazebo_boxes),'matched':len(matched),
            'median_iou':float(np.median(matched)) if matched else None,
            'min_matched_iou':float(min(matched)) if matched else None,
            'labels_without_gazebo_match':missing,'gazebo_boxes_for_rejected_rocks':explained,
            'gazebo_boxes_below_policy':tiny,'gazebo_boxes_unexplained':len(unexplained),'unexplained_boxes':unexplained,
            'agrees':not missing and not unexplained}


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
        class_name = obj.get('class_name', 'rock')
        class_id = class_id_for(class_name)
        if class_id is None:
            raise ValueError(f'Unknown label class {class_name}')
        record = {'name':obj['name'],'asset_id':obj['asset_id'],'class_name':class_name,
                  'source':obj.get('source', 'mesh'),'projected_pixels':total,'visible_pixels':count,
                  'visible_fraction':fraction,'rejected_reason':reason,'bbox_xyxy':bounds,
                  'truncated': bool(bounds and (bounds[0]==0 or bounds[1]==0 or bounds[2]==w or bounds[3]==h))}
        if obj.get('synthetic_analogue'):
            record['synthetic_analogue'] = True
        if obj.get('exaggerated_analogue'):
            record['exaggerated_analogue'] = True
        records.append(record)
        if reason is None:
            labels.append({'class_id':class_id,'xmin':bounds[0],'ymin':bounds[1],
                           'xmax':bounds[2],'ymax':bounds[3],**record})
    return labels,records
