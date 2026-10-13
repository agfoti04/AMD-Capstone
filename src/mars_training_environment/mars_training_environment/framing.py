"""Check intended capture targets in image coordinates, without ROS."""
import numpy as np

from rover_training.geometry import camera_rigid_from_base, compose, rpy_to_matrix
from .common import quaternion


def configured_extrinsics(camera):
    # Match camera_joint and camera_optical_joint in the rover xacro.
    return compose(rpy_to_matrix(0, camera['tilt_rad'], 0), camera['mount_xyz'],
                   rpy_to_matrix(-1.5708, 0, -1.5708), (0, 0, 0))


def target_framing(labels, target_name, intrinsics, policy, difficult=False, size_guard_px=0.):
    """Framing gates apply to the primary target, never to incidental rocks."""
    target = next((label for label in labels if label['name'] == target_name), None)
    result = {'target_name': target_name, 'difficult': difficult, 'passed': False, 'reasons': []}
    if target is None:
        result['reasons'].append('target_not_visible')
        return result
    width, height = intrinsics['width'], intrinsics['height']
    scale = policy['training_imgsz'] / max(width, height)
    side = min(target['xmax']-target['xmin'], target['ymax']-target['ymin']) * scale
    result.update(bbox_xyxy=[target[k] for k in ('xmin', 'ymin', 'xmax', 'ymax')],
                  side_at_training_px=float(side), visible_fraction=target['visible_fraction'])
    minimum = policy['target_min_side_px'] * (.5 if difficult else 1) + size_guard_px
    result['minimum_side_at_training_px'] = float(minimum)
    if side < minimum:
        result['reasons'].append('target_too_small_at_training_resolution')
    fraction = .3 if difficult else policy['target_min_visible_fraction']
    if target['visible_fraction'] < fraction:
        result['reasons'].append('target_too_occluded')
    margin = 0 if difficult else policy['target_margin_fraction']
    if (target['xmin'] < margin*width or target['ymin'] < margin*height
            or target['xmax'] > (1-margin)*width or target['ymax'] > (1-margin)*height):
        result['reasons'].append('target_outside_frame_margin')
    if not difficult and target['truncated']:
        result['reasons'].append('target_clipped')
    result['passed'] = not result['reasons']
    return result


def planning_size_guard(intrinsics, policy):
    """Allow for both coarse raster edges plus a small settling margin.

    At 384 pixels, each raster pixel represents 2.5 pixels at training size
    960. A five-pixel box can otherwise pass 12 pixels while its full-size
    counterpart measures only 11. Capture still uses the original threshold.
    """
    return max(2., 2 * policy['training_imgsz'] / max(intrinsics['width'], intrinsics['height']))


def projected_target_fits(vertices, view, camera, intrinsics, policy, difficult=False):
    """Cheap bounds rejection before rasterizing an entire proposed viewpoint."""
    r, t = camera_rigid_from_base(view['xyz'], quaternion(view['rpy']), configured_extrinsics(camera))
    points = np.asarray(vertices) @ np.asarray(r).T + np.asarray(t)
    z = points[:, 2]
    if z.min() <= camera['near_m'] or np.linalg.norm(points,axis=1).max() >= camera['far_m']:
        return False
    x = intrinsics['fx'] * points[:, 0] / z + intrinsics['cx']
    y = intrinsics['fy'] * points[:, 1] / z + intrinsics['cy']
    # Difficult views may touch a boundary; require some on-screen extent.
    margin = 0 if difficult else policy['target_margin_fraction']
    w, h = intrinsics['width'], intrinsics['height']
    if difficult:
        if x.max() <= 0 or x.min() >= w or y.max() <= 0 or y.min() >= h:
            return False
    elif x.min() < margin*w or x.max() > (1-margin)*w or y.min() < margin*h or y.max() > (1-margin)*h:
        return False
    side = min(min(x.max(), w)-max(x.min(), 0), min(y.max(), h)-max(y.min(), 0))
    return side * policy['training_imgsz']/max(w, h) >= policy['target_min_side_px'] * (.5 if difficult else 1)
