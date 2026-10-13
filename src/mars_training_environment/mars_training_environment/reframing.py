"""Bounded camera-pose corrections for undersized targets in existing plans."""
import math

import numpy as np

from .terrain import Terrain


MAX_REFRAMES = 3
SIZE_FAILURE = 'target_too_small_at_training_resolution'


def closer_target_view(scene, view, checks, measured_xyz):
    """Propose a closer teleport pose only for an otherwise valid frame.

    The captured image must pass every normal check after this move. Keep the
    original target, azimuth, difficult-view offset and configured distance
    range. Reject endpoints that violate the planner's collision/slope limits.
    No view or scene manifest is modified in place.
    """
    if checks.get('errors') != [SIZE_FAILURE]:
        return None
    history = list(view.get('framing_adjustments', []))
    if len(history) >= MAX_REFRAMES:
        return None
    target = next((obj for obj in scene['objects'] + scene.get('regions', [])
                   if obj['name'] == view.get('target_name')), None)
    if target is None:
        return None
    framing = checks.get('target_framing') or {}
    side = framing.get('side_at_training_px', 0)
    if not math.isfinite(side) or side <= 0:
        return None
    config = scene['config']
    cap, camera = config['capture'], config['camera']
    minimum = cap['target_min_side_px'] * (.5 if view.get('difficult', False) else 1)
    target_xy = np.asarray(target['xyz'][:2], dtype=float)
    xy = np.asarray(measured_xyz[:2], dtype=float)
    offset = xy-target_xy
    distance = float(np.linalg.norm(offset))
    lens = float(np.linalg.norm(camera['mount_xyz'][:2]))
    low, high = cap['distance_m']
    # Triangle inequality keeps the lens at least low metres from the target,
    # including the difficult-view yaw offset and camera's forward mount.
    lower = max(low, target['radius_m']+camera['near_m']+.25) + lens
    if not math.isfinite(distance) or distance <= lower+.03:
        return None
    # Aim above the acceptance floor so rounding/settling do not cause another
    # marginal result; move at most 25% of the current distance per correction.
    factor = float(np.clip(side/(minimum+2), .75, .90))
    desired = max(lower, distance*factor)
    if distance-desired < .03:
        return None
    terrain = scene['terrain']
    field = Terrain(terrain['base_heights'], terrain['size_m'])
    yaw = view['rpy'][2]
    cy, sy = math.cos(yaw), math.sin(yaw)
    mount_x, mount_y = camera['mount_xyz'][:2]
    camera_offset = np.array([mount_x*cy-mount_y*sy, mount_x*sy+mount_y*cy])
    # If the preferred endpoint is blocked, inspect closer endpoints within
    # the same configured range. This is a teleport, not a driving route.
    for candidate_distance in np.linspace(desired, lower, 5):
        candidate = target_xy + offset/distance*candidate_distance
        x, y = map(float, candidate)
        if max(abs(x), abs(y)) > field.size/2-1.2:
            continue
        if any(np.linalg.norm(candidate-np.asarray(obj['xyz'][:2])) < .42+obj['radius_m']
               for obj in scene['objects']):
            continue
        camera_xy = candidate+camera_offset
        gap = float(np.linalg.norm(camera_xy-target_xy))
        if not low <= gap <= high:
            continue
        if any(np.linalg.norm(camera_xy-np.asarray(obj['xyz'][:2])) < obj['radius_m']+camera['near_m']+.08
               for obj in scene['objects']):
            continue
        gx, gy = field.gradient(x, y)
        if math.hypot(gx, gy) > math.tan(math.radians(config['terrain']['max_slope_deg'])):
            continue
        roll = math.atan(-gx*sy+gy*cy)
        pitch = -math.atan(gx*cy+gy*sy)
        new_xyz = [x, y, float(field.height(x, y))+.09]
        adjustment = {'reason':SIZE_FAILURE, 'side_at_training_px':float(side),
                      'minimum_side_at_training_px':float(minimum),
                      'from_measured_xyz':list(map(float, measured_xyz)), 'to_xyz':new_xyz}
        return dict(view, xyz=new_xyz, rpy=[roll, pitch, yaw],
                    framing_adjustments=history+[adjustment])
    return None
