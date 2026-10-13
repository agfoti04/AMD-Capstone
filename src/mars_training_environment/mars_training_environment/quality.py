"""Offline acceptance checks shared by capture and dataset validation."""
from .framing import target_framing


def frame_checks(view, labels, quality, intrinsics, capture):
    framing = None
    errors = []
    if view.get('target_name'):
        framing = target_framing(labels, view['target_name'], intrinsics, capture, view.get('difficult', False))
        errors.extend(framing['reasons'])
    if capture.get('require_visibility_check'):
        visibility = quality.get('visibility_check', {})
        if not visibility.get('checked'):
            errors.append('visibility_check_missing_sensor_depth')
        if {c['name'] for c in visibility.get('objects', [])} != {l['name'] for l in labels}:
            errors.append('per_object_visibility_checks_incomplete')
        for check in visibility.get('objects', []):
            if not check['passed']:
                errors.append(f'{check["name"]}: '+', '.join(check['reasons']))
    if quality.get('rock_depth_median_error_m', 0) > .08:
        errors.append('aggregate_mesh_depth_mismatch')
    return {'passed':not errors, 'errors':errors, 'target_framing':framing}
