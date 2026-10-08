"""Camera geometry and rock boxes for Gazebo Classic.

Gazebo Classic 11 has no bounding-box camera. Labels are the projected mesh
vertices of each rock (the collision AABB as a fallback), with optional
depth-buffer occlusion. They are not detector predictions.
"""

import math
import re

CLASS_NAMES = (
    "rock",
    "silica_soil",
    "pale_lookalike",
    "sulfur_rock",
    "pale_stone",
    "cheyava_rock",
    "veined_lookalike",
)


def class_id_for(class_name):
    key = str(class_name).strip().lower()
    if key not in CLASS_NAMES:
        return None
    return CLASS_NAMES.index(key)


def rpy_to_matrix(roll, pitch, yaw):
    """URDF fixed-axis rotation: R = Rz(yaw) * Ry(pitch) * Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def mat_vec(matrix, vector):
    return tuple(sum(matrix[row][col] * vector[col] for col in range(3)) for row in range(3))


def mat_mul(left, right):
    return tuple(
        tuple(sum(left[row][k] * right[k][col] for k in range(3)) for col in range(3))
        for row in range(3)
    )


def mat_transpose(matrix):
    return tuple(tuple(matrix[row][col] for row in range(3)) for col in range(3))


def compose(parent_r, parent_t, child_r, child_t):
    """Rigid transform that applies the child frame, then the parent frame."""
    return mat_mul(parent_r, child_r), tuple(
        parent_t[i] + mat_vec(parent_r, child_t)[i] for i in range(3)
    )


def invert_rigid(rotation, translation):
    transposed = mat_transpose(rotation)
    return transposed, tuple(-value for value in mat_vec(transposed, translation))


def transform_point(rotation, translation, point):
    rotated = mat_vec(rotation, point)
    return tuple(rotated[i] + translation[i] for i in range(3))


def yaw_quaternion(yaw):
    return (0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5))


def quaternion_to_matrix(x, y, z, w):
    norm = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def pose_to_rigid(xyz, quaternion):
    return quaternion_to_matrix(*quaternion), tuple(xyz)


def parse_camera_extrinsics(xacro_text):
    """base_link -> camera_optical_frame from the rover xacro joint origins."""
    frames = {}
    for name in ("camera_joint", "camera_optical_joint"):
        match = re.search(
            rf'<joint name="{name}"[^>]*>.*?</joint>',
            xacro_text,
            flags=re.DOTALL,
        )
        if not match:
            raise ValueError(f"missing {name} in rover xacro")
        origin = re.search(r"<origin\b([^>]*)/?>", match.group(0))
        attrs = origin.group(1) if origin else ""
        xyz = _xyz_attr(attrs, "xyz", (0.0, 0.0, 0.0))
        rpy = _xyz_attr(attrs, "rpy", (0.0, 0.0, 0.0))
        frames[name] = (rpy_to_matrix(*rpy), xyz)
    return compose(*frames["camera_joint"], *frames["camera_optical_joint"])


# 30 inches. This is the optical-center height above the wheel contact plane,
# not the camera_joint Z offset.
OPTICAL_CENTER_HEIGHT_M = 0.762


def wheel_contact_drop(xacro_text):
    """Meters from the base_link origin down to the wheel contact point.

    The collision spheres are centered on the wheel links. The contact point
    is one radius below that center. The ros2_control joint of the same name
    has no origin, so the drop comes from the wheel macro.
    """
    macro = re.search(r'<xacro:macro name="wheel".*?</xacro:macro>', xacro_text, flags=re.DOTALL)
    if not macro:
        raise ValueError("missing wheel macro")
    body = macro.group(0)
    origin = re.search(r'xyz="([^"]+)"', body)
    if not origin:
        raise ValueError("missing wheel joint origin")
    wheel_z = float(origin.group(1).split()[2])
    radius_match = re.search(r'<sphere radius="([^"]+)"', body)
    if not radius_match:
        raise ValueError("missing wheel collision sphere")
    return -wheel_z + float(radius_match.group(1))


def optical_center_above_ground(xacro_text):
    """Height of the optical-frame origin above the wheel contact plane.

    camera_optical_joint has no translation, so the optical center is the
    camera_joint origin. Adding 0.762 m to that joint would leave the lens
    0.762 m above base_link, which is already above the ground.
    """
    _rotation, translation = parse_camera_extrinsics(xacro_text)
    return translation[2] + wheel_contact_drop(xacro_text)


def nominal_intrinsics(width=1920, height=1080, hfov=1.22):
    """Pinhole values for the rover RGB-D camera.

    hfov 1.22 rad is provisional. It is not a measured OAK-D calibration.
    Vertical focal length follows the image aspect ratio at that horizontal
    field of view. Depth uses the same image size as RGB.
    """
    fx = (width * 0.5) / math.tan(hfov * 0.5)
    vfov = 2.0 * math.atan(math.tan(hfov * 0.5) * (height / float(width)))
    fy = (height * 0.5) / math.tan(vfov * 0.5)
    return {
        "width": int(width),
        "height": int(height),
        "fx": fx,
        "fy": fy,
        "cx": (width - 1) * 0.5,
        "cy": (height - 1) * 0.5,
        "hfov": hfov,
    }


def intrinsics_from_camera_info(msg):
    k = list(msg.k)
    return {
        "width": int(msg.width),
        "height": int(msg.height),
        "fx": float(k[0]),
        "fy": float(k[4]),
        "cx": float(k[2]),
        "cy": float(k[5]),
    }


def box_corners(bounds_min, bounds_max):
    corners = []
    for x in (bounds_min[0], bounds_max[0]):
        for y in (bounds_min[1], bounds_max[1]):
            for z in (bounds_min[2], bounds_max[2]):
                corners.append((x, y, z))
    return corners


def scaled_bounds(bounds_min, bounds_max, scale):
    return (
        tuple(value * scale for value in bounds_min),
        tuple(value * scale for value in bounds_max),
    )


def seat_on_ground(bounds_min, bounds_max, scale, roll, pitch, yaw):
    """Model-origin height that puts the scaled, rotated box on z = 0."""
    rotation = rpy_to_matrix(roll, pitch, yaw)
    low, high = scaled_bounds(bounds_min, bounds_max, scale)
    lowest = min(transform_point(rotation, (0.0, 0.0, 0.0), corner)[2]
                 for corner in box_corners(low, high))
    return -lowest


def project_box(world_r, world_t, bounds_min, bounds_max, cam_r, cam_t, intrinsics,
                near=0.05, vertices=None):
    """Pixel box of a rock. cam_r/cam_t is world -> optical.

    With model-frame mesh vertices the box hugs the rendered silhouette. The
    AABB corners are used when vertices are missing or some cross the near plane.
    """
    camera_vertices = None
    if vertices:
        camera_vertices = [
            transform_point(cam_r, cam_t, transform_point(world_r, world_t, vertex))
            for vertex in vertices
        ]
        if any(point[2] < near for point in camera_vertices):
            camera_vertices = None
    if camera_vertices is not None:
        visible = camera_vertices
    else:
        corners = [transform_point(world_r, world_t, corner)
                   for corner in box_corners(bounds_min, bounds_max)]
        camera_points = [transform_point(cam_r, cam_t, point) for point in corners]
        visible = _visible_edge_points(camera_points, near)
    if len(visible) < 2:
        return None
    pixels = [_project(point, intrinsics) for point in visible]
    xmin = min(pixel[0] for pixel in pixels)
    xmax = max(pixel[0] for pixel in pixels)
    ymin = min(pixel[1] for pixel in pixels)
    ymax = max(pixel[1] for pixel in pixels)
    clipped = _clip_box(xmin, ymin, xmax, ymax, intrinsics["width"], intrinsics["height"])
    if clipped is None:
        return None
    return {
        "xmin": clipped[0],
        "ymin": clipped[1],
        "xmax": clipped[2],
        "ymax": clipped[3],
        "truncated": clipped[4],
    }


def visibility_fraction(box, world_r, world_t, bounds_min, bounds_max,
                        cam_r, cam_t, intrinsics, depth, grid=5, tolerance=0.08,
                        others=()):
    """Fraction of box samples whose depth matches the rock, not a closer occluder.

    depth is optical-axis Z in meters, the encoding published by
    libgazebo_ros_camera (32FC1). None means the check was not run.
    others holds (world_r, world_t, bounds_min, bounds_max) of the other rocks;
    a sample whose depth also fits a rock entered earlier on the ray belongs
    to that rock.
    """
    if depth is None:
        return None
    width = intrinsics["width"]
    height = intrinsics["height"]
    if len(depth) != width * height:
        raise ValueError("depth buffer does not match the camera size")
    cam_from_model_r, cam_from_model_t = _model_pose_in_camera(
        world_r, world_t, cam_r, cam_t)
    inverse_r, inverse_t = invert_rigid(cam_from_model_r, cam_from_model_t)
    occluders = []
    for other_r, other_t, other_min, other_max in others:
        other_cam_r, other_cam_t = _model_pose_in_camera(other_r, other_t, cam_r, cam_t)
        occluders.append((*invert_rigid(other_cam_r, other_cam_t), other_min, other_max))
    samples = _sample_pixels(box, width, height, grid)
    if not samples:
        return 0.0
    visible = 0
    for u, v in samples:
        direction = ((u - intrinsics["cx"]) / intrinsics["fx"],
                     (v - intrinsics["cy"]) / intrinsics["fy"],
                     1.0)
        hit = _ray_aabb(inverse_t, mat_vec(inverse_r, direction), bounds_min, bounds_max)
        measured = depth[int(v) * width + int(u)]
        if hit is None or not _depth_sees_surface(measured, hit[0], hit[1], tolerance):
            continue
        if any(_nearer_rock_owns(measured, hit[0], direction, occluder, tolerance)
               for occluder in occluders):
            continue
        visible += 1
    return visible / float(len(samples))


def _nearer_rock_owns(measured, z_enter, direction, occluder, tolerance):
    inverse_r, inverse_t, low, high = occluder
    hit = _ray_aabb(inverse_t, mat_vec(inverse_r, direction), low, high)
    if hit is None or hit[0] >= z_enter:
        return False
    return _depth_sees_surface(measured, hit[0], hit[1], tolerance)


def annotate_rocks(rocks, camera_rigid, intrinsics, depth=None,
                   min_visible_fraction=0.30, tolerance=0.08, near=0.05):
    """Project simulator rock poses into YOLO boxes. Unknown classes are skipped."""
    cam_r, cam_t = camera_rigid
    solids = [
        (*pose_to_rigid(rock["xyz"], rock["quaternion"]),
         *scaled_bounds(rock["bounds_min"], rock["bounds_max"], rock["scale"]))
        for rock in rocks
    ]
    labels = []
    for index, rock in enumerate(rocks):
        class_id = class_id_for(rock.get("class_name", "rock"))
        if class_id is None:
            continue
        world_r, world_t, low, high = solids[index]
        vertices = rock.get("vertices")
        if vertices:
            scale = rock["scale"]
            vertices = [tuple(value * scale for value in vertex) for vertex in vertices]
        box = project_box(world_r, world_t, low, high, cam_r, cam_t, intrinsics,
                          near=near, vertices=vertices)
        if box is None:
            continue
        fraction = visibility_fraction(
            box, world_r, world_t, low, high, cam_r, cam_t, intrinsics, depth,
            tolerance=tolerance,
            others=solids[:index] + solids[index + 1:],
        )
        if fraction is not None and fraction < min_visible_fraction:
            continue
        labels.append({
            "class_name": "rock",
            "class_id": class_id,
            "asset_id": rock.get("asset_id", ""),
            "model_name": rock.get("name", ""),
            "visible_fraction": fraction,
            "truncated": box["truncated"],
            "xmin": box["xmin"],
            "ymin": box["ymin"],
            "xmax": box["xmax"],
            "ymax": box["ymax"],
        })
    return labels


def camera_rigid_from_base(base_xyz, base_quaternion, base_to_optical):
    base_r, base_t = pose_to_rigid(base_xyz, base_quaternion)
    world_r, world_t = compose(base_r, base_t, *base_to_optical)
    return invert_rigid(world_r, world_t)


def _xyz_attr(attrs, name, default):
    match = re.search(rf'{name}="([^"]+)"', attrs)
    if not match:
        return default
    parts = [float(part) for part in match.group(1).split()]
    if len(parts) != 3:
        raise ValueError(f"{name} must have three values")
    return tuple(parts)


def _project(point, intrinsics):
    return (
        intrinsics["fx"] * point[0] / point[2] + intrinsics["cx"],
        intrinsics["fy"] * point[1] / point[2] + intrinsics["cy"],
    )


def _visible_edge_points(corners, near):
    edges = (
        (0, 1), (0, 2), (0, 4),
        (3, 1), (3, 2), (3, 7),
        (5, 1), (5, 4), (5, 7),
        (6, 2), (6, 4), (6, 7),
    )
    points = []
    for start, end in edges:
        points.extend(_clip_edge(corners[start], corners[end], near))
    return points


def _clip_edge(start, end, near):
    if start[2] >= near and end[2] >= near:
        return (start, end)
    if start[2] < near and end[2] < near:
        return ()
    span = end[2] - start[2]
    if span == 0.0:
        return ()
    blend = (near - start[2]) / span
    mid = tuple(start[i] + blend * (end[i] - start[i]) for i in range(3))
    if start[2] >= near:
        return (start, mid)
    return (mid, end)


def _clip_box(xmin, ymin, xmax, ymax, width, height):
    right = width - 1.0
    bottom = height - 1.0
    clipped = (
        xmin < 0.0 or ymin < 0.0 or xmax > right or ymax > bottom
    )
    xmin = min(max(xmin, 0.0), right)
    xmax = min(max(xmax, 0.0), right)
    ymin = min(max(ymin, 0.0), bottom)
    ymax = min(max(ymax, 0.0), bottom)
    if xmax - xmin < 2.0 or ymax - ymin < 2.0:
        return None
    return xmin, ymin, xmax, ymax, clipped


def _model_pose_in_camera(world_r, world_t, cam_r, cam_t):
    return compose(cam_r, cam_t, world_r, world_t)


def _sample_pixels(box, width, height, grid):
    x0 = int(math.floor(box["xmin"]))
    x1 = int(math.ceil(box["xmax"]))
    y0 = int(math.floor(box["ymin"]))
    y1 = int(math.ceil(box["ymax"]))
    x0 = min(max(x0, 0), width - 1)
    x1 = min(max(x1, 0), width - 1)
    y0 = min(max(y0, 0), height - 1)
    y1 = min(max(y1, 0), height - 1)
    if x1 <= x0 or y1 <= y0:
        return []
    samples = []
    steps = max(int(grid), 1)
    for row in range(steps):
        for col in range(steps):
            u = x0 + (x1 - x0) * (col + 0.5) / steps
            v = y0 + (y1 - y0) * (row + 0.5) / steps
            samples.append((u, v))
    return samples


def _ray_aabb(origin, direction, bounds_min, bounds_max):
    tmin = 0.0
    tmax = 1.0e9
    for axis in range(3):
        if abs(direction[axis]) < 1.0e-9:
            if origin[axis] < bounds_min[axis] or origin[axis] > bounds_max[axis]:
                return None
            continue
        inv = 1.0 / direction[axis]
        t1 = (bounds_min[axis] - origin[axis]) * inv
        t2 = (bounds_max[axis] - origin[axis]) * inv
        if t1 > t2:
            t1, t2 = t2, t1
        tmin = max(tmin, t1)
        tmax = min(tmax, t2)
        if tmin > tmax:
            return None
    if tmax < 0.0:
        return None
    return tmin, tmax


def _depth_sees_surface(measured, z_enter, z_exit, tolerance):
    if measured is None or isinstance(measured, float) and (math.isnan(measured) or math.isinf(measured)):
        return False
    if measured <= 0.0:
        return False
    slack = max(tolerance, 0.02 * z_enter)
    if measured < z_enter - slack:
        return False
    return measured <= z_exit + slack
