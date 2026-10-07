"""Seeded scenes. A scene and its views stay in one split.

Held-out rock assets appear only in the test split, and never in the same
image as a training asset.
"""

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

from rover_training.assets import load_assets
from rover_training.geometry import (
    annotate_rocks,
    camera_rigid_from_base,
    nominal_intrinsics,
    parse_camera_extrinsics,
    seat_on_ground,
    yaw_quaternion,
)
from rover_training.world_builder import ROBOT_SPAWN_Z

ROBOT_Z = ROBOT_SPAWN_Z


def default_xacro():
    try:
        from ament_index_python.packages import get_package_share_directory
        shared = Path(get_package_share_directory("four_wheel_sim")) / "urdf" / "robot.urdf.xacro"
        if shared.is_file():
            return shared
    except Exception:
        pass
    return Path(__file__).resolve().parents[2] / "four_wheel_sim" / "urdf" / "robot.urdf.xacro"


def build_plan(seed, assets, xacro_text, random_scenes=2, intrinsics=None):
    intrinsics = intrinsics or nominal_intrinsics()
    base_to_optical = parse_camera_extrinsics(xacro_text)
    rng = random.Random(seed)
    train_assets = [asset for asset in assets if not asset["held_out"]]
    test_assets = [asset for asset in assets if asset["held_out"]]
    if len(train_assets) < 3:
        raise ValueError("need at least three training rock assets")
    if not test_assets:
        raise ValueError("need one held-out rock asset")

    scenes = [
        _three_rock_scene(train_assets, base_to_optical, intrinsics),
        _empty_scene(),
        _clip_scene(train_assets[0], base_to_optical, intrinsics),
        _aligned_scene(train_assets[0], train_assets[1], base_to_optical, intrinsics),
        _holdout_scene(test_assets[0], base_to_optical, intrinsics),
    ]
    for index in range(random_scenes):
        scenes.append(_random_scene(
            rng, train_assets, test_assets, base_to_optical, intrinsics, index, seed))
    for scene in scenes:
        scene["split"] = _split_for(scene, seed)
    return {
        "seed": seed,
        "class_names": ["rock"],
        "image_size": [intrinsics["width"], intrinsics["height"]],
        "hfov": intrinsics.get("hfov", 1.22),
        "held_out_assets": [asset["id"] for asset in test_assets],
        "scenes": scenes,
    }


def rocks_for_annotation(scene, assets):
    by_id = {asset["id"]: asset for asset in assets}
    rocks = []
    for rock in scene["rocks"]:
        asset = by_id[rock["asset_id"]]
        rocks.append({
            "name": rock["name"],
            "asset_id": rock["asset_id"],
            "class_name": "rock",
            "xyz": rock["xyz"],
            "quaternion": yaw_quaternion(rock["rpy"][2]) if _yaw_only(rock["rpy"]) else _rpy_quat(rock["rpy"]),
            "rpy": rock["rpy"],
            "scale": rock["scale"],
            "bounds_min": asset["bounds_min_m"],
            "bounds_max": asset["bounds_max_m"],
        })
    return rocks


def camera_for_view(view, base_to_optical):
    return camera_rigid_from_base(view["xyz"], yaw_quaternion(view["yaw"]), base_to_optical)


def list_scene_ids(plan):
    return [scene["id"] for scene in plan["scenes"]]


def scene_by_id(plan, scene_id):
    for scene in plan["scenes"]:
        if scene["id"] == scene_id:
            return scene
    raise KeyError(scene_id)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Write a reproducible rock-scene plan.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--random-scenes", type=int, default=2)
    parser.add_argument("--xacro", type=Path, default=default_xacro())
    parser.add_argument("--assets", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    assets = load_assets(args.assets)
    plan = build_plan(args.seed, assets, args.xacro.read_text(encoding="utf-8"), args.random_scenes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(args.out)


def list_main(argv=None):
    parser = argparse.ArgumentParser(description="Print scene ids for a seed.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--random-scenes", type=int, default=2)
    parser.add_argument("--xacro", type=Path, default=default_xacro())
    parser.add_argument("--assets", type=Path, default=None)
    args = parser.parse_args(argv)
    assets = load_assets(args.assets)
    plan = build_plan(args.seed, assets, args.xacro.read_text(encoding="utf-8"), args.random_scenes)
    for scene_id in list_scene_ids(plan):
        print(scene_id)


def _three_rock_scene(train_assets, base_to_optical, intrinsics):
    specs = (
        (1.55, -0.15, 0.2),
        (1.85, 0.45, 1.1),
        (2.15, -0.40, 2.4),
    )
    rocks = []
    for index, (x, y, yaw) in enumerate(specs):
        rocks.append(_placed(train_assets[index], f"rock_{index}", x, y, 0.0, 0.0, yaw, 1.0))
    views = _views_looking_at(rocks, base_to_optical, intrinsics, "scene_three")
    return _scene("scene_three", "three_rocks", rocks, views, ambient=0.42, azimuth=0.6, elevation=1.05, brightness=0.8)


def _empty_scene():
    return _scene(
        "scene_empty",
        "empty",
        [],
        [{"name": "view_00", "xyz": [0.0, 0.0, ROBOT_Z], "yaw": math.pi}],
        ambient=0.35,
        azimuth=1.2,
        elevation=0.9,
        brightness=0.55,
    )


def _clip_scene(asset, base_to_optical, intrinsics):
    rock = _placed(asset, "rock_0", 1.5, 0.15, 0.0, 0.0, 0.2, 1.0)
    view = None
    for distance in (1.2, 1.6, 2.2):
        for offset in (i * 0.04 for i in range(8, 18)):
            x = rock["xyz"][0] - distance * math.cos(offset)
            y = rock["xyz"][1] - distance * math.sin(offset)
            candidate = {"name": "view_00", "xyz": [x, y, ROBOT_Z], "yaw": 0.0}
            labels = _labels([rock], candidate, base_to_optical, intrinsics)
            if labels and any(item["truncated"] for item in labels):
                view = candidate
                break
        if view is not None:
            break
    if view is None:
        raise RuntimeError("clip scene did not place a rock on the image border")
    return _scene("scene_clip", "clip", [rock], [view], ambient=0.5, azimuth=2.2, elevation=0.7, brightness=0.95)


def _aligned_scene(front, back, base_to_optical, intrinsics):
    rocks = [
        _placed(front, "rock_0", 1.45, 0.0, 0.0, 0.0, 0.3, 1.15),
        _placed(back, "rock_1", 2.35, 0.02, 0.0, 0.0, 0.1, 1.0),
    ]
    view = {"name": "view_00", "xyz": [0.0, 0.0, ROBOT_Z], "yaw": 0.0}
    labels = _labels(rocks, view, base_to_optical, intrinsics)
    if len(labels) < 2 or not _boxes_overlap(labels[0], labels[1]):
        raise RuntimeError("aligned scene boxes do not overlap")
    return _scene("scene_aligned", "aligned", rocks, [view], ambient=0.28, azimuth=3.4, elevation=1.2, brightness=0.45)


def _holdout_scene(asset, base_to_optical, intrinsics):
    rock = _placed(asset, "rock_0", 1.7, 0.2, 0.0, 0.0, 0.8, 1.0)
    views = _views_looking_at([rock], base_to_optical, intrinsics, "scene_holdout", count=2)
    return _scene("scene_holdout", "holdout", [rock], views, ambient=0.6, azimuth=4.5, elevation=0.85, brightness=0.7)


def _random_scene(rng, train_assets, test_assets, base_to_optical, intrinsics, index, seed):
    use_holdout = rng.random() < 0.25
    pool = test_assets if use_holdout else train_assets
    count = 1 if use_holdout else rng.randint(1, 3)
    rocks = []
    for rock_index in range(count):
        asset = rng.choice(pool)
        placed = False
        for _ in range(40):
            x = rng.uniform(1.3, 2.6)
            y = rng.uniform(-0.7, 0.7)
            if all(math.hypot(x - other["xyz"][0], y - other["xyz"][1]) > 0.55 for other in rocks):
                rocks.append(_placed(
                    asset, f"rock_{rock_index}", x, y,
                    rng.uniform(-0.15, 0.15), rng.uniform(-0.1, 0.1), rng.uniform(0.0, math.tau),
                    rng.uniform(0.75, 1.25),
                ))
                placed = True
                break
        if not placed:
            break
    if not rocks:
        rocks.append(_placed(pool[0], "rock_0", 1.8, 0.0, 0.0, 0.0, rng.random(), 1.0))
    views = _views_looking_at(rocks, base_to_optical, intrinsics, f"scene_random_{index:02d}", count=2, rng=rng)
    purpose = "holdout_random" if use_holdout else "random"
    return _scene(
        f"scene_random_{index:02d}",
        purpose,
        rocks,
        views,
        ambient=rng.uniform(0.25, 0.7),
        azimuth=rng.uniform(0.0, math.tau),
        elevation=rng.uniform(0.45, 1.3),
        brightness=rng.uniform(0.35, 1.0),
        seed=seed,
    )


def _views_looking_at(rocks, base_to_optical, intrinsics, scene_id, count=2, rng=None):
    rng = rng or random.Random(0)
    cx = sum(rock["xyz"][0] for rock in rocks) / len(rocks)
    cy = sum(rock["xyz"][1] for rock in rocks) / len(rocks)
    views = []
    for index in range(count):
        found = None
        for attempt in range(36):
            angle = (index / count) * math.tau + attempt * 0.17
            distance = 1.35 + (attempt % 5) * 0.12
            x = cx - distance * math.cos(angle)
            y = cy - distance * math.sin(angle)
            yaw = math.atan2(cy - y, cx - x)
            view = {"name": f"view_{index:02d}", "xyz": [x, y, ROBOT_Z], "yaw": yaw}
            labels = _labels(rocks, view, base_to_optical, intrinsics)
            if labels:
                found = view
                break
        if found is None:
            raise RuntimeError(f"{scene_id} view {index} cannot see a rock")
        views.append(found)
    return views


def _placed(asset, name, x, y, roll, pitch, yaw, scale_factor):
    scale = asset["nominal_scale"] * scale_factor
    z = seat_on_ground(asset["bounds_min_m"], asset["bounds_max_m"], scale, roll, pitch, yaw)
    return {
        "name": name,
        "asset_id": asset["id"],
        "xyz": [x, y, z],
        "rpy": [roll, pitch, yaw],
        "scale": scale,
        "bounds_min": asset["bounds_min_m"],
        "bounds_max": asset["bounds_max_m"],
    }


def _labels(rocks, view, base_to_optical, intrinsics):
    records = []
    for rock in rocks:
        records.append({
            "name": rock["name"],
            "asset_id": rock["asset_id"],
            "class_name": "rock",
            "xyz": rock["xyz"],
            "quaternion": _rpy_quat(rock["rpy"]),
            "scale": rock["scale"],
            "bounds_min": rock["bounds_min"],
            "bounds_max": rock["bounds_max"],
        })
    camera = camera_for_view(view, base_to_optical)
    return annotate_rocks(records, camera, intrinsics, depth=None)


def _scene(scene_id, purpose, rocks, views, ambient, azimuth, elevation, brightness, seed=None):
    public_rocks = []
    for rock in rocks:
        public_rocks.append({
            "name": rock["name"],
            "asset_id": rock["asset_id"],
            "xyz": [float(value) for value in rock["xyz"]],
            "rpy": [float(value) for value in rock["rpy"]],
            "scale": float(rock["scale"]),
        })
    scene = {
        "id": scene_id,
        "purpose": purpose,
        "ambient": ambient,
        "sun_azimuth": azimuth,
        "sun_elevation": elevation,
        "sun_brightness": brightness,
        "rocks": public_rocks,
        "views": views,
    }
    if seed is not None:
        scene["generator_seed"] = seed
    return scene


def _split_for(scene, seed):
    purpose = scene["purpose"]
    if purpose in ("holdout", "holdout_random"):
        return "test"
    if purpose == "three_rocks":
        return "train"
    if purpose == "empty":
        return "train"
    if purpose in ("clip", "aligned"):
        return "val"
    digest = hashlib.sha256(f"{seed}:{scene['id']}".encode("utf-8")).hexdigest()
    if int(digest[:8], 16) % 5 == 0:
        return "val"
    return "train"


def _boxes_overlap(first, second):
    return not (
        first["xmax"] < second["xmin"] or second["xmax"] < first["xmin"]
        or first["ymax"] < second["ymin"] or second["ymax"] < first["ymin"]
    )


def _yaw_only(rpy):
    return abs(rpy[0]) < 1.0e-9 and abs(rpy[1]) < 1.0e-9


def _rpy_quat(rpy):
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "list":
        list_main(sys.argv[2:])
    else:
        main()
