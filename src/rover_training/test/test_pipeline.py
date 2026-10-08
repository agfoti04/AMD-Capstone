import json
import math
import tempfile
import unittest
from pathlib import Path

from rover_training.assets import build_catalog, catalog_path, load_assets, models_dir
from rover_training.geometry import (
    annotate_rocks,
    camera_rigid_from_base,
    class_id_for,
    nominal_intrinsics,
    parse_camera_extrinsics,
    pose_to_rigid,
    quaternion_to_matrix,
    rpy_to_matrix,
    transform_point,
    yaw_quaternion,
)
from rover_training.image_io import encode_bmp
from rover_training.preview import draw_boxes, load_rgb, preview_dataset
from rover_training.randomize import _rpy_quat, build_plan, default_xacro, rocks_for_annotation
from rover_training.world_builder import build_world_sdf
from rover_training.yolo_export import read_label_file, write_dataset_index, write_label_file

XACRO = default_xacro()
GAZEBO_LAUNCH = XACRO.parents[1] / "launch" / "gazebo.launch.py"
DETECTOR = Path(__file__).resolve().parents[2] / "rover_perception" / "rover_perception" / "detector_node.py"


class GeometryTests(unittest.TestCase):
    def test_optical_frame_looks_forward_and_down(self):
        extrinsics = parse_camera_extrinsics(XACRO.read_text(encoding="utf-8"))
        camera = camera_rigid_from_base((0.0, 0.0, 0.09), yaw_quaternion(0.0), extrinsics)
        intrinsics = nominal_intrinsics()
        ground = transform_point(*camera, (2.0, 0.0, 0.0))
        self.assertGreater(ground[2], 0.2)
        u = intrinsics["fx"] * ground[0] / ground[2] + intrinsics["cx"]
        v = intrinsics["fy"] * ground[1] / ground[2] + intrinsics["cy"]
        self.assertGreater(u, intrinsics["width"] * 0.35)
        self.assertLess(u, intrinsics["width"] * 0.65)
        self.assertGreater(v, intrinsics["cy"])
        self.assertLess(v, intrinsics["height"])

    def test_rpy_matches_quaternion(self):
        rpy = (0.1, -0.2, 0.7)
        matrix = rpy_to_matrix(*rpy)
        quat_matrix = quaternion_to_matrix(*_rpy_quat(rpy))
        for row in range(3):
            for col in range(3):
                self.assertAlmostEqual(matrix[row][col], quat_matrix[row][col], places=6)

    def test_centered_box_is_kept_and_maps_to_class_zero(self):
        intrinsics = _small_camera()
        labels = annotate_rocks([_front_rock()], _identity_camera(), intrinsics, depth=_flat_depth(intrinsics, 2.0))
        self.assertEqual(len(labels), 1)
        self.assertEqual(labels[0]["class_id"], 0)
        self.assertEqual(class_id_for("rock"), 0)
        self.assertIsNone(class_id_for("tree"))
        self.assertGreater(labels[0]["visible_fraction"], 0.8)

    def test_full_occlusion_drops_the_label(self):
        intrinsics = _small_camera()
        labels = annotate_rocks([_front_rock()], _identity_camera(), intrinsics, depth=_flat_depth(intrinsics, 0.4))
        self.assertEqual(labels, [])

    def test_partial_occlusion_keeps_the_full_box(self):
        intrinsics = _small_camera()
        depth = []
        for _y in range(intrinsics["height"]):
            for x in range(intrinsics["width"]):
                depth.append(2.0 if x < intrinsics["width"] // 2 else 0.4)
        clear = annotate_rocks([_front_rock()], _identity_camera(), intrinsics, depth=_flat_depth(intrinsics, 2.0))
        partial = annotate_rocks([_front_rock()], _identity_camera(), intrinsics, depth=depth)
        self.assertEqual(len(partial), 1)
        self.assertAlmostEqual(partial[0]["xmin"], clear[0]["xmin"], places=4)
        self.assertAlmostEqual(partial[0]["xmax"], clear[0]["xmax"], places=4)
        self.assertGreater(partial[0]["visible_fraction"], 0.30)
        self.assertLess(partial[0]["visible_fraction"], 0.85)

    def test_empty_scene_writes_an_empty_label_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.txt"
            write_label_file(path, [], 100, 80)
            self.assertEqual(path.read_text(encoding="utf-8"), "")
            self.assertEqual(read_label_file(path, 100, 80), [])

    def test_clipped_box_stays_inside_the_image(self):
        intrinsics = _small_camera()
        rock = _front_rock()
        rock["bounds_min"] = (1.2, -0.25, 2.0)
        rock["bounds_max"] = (2.4, 0.25, 2.5)
        labels = annotate_rocks([rock], _identity_camera(), intrinsics, depth=None)
        self.assertEqual(len(labels), 1)
        self.assertTrue(labels[0]["truncated"])
        self.assertAlmostEqual(labels[0]["xmax"], intrinsics["width"] - 1, places=3)
        self.assertGreater(labels[0]["xmin"], intrinsics["cx"])
        line = write_and_read(labels, intrinsics)
        self.assertEqual(line[0]["class_id"], 0)
        self.assertLessEqual(line[0]["xmax"], intrinsics["width"])

    def test_rock_hidden_just_behind_another_rock_is_dropped(self):
        intrinsics = _small_camera()
        hidden = _front_rock()
        hidden["name"] = "rock_1"
        hidden["bounds_min"] = (-0.3, -0.2, 2.05)
        hidden["bounds_max"] = (0.3, 0.2, 2.3)
        depth = _flat_depth(intrinsics, 2.0)
        alone = annotate_rocks([hidden], _identity_camera(), intrinsics, depth=depth)
        self.assertEqual(len(alone), 1)
        labels = annotate_rocks([_front_rock(), hidden], _identity_camera(), intrinsics, depth=depth)
        self.assertEqual([label["model_name"] for label in labels], ["rock_0"])

    def test_mesh_vertices_give_a_tighter_box_inside_the_aabb_box(self):
        intrinsics = _small_camera()
        loose = annotate_rocks([_front_rock()], _identity_camera(), intrinsics)[0]
        rock = _front_rock()
        rock["vertices"] = [(-0.4, 0.0, 2.3), (0.4, 0.0, 2.3), (0.0, -0.2, 2.3), (0.0, 0.2, 2.3)]
        tight = annotate_rocks([rock], _identity_camera(), intrinsics)[0]
        self.assertGreater(tight["xmin"], loose["xmin"])
        self.assertLess(tight["xmax"], loose["xmax"])
        self.assertGreater(tight["ymin"], loose["ymin"])
        self.assertLess(tight["ymax"], loose["ymax"])
        self.assertAlmostEqual(tight["xmin"], 50.0 * -0.4 / 2.3 + 49.5, places=4)
        self.assertAlmostEqual(tight["ymax"], 50.0 * 0.2 / 2.3 + 39.5, places=4)

    def test_vertices_behind_the_camera_fall_back_to_the_aabb(self):
        intrinsics = _small_camera()
        loose = annotate_rocks([_front_rock()], _identity_camera(), intrinsics)[0]
        rock = _front_rock()
        rock["vertices"] = [(0.0, 0.0, -1.0), (0.1, 0.1, 2.3)]
        fallback = annotate_rocks([rock], _identity_camera(), intrinsics)[0]
        self.assertAlmostEqual(fallback["xmin"], loose["xmin"], places=6)
        self.assertAlmostEqual(fallback["ymax"], loose["ymax"], places=6)

    def test_unknown_class_is_not_written_as_a_rock(self):
        rock = _front_rock()
        rock["class_name"] = "wall"
        self.assertEqual(annotate_rocks([rock], _identity_camera(), _small_camera()), [])


class PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assets = build_catalog()["assets"]
        cls.xacro = XACRO.read_text(encoding="utf-8")
        cls.plan = build_plan(7, cls.assets, cls.xacro, random_scenes=2)

    def test_saved_catalog_matches_the_meshes(self):
        saved = json.loads(catalog_path().read_text(encoding="utf-8"))
        self.assertEqual(saved, build_catalog(models_dir()))

    def test_catalog_bounds_match_the_meshes(self):
        fresh = build_catalog(models_dir())
        self.assertEqual(len(fresh["assets"]), 4)
        self.assertTrue(fresh["assets"][3]["held_out"])
        for asset in fresh["assets"]:
            self.assertFalse(asset["temporary"])
            self.assertGreater(asset["nominal_scale"], 0.0)
            self.assertLess(asset["nominal_scale"], 2.0)

    def test_three_rock_scene_and_scene_level_splits(self):
        three = _scene(self.plan, "scene_three")
        self.assertEqual(len(three["rocks"]), 3)
        self.assertEqual(three["split"], "train")
        self.assertGreaterEqual(len(three["views"]), 2)
        self.assertGreaterEqual(len(_scene(self.plan, "scene_holdout")["views"]), 2)
        self.assertEqual(_scene(self.plan, "scene_empty")["rocks"], [])
        self.assertEqual(_scene(self.plan, "scene_empty")["split"], "train")
        self.assertEqual(_scene(self.plan, "scene_clip")["split"], "val")
        self.assertEqual(_scene(self.plan, "scene_aligned")["split"], "val")
        holdout = _scene(self.plan, "scene_holdout")
        self.assertEqual(holdout["split"], "test")
        self.assertTrue(all(rock["asset_id"] == "falling_rock4" for rock in holdout["rocks"]))
        for scene in self.plan["scenes"]:
            ids = {rock["asset_id"] for rock in scene["rocks"]}
            if ids & set(self.plan["held_out_assets"]):
                self.assertEqual(ids, set(self.plan["held_out_assets"]))
                self.assertEqual(scene["split"], "test")
            else:
                self.assertNotEqual(scene["split"], "test")

    def test_same_seed_is_stable_and_views_share_a_scene(self):
        again = build_plan(7, self.assets, self.xacro, random_scenes=2)
        self.assertEqual(again, self.plan)
        other = build_plan(8, self.assets, self.xacro, random_scenes=2)
        self.assertNotEqual(other["scenes"], self.plan["scenes"])

    def test_maze_launch_does_not_take_a_world_argument(self):
        text = GAZEBO_LAUNCH.read_text(encoding="utf-8")
        self.assertIn("make_world()", text)
        self.assertNotIn("DeclareLaunchArgument", text)

    def test_world_sdf_is_classic_and_uses_box_collision(self):
        three = _scene(self.plan, "scene_three")
        sdf = build_world_sdf(three, self.assets)
        self.assertIn("<sdf version=\"1.6\">", sdf)
        self.assertIn("libgazebo_ros_state.so", sdf)
        self.assertNotIn("boundingbox", sdf.lower())
        self.assertNotIn("wall_", sdf)
        self.assertEqual(sdf.count("<model name=\"rock_"), 3)
        self.assertIn("<box>", sdf)
        self.assertNotIn("Collider", sdf)

    def test_preview_alignment_on_a_synthetic_image(self):
        intrinsics = _small_camera()
        labels = annotate_rocks([_front_rock()], _identity_camera(), intrinsics, depth=None)
        rgb = bytes([20, 20, 20]) * (intrinsics["width"] * intrinsics["height"])
        painted = draw_boxes(rgb, intrinsics["width"], intrinsics["height"], labels)
        x = int(round((labels[0]["xmin"] + labels[0]["xmax"]) * 0.5))
        y = int(round(labels[0]["ymin"]))
        pixel = painted[(y * intrinsics["width"] + x) * 3:(y * intrinsics["width"] + x) * 3 + 3]
        self.assertEqual(pixel, b"\x28\xdc\x3c")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "images" / "train" / "frame.bmp"
            image.parent.mkdir(parents=True)
            image.write_bytes(encode_bmp(intrinsics["width"], intrinsics["height"], rgb))
            write_label_file(root / "labels" / "train" / "frame.txt", labels, intrinsics["width"], intrinsics["height"])
            write_dataset_index(root)
            written = preview_dataset(root)
            self.assertEqual(len(written), 1)
            preview_w, preview_h, preview_rgb = load_rgb(written[0])
            self.assertEqual(preview_w, intrinsics["width"])
            self.assertEqual(preview_h, intrinsics["height"])
            saved = preview_rgb[(y * intrinsics["width"] + x) * 3:(y * intrinsics["width"] + x) * 3 + 3]
            self.assertEqual(saved, b"\x28\xdc\x3c")
            self.assertIn("names:", (root / "data.yaml").read_text(encoding="utf-8"))
            self.assertIn("0: rock", (root / "data.yaml").read_text(encoding="utf-8"))

    def test_detector_source_does_not_read_simulator_truth(self):
        text = DETECTOR.read_text(encoding="utf-8")
        for banned in ("model_states", "odom_gt", "get_entity_state", "set_entity_state"):
            self.assertNotIn(banned, text)


def _scene(plan, scene_id):
    for scene in plan["scenes"]:
        if scene["id"] == scene_id:
            return scene
    raise KeyError(scene_id)


def _small_camera():
    return {"width": 100, "height": 80, "fx": 50.0, "fy": 50.0, "cx": 49.5, "cy": 39.5}


def _identity_camera():
    return pose_to_rigid((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))


def _front_rock():
    return {
        "name": "rock_0",
        "asset_id": "unit",
        "class_name": "rock",
        "xyz": (0.0, 0.0, 0.0),
        "quaternion": (0.0, 0.0, 0.0, 1.0),
        "scale": 1.0,
        "bounds_min": (-0.8, -0.4, 2.0),
        "bounds_max": (0.8, 0.4, 2.6),
    }


def _flat_depth(intrinsics, value):
    return [value] * (intrinsics["width"] * intrinsics["height"])


def write_and_read(labels, intrinsics):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "box.txt"
        write_label_file(path, labels, intrinsics["width"], intrinsics["height"])
        return read_label_file(path, intrinsics["width"], intrinsics["height"])


class ExportRoundTripTests(unittest.TestCase):
    def test_annotation_rocks_use_catalog_bounds(self):
        assets = build_catalog()["assets"]
        plan = build_plan(7, assets, XACRO.read_text(encoding="utf-8"), random_scenes=0)
        three = _scene(plan, "scene_three")
        rocks = rocks_for_annotation(three, assets)
        self.assertEqual(len(rocks), 3)
        self.assertTrue(all(rock["class_name"] == "rock" for rock in rocks))
        self.assertTrue(math.isfinite(rocks[0]["xyz"][2]))

    def test_loaded_assets_carry_mesh_vertices_inside_the_bounds(self):
        for asset in load_assets():
            vertices = asset["vertices_m"]
            self.assertEqual(len(vertices), asset["vertices"])
            for axis in range(3):
                self.assertAlmostEqual(min(v[axis] for v in vertices), asset["bounds_min_m"][axis], places=9)
                self.assertAlmostEqual(max(v[axis] for v in vertices), asset["bounds_max_m"][axis], places=9)


if __name__ == "__main__":
    unittest.main()
