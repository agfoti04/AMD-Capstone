"""Save RGB, calibration, depth, and YOLO labels from a running Gazebo scene.

The running world must be the scene named by `scene_id`. This node moves the
rover between that scene's viewpoints. It reads simulator poses only to write
labels; the perception node does not use this.
"""

import json
import time
from pathlib import Path

import rclpy
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image

from rover_training.assets import load_assets
from rover_training.geometry import (
    annotate_rocks,
    camera_rigid_from_base,
    intrinsics_from_camera_info,
    parse_camera_extrinsics,
    yaw_quaternion,
)
from rover_training.image_io import decode_ros_depth, decode_ros_image, write_calibration, write_image, write_pfm
from rover_training.randomize import build_plan, default_xacro, rocks_for_annotation, scene_by_id
from rover_training.yolo_export import write_dataset_index, write_label_file

SENSOR_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=5)


class DatasetCapture(Node):
    def __init__(self):
        super().__init__("dataset_capture")
        self.declare_parameter("seed", 7)
        self.declare_parameter("random_scenes", 2)
        self.declare_parameter("scene_id", "scene_three")
        self.declare_parameter("output_dir", "dataset/rocks")
        self.declare_parameter("plan_path", "")
        self.declare_parameter("xacro_path", "")
        self.declare_parameter("image_topic", "/oak/image_raw")
        self.declare_parameter("depth_topic", "/oak/depth/image_raw")
        self.declare_parameter("camera_info_topic", "/oak/camera_info")
        self.declare_parameter("robot_name", "four_wheel_bot")
        self.declare_parameter("settle_sec", 0.8)
        self.declare_parameter("ready_timeout_sec", 90.0)
        self.declare_parameter("min_visible_fraction", 0.30)
        self.declare_parameter("occlusion_tolerance_m", 0.08)
        self.declare_parameter("save_depth", True)

        self.output_dir = Path(self._string("output_dir")).expanduser()
        self.robot_name = self._string("robot_name")
        self.settle_sec = self._float("settle_sec")
        self.ready_timeout_sec = self._float("ready_timeout_sec")
        self.scene = self._load_scene()
        self.assets = load_assets()
        self.extrinsics = parse_camera_extrinsics(_xacro_text(self._string("xacro_path")))
        self.views = list(self.scene["views"])
        self.view_index = 0
        self.phase = "wait"
        self.started = time.monotonic()
        self.exit_code = 0
        self.done = False
        self.move_future = None
        self.settle_until = 0.0
        self.stamp_before = None
        self.image = None
        self.depth = None
        self.camera_info = None
        self.model_states = None
        self.odom = None
        self._set_state_clients = [
            self.create_client(SetEntityState, "/gazebo/set_entity_state"),
            self.create_client(SetEntityState, "/set_entity_state"),
        ]
        self.client = None

        self.create_subscription(Image, self._string("image_topic"), self._on_image, SENSOR_QOS)
        self.create_subscription(Image, self._string("depth_topic"), self._on_depth, SENSOR_QOS)
        self.create_subscription(CameraInfo, self._string("camera_info_topic"), self._on_info, SENSOR_QOS)
        self.create_subscription(ModelStates, "/gazebo/model_states", self._on_models, 10)
        self.create_subscription(Odometry, "/odom", self._on_odom, 10)
        self.get_logger().info(
            f"capturing {self.scene['id']} ({len(self.views)} views) into {self.output_dir}"
        )

    def tick(self):
        if self.done:
            return
        if self.phase == "wait":
            self._wait_ready()
        elif self.phase == "move":
            self._move()
        elif self.phase == "settle":
            self._settle()

    def _wait_ready(self):
        ready = next((client for client in self._set_state_clients if client.service_is_ready()), None)
        if self.image is not None and self.camera_info is not None and ready is not None:
            self.client = ready
            self.get_logger().info(f"camera and {self.client.srv_name} are ready")
            self.phase = "move"
            return
        if time.monotonic() - self.started > self.ready_timeout_sec:
            self._fail(
                "timed out waiting for the camera or set_entity_state. "
                "Run: ros2 topic list | grep oak   and   ros2 service list | grep entity_state"
            )

    def _move(self):
        if self.move_future is None:
            view = self.views[self.view_index]
            request = SetEntityState.Request()
            request.state.name = self.robot_name
            request.state.reference_frame = "world"
            request.state.pose.position.x = float(view["xyz"][0])
            request.state.pose.position.y = float(view["xyz"][1])
            request.state.pose.position.z = float(view["xyz"][2])
            qx, qy, qz, qw = yaw_quaternion(float(view["yaw"]))
            request.state.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
            self.stamp_before = _stamp_ns(self.image)
            self.move_future = self.client.call_async(request)
            self.get_logger().info(f"moving to {self.scene['id']} {view['name']}")
            return
        if not self.move_future.done():
            return
        result = self.move_future.result()
        self.move_future = None
        if result is None or not result.success:
            self._fail(f"set_entity_state rejected {self.views[self.view_index]['name']}")
            return
        self.settle_until = time.monotonic() + self.settle_sec
        self.phase = "settle"

    def _settle(self):
        if time.monotonic() < self.settle_until:
            return
        if self.image is None or self.camera_info is None:
            return
        if _stamp_ns(self.image) <= (self.stamp_before or -1):
            if time.monotonic() > self.settle_until + 8.0:
                self.get_logger().warning("no newer camera frame arrived; saving the latest frame")
            else:
                return
        self._save_view()
        self.view_index += 1
        if self.view_index >= len(self.views):
            write_dataset_index(self.output_dir)
            self.get_logger().info(f"wrote {self.output_dir}")
            self.done = True
            return
        self.phase = "move"

    def _save_view(self):
        view = self.views[self.view_index]
        intrinsics = intrinsics_from_camera_info(self.camera_info)
        width, height, rgb = decode_ros_image(self.image)
        base_xyz, base_quat, pose_source = self._robot_pose(view)
        camera = camera_rigid_from_base(base_xyz, base_quat, self.extrinsics)
        rocks = self._rocks()
        depth, depth_used = self._depth_for(intrinsics)
        labels = annotate_rocks(
            rocks,
            camera,
            intrinsics,
            depth=depth,
            min_visible_fraction=self._float("min_visible_fraction"),
            tolerance=self._float("occlusion_tolerance_m"),
        )
        split = self.scene["split"]
        stem = f"{self.scene['id']}_{view['name']}"
        image_path = write_image(self.output_dir / "images" / split / stem, width, height, rgb)
        write_label_file(self.output_dir / "labels" / split / f"{stem}.txt", labels, width, height)
        write_calibration(
            self.output_dir / "calibration" / split / f"{stem}.json",
            intrinsics,
            self.camera_info.header.frame_id,
            _stamp_ns(self.camera_info),
        )
        if depth is not None and self.get_parameter("save_depth").get_parameter_value().bool_value:
            write_pfm(self.output_dir / "depth" / split / f"{stem}.pfm", width, height, depth)
        record = {
            "scene_id": self.scene["id"],
            "view": view["name"],
            "split": split,
            "image": str(image_path),
            "boxes": len(labels),
            "pose_source": pose_source,
            "depth_used_for_occlusion": depth_used,
            "assets": [rock["asset_id"] for rock in self.scene["rocks"]],
            "labels": [
                {
                    "class_id": label["class_id"],
                    "class_name": label["class_name"],
                    "asset_id": label["asset_id"],
                    "visible_fraction": label["visible_fraction"],
                    "truncated": label["truncated"],
                }
                for label in labels
            ],
        }
        manifest = self.output_dir / "manifest.jsonl"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        self.get_logger().info(f"saved {image_path.name} with {len(labels)} rock boxes")

    def _rocks(self):
        planned = rocks_for_annotation(self.scene, self.assets)
        poses = self._model_pose_map()
        for rock in planned:
            if rock["name"] in poses:
                rock["xyz"] = poses[rock["name"]][0]
                rock["quaternion"] = poses[rock["name"]][1]
        return planned

    def _robot_pose(self, view):
        poses = self._model_pose_map()
        if self.robot_name in poses:
            return poses[self.robot_name][0], poses[self.robot_name][1], "model_states"
        if self.odom is not None:
            pose = self.odom.pose.pose
            xyz = (pose.position.x, pose.position.y, pose.position.z)
            quat = (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
            return xyz, quat, "odom"
        return tuple(view["xyz"]), yaw_quaternion(float(view["yaw"])), "commanded_view"

    def _model_pose_map(self):
        if self.model_states is None:
            return {}
        poses = {}
        for name, pose in zip(self.model_states.name, self.model_states.pose):
            poses[name] = (
                (pose.position.x, pose.position.y, pose.position.z),
                (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w),
            )
        return poses

    def _depth_for(self, intrinsics):
        if self.depth is None or abs(_stamp_ns(self.depth) - _stamp_ns(self.image)) > 100_000_000:
            return None, False
        try:
            values = decode_ros_depth(self.depth)
        except ValueError as exc:
            self.get_logger().warning(str(exc))
            return None, False
        if len(values) != intrinsics["width"] * intrinsics["height"]:
            return None, False
        return values, True

    def _load_scene(self):
        plan_path = self._string("plan_path")
        if plan_path:
            plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
        else:
            plan = build_plan(
                self.get_parameter("seed").get_parameter_value().integer_value,
                load_assets(),
                _xacro_text(self._string("xacro_path")),
                self.get_parameter("random_scenes").get_parameter_value().integer_value,
            )
        return scene_by_id(plan, self._string("scene_id"))

    def _on_image(self, msg):
        self.image = msg

    def _on_depth(self, msg):
        self.depth = msg

    def _on_info(self, msg):
        self.camera_info = msg

    def _on_models(self, msg):
        self.model_states = msg

    def _on_odom(self, msg):
        self.odom = msg

    def _fail(self, message):
        self.get_logger().error(message)
        self.exit_code = 2
        self.done = True

    def _string(self, name):
        return self.get_parameter(name).get_parameter_value().string_value

    def _float(self, name):
        return self.get_parameter(name).get_parameter_value().double_value


def _xacro_text(path):
    if path:
        return Path(path).read_text(encoding="utf-8")
    try:
        from ament_index_python.packages import get_package_share_directory
        shared = Path(get_package_share_directory("four_wheel_sim")) / "urdf" / "robot.urdf.xacro"
        if shared.is_file():
            return shared.read_text(encoding="utf-8")
    except Exception:
        pass
    return default_xacro().read_text(encoding="utf-8")


def _stamp_ns(msg):
    if msg is None:
        return -1
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


def main():
    rclpy.init()
    node = DatasetCapture()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
            node.tick()
    except KeyboardInterrupt:
        node.exit_code = 130
    code = node.exit_code
    node.destroy_node()
    rclpy.shutdown()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
