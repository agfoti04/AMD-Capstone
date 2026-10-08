"""Capture clean Gazebo Harmonic RGB with time-matched simulator geometry and labels."""
import json
import math
import time
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.pose_pb2 import Pose
from gz.transport13 import Node as GzNode
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import Detection2DArray

# Match the reliable camera and ground-truth publishers already used by rover_training.
SYNC_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=20)

from rover_training.geometry import intrinsics_from_camera_info, parse_camera_extrinsics
from rover_training.image_io import decode_ros_depth, decode_ros_image, write_image, write_pfm
from rover_training.preview import draw_boxes
from rover_training.yolo_export import write_label_file

from .annotation import Annotator, compare_boxes
from .common import quaternion, write_json
from .generate import ROCK_LABEL, WORLD_NAME


def stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9


def pose_values(msg):
    p = msg.pose.pose
    return np.array([p.position.x,p.position.y,p.position.z]), np.array([p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w])


def _nearest(buffer, time_sec, window):
    best = None
    best_dt = window
    for msg in buffer:
        delta = abs(stamp(msg) - time_sec)
        if delta <= best_dt:
            best, best_dt = msg, delta
    return best


def gazebo_rock_boxes(msg):
    """Pixel xyxy boxes for the rock label from Gazebo's bounding-box camera."""
    boxes = []
    for detection in msg.detections:
        if detection.results and detection.results[0].hypothesis.class_id != str(ROCK_LABEL):
            continue
        cx, cy = detection.bbox.center.position.x, detection.bbox.center.position.y
        w, h = detection.bbox.size_x, detection.bbox.size_y
        boxes.append([cx-w/2, cy-h/2, cx+w/2, cy+h/2])
    return boxes


def interpolated_pose(odometry, image_time, max_gap):
    before = [m for m in odometry if stamp(m) <= image_time]
    after = [m for m in odometry if stamp(m) >= image_time]
    if not before or not after:
        return None
    a,b = before[-1],after[0]
    ta,tb = stamp(a),stamp(b)
    if image_time-ta > max_gap or tb-image_time > max_gap:
        return None
    x0,q0 = pose_values(a)
    x1,q1 = pose_values(b)
    alpha = (image_time-ta)/(tb-ta) if tb > ta else 0.
    if np.dot(q0,q1)<0:
        q1 = -q1
    q = q0*(1-alpha)+q1*alpha
    q /= np.linalg.norm(q)
    return x0*(1-alpha)+x1*alpha,q,[ta,tb]


class Capture(Node):
    def __init__(self):
        super().__init__('mars_capture')
        self.declare_parameter('scene','')
        self.declare_parameter('output','dataset/mars_sample')
        path = Path(self.get_parameter('scene').value).resolve()
        self.scene = json.loads(path.read_text())
        self.root = Path(self.get_parameter('output').value).resolve()
        self.cfg = self.scene['config']
        self.cap = self.cfg['capture']
        self.annotator = Annotator(path.parents[2],self.scene)
        self.extrinsics = parse_camera_extrinsics((path.parent/'robot_capture.urdf').read_text())
        self.images,self.depths,self.infos,self.odometry = deque(maxlen=16),deque(maxlen=16),deque(maxlen=16),deque(maxlen=200)
        self.boxes = deque(maxlen=16)
        camera = self.cfg['camera']
        for msg_type,topic,buffer in ((Image,camera['image_topic'],self.images),(Image,camera['depth_topic'],self.depths),
                                       (CameraInfo,camera['camera_info_topic'],self.infos),(Odometry,'/odom',self.odometry),
                                       (Detection2DArray,camera['boxes_topic'],self.boxes)):
            self.create_subscription(msg_type,topic,lambda msg,buffer=buffer: buffer.append(msg),SYNC_QOS)
        self.gz = GzNode()
        self.records = []

    def set_pose(self, view):
        req = Pose()
        req.name = 'four_wheel_bot'
        req.position.x,req.position.y,req.position.z = view['xyz']
        q = quaternion(view['rpy'])
        req.orientation.x,req.orientation.y,req.orientation.z,req.orientation.w = q
        ok,reply = self.gz.request(f'/world/{WORLD_NAME}/set_pose',req,Pose,Boolean,3000)
        return ok and reply.data

    def wait(self, predicate, reason, timeout=None):
        deadline = time.monotonic() + (timeout or self.cap['timeout_wall_sec'])
        while rclpy.ok() and time.monotonic()<deadline:
            rclpy.spin_once(self,timeout_sec=.03)
            result = predicate()
            if result:
                return result
        text = reason() if callable(reason) else reason
        raise TimeoutError(text)

    def synchronized(self, after_time):
        for image in reversed(self.images):
            ts = stamp(image)
            if ts <= after_time:
                continue
            info = _nearest(self.infos, ts, 0.05)
            if info is None:
                continue
            # Software rendering can drop the ground-truth rate below the nominal
            # 0.06 s bracket. Keep the configured value, but not below 0.2 s.
            pose = interpolated_pose(self.odometry, ts, max(self.cap['max_pose_gap_sec'], 0.2))
            if pose is None:
                continue
            depth = _nearest(self.depths, ts, 0.05)
            # Depth is the independent check that the mesh projection matches the sensor.
            # Saving the depth file stays optional.
            if (self.cap.get('require_depth_check', True) or self.cap['save_depth']) and depth is None:
                continue
            boxes = _nearest(self.boxes, ts, 0.05)
            if self.cap['require_bbox_check'] and boxes is None:
                continue
            if (image.width,image.height)!=(info.width,info.height) or image.header.frame_id != info.header.frame_id:
                raise ValueError('RGB/CameraInfo geometry or frame mismatch')
            if depth is not None and (depth.width, depth.height) != (image.width, image.height):
                raise ValueError(
                    f'Depth {depth.width}x{depth.height} does not match RGB {image.width}x{image.height}')
            if (depth is not None and depth.header.frame_id and image.header.frame_id
                    and depth.header.frame_id != image.header.frame_id):
                raise ValueError('RGB and depth frame_ids differ')
            return image,info,depth,pose,boxes
        return None

    def _sync_hint(self):
        def last(buffer):
            return None if not buffer else round(stamp(buffer[-1]), 3)
        return (f'images={len(self.images)} depth={len(self.depths)} boxes={len(self.boxes)} '
                f'odom={len(self.odometry)} last_image={last(self.images)} '
                f'last_depth={last(self.depths)} last_boxes={last(self.boxes)} last_odom={last(self.odometry)}')

    def run(self):
        started = time.monotonic()
        self.wait(lambda:self.images and self.depths and self.odometry and
                  (self.boxes or not self.cap['require_bbox_check']),
                  lambda: 'Camera, depth, boxes, or /odom not ready; '+self._sync_hint())
        sim_start = stamp(self.odometry[-1])
        for view in self.scene['views']:
            view_started = time.monotonic()
            before = stamp(self.images[-1])
            if not self.set_pose(view):
                raise RuntimeError('Gazebo set_pose rejected the viewpoint')
            # Require odometry at the commanded XY, then a full settling interval
            # in simulation time. Never fall back to commanded pose or stale RGB.
            def arrived():
                if not self.odometry or stamp(self.odometry[-1]) <= before:
                    return False
                xyz,_ = pose_values(self.odometry[-1])
                return np.linalg.norm(xyz[:2]-np.asarray(view['xyz'][:2]))<.10
            self.wait(arrived,'Rover did not reach the requested viewpoint')
            after = stamp(self.odometry[-1]) + self.cap['settle_sim_sec']
            sample = self.wait(lambda:self.synchronized(after),
                                lambda: 'No synchronized RGB / camera info / odometry sample; '+self._sync_hint())
            self.save(view,sample,time.monotonic()-view_started)
        elapsed = time.monotonic()-started
        sim_elapsed = stamp(self.odometry[-1])-sim_start
        report = {'scene_id':self.scene['id'],'config_sha256':self.scene['config_sha256'],
                  'scene_sha256':self.scene['content_sha256'],'frames':len(self.records),
                  'wall_seconds':elapsed,'sim_seconds':sim_elapsed,'real_time_factor':sim_elapsed/max(elapsed,1e-9),
                  'images_per_wall_second':len(self.records)/elapsed,'records':self.records}
        write_json(self.root/'reports'/f'{self.scene["id"]}.json',report)
        self.get_logger().info(f'Completed {self.scene["id"]}: {len(self.records)} frames in {elapsed:.1f}s')

    def save(self,view,sample,wait_seconds):
        image,info,depth,pose,boxes = sample
        xyz,quat,bracket = pose
        intrinsics = intrinsics_from_camera_info(info)
        width,height,rgb = decode_ros_image(image)
        measured = None
        if depth is not None and depth.width==width and depth.height==height:
            measured = decode_ros_depth(depth)
        began = time.monotonic()
        labels,objects,quality,instances = self.annotator.annotate(xyz,quat,self.extrinsics,intrinsics,measured)
        annotation_seconds = time.monotonic()-began
        # Independent second check from Gazebo's renderer. Recorded, never used as labels.
        bbox_check = None
        if boxes is not None:
            bbox_check = compare_boxes(labels,objects,gazebo_rock_boxes(boxes),self.cfg['labels'])
            bbox_check['timestamp_sim_sec'] = stamp(boxes)
        # Geometric cross-check independent of labels. Fail on gross calibration,
        # stale-pose, mesh scale or transform errors rather than write a bad dataset.
        if quality.get('rock_depth_median_error_m',0)>.08:
            diagnostics = self.root/'diagnostics'/self.scene['id']/view['id']
            diagnostics.mkdir(parents=True,exist_ok=True)
            write_image(diagnostics/'rgb',width,height,rgb)
            write_image(diagnostics/'boxes',width,height,draw_boxes(rgb,width,height,labels))
            np.savez_compressed(diagnostics/'geometry.npz',instances=instances,depth=measured,
                                xyz=xyz,quaternion=quat)
            write_json(diagnostics/'camera.json',{'intrinsics':intrinsics,'extrinsics':self.extrinsics,
                                                  'objects':objects,'quality':quality})
            raise ValueError(f'Rock mesh / sensor depth disagree: {quality}')
        split = self.scene['split']
        stem = self.scene['id']+'_'+view['id']
        metadata_path = self.root/'metadata'/split/(stem+'.json')
        if metadata_path.exists():
            raise FileExistsError(f'{metadata_path} already exists; resume completed scenes or use a fresh output')
        dest = self.root/'images'/split/stem
        image_path = write_image(dest,width,height,rgb)
        label_path = self.root/'labels'/split/(stem+'.txt')
        write_label_file(label_path,labels,width,height)
        if self.cap['previews']:
            painted = draw_boxes(rgb,width,height,labels)
            write_image(self.root/'previews'/split/stem,width,height,painted)
        if measured is not None and self.cap['save_depth']:
            write_pfm(self.root/'depth'/split/(stem+'.pfm'),width,height,measured)
        record = {'scene_id':self.scene['id'],'scene_group':self.scene['scene_group'],'seed':self.scene['seed'],
                  'split':split,'view':view['id'],'scene_sha256':self.scene['content_sha256'],
                  'config_sha256':self.scene['config_sha256'],'image':image_path.relative_to(self.root).as_posix(),
                  'label':label_path.relative_to(self.root).as_posix(),'timestamp_sim_sec':stamp(image),
                  'camera_frame':image.header.frame_id,'intrinsics':intrinsics,
                  'base_xyz':xyz.tolist(),'base_quaternion':quat.tolist(),'pose_source':'interpolated_ground_truth_odom',
                  'pose_bracket_sim_sec':bracket,'camera_extrinsics':self.extrinsics,
                  'objects':objects,'labels':labels,'depth_quality':quality,'bbox_check':bbox_check,
                  'wait_seconds':wait_seconds,'annotation_seconds':annotation_seconds,'clean_rgb':True}
        write_json(metadata_path,record)
        self.records.append(metadata_path.relative_to(self.root).as_posix())
        agreement = 'n/a' if bbox_check is None else (
            f"{bbox_check['matched']}/{len(labels)} matched, {bbox_check['gazebo_boxes_unexplained']} unexplained")
        self.get_logger().info(f'{stem}: {len(labels)} labels; annotation {annotation_seconds:.2f}s; '
                               f'depth {quality}; gazebo boxes {agreement}')


def main():
    rclpy.init()
    node = None
    try:
        node = Capture()
        node.run()
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
