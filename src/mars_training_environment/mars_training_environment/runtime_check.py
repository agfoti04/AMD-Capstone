"""Launch the original maze or a Mars world in Harmonic and verify camera, boxes and driving."""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection2DArray

from rover_training.geometry import (
    OPTICAL_CENTER_HEIGHT_M,
    mat_vec,
    optical_center_above_ground,
    quaternion_to_matrix,
    wheel_contact_drop,
)
from rover_training.randomize import default_xacro

from .common import write_json
from .dataset import simulator_process


def _stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


# Match the reliable publishers used by capture. Sensor-data best effort does
# not connect to those publishers.
SYNC_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=30)


class RuntimeCheck(Node):
    def __init__(self):
        super().__init__('mars_runtime_check')
        self.odom = None
        self.images = []
        self.image_times = []
        self.depth = None
        self.info = None
        self.boxes = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(Odometry,'/odom',lambda m:setattr(self,'odom',m),SYNC_QOS)
        self.create_subscription(Image,'/oak/image_raw',self._on_image,SYNC_QOS)
        self.create_subscription(Image,'/oak/depth/image_raw',lambda m:setattr(self,'depth',m),SYNC_QOS)
        self.create_subscription(CameraInfo,'/oak/camera_info',lambda m:setattr(self,'info',m),SYNC_QOS)
        self.create_subscription(Detection2DArray,'/oak/boxes',lambda m:setattr(self,'boxes',m),SYNC_QOS)
        self.cmd = self.create_publisher(Twist,'/cmd_vel',10)

    def _on_image(self, msg):
        self.images.append(msg)
        self.image_times.append((_stamp(msg), time.monotonic()))

    def wait(self,check,timeout=180):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            rclpy.spin_once(self,timeout_sec=.03)
            value=check()
            if value:
                return value
        raise TimeoutError('Runtime verification timed out')

    def sim_time(self):
        return self.odom.header.stamp.sec+self.odom.header.stamp.nanosec*1e-9

    def verify(self,mars):
        wall=time.monotonic()
        self.wait(lambda:len(self.images)>10 and self.odom is not None and self.depth is not None and self.info is not None
                  and (self.boxes is not None or not mars))
        beginning=self.sim_time()
        start_xyz=[self.odom.pose.pose.position.x,self.odom.pose.pose.position.y,self.odom.pose.pose.position.z]
        # Let controller spawners finish, then command a short straight drive.
        self.wait(lambda:self.sim_time()>beginning+3.)
        start_xy=np.array([self.odom.pose.pose.position.x,self.odom.pose.pose.position.y])
        start=self.sim_time()
        def drive():
            msg=Twist()
            msg.linear.x=.12
            self.cmd.publish(msg)
            return self.sim_time()>start+1.
        self.wait(drive)
        self.cmd.publish(Twist())
        self.wait(lambda:self.sim_time()>start+1.4)
        distance=float(np.linalg.norm(np.array([self.odom.pose.pose.position.x,self.odom.pose.pose.position.y])-start_xy))
        end_sim = self.image_times[-1][0]
        window = [item for item in self.image_times if item[0] >= end_sim - 2.0]
        if len(window) < 2 or window[-1][0] <= window[0][0] or window[-1][1] <= window[0][1]:
            sim_hz = 0.0
            wall_hz = 0.0
        else:
            sim_hz = (len(window) - 1) / (window[-1][0] - window[0][0])
            wall_hz = (len(window) - 1) / (window[-1][1] - window[0][1])
        rgb = self.images[-1]
        k = list(self.info.k)
        xacro_text = default_xacro().read_text(encoding='utf-8')
        expected_height = optical_center_above_ground(xacro_text)
        contact_drop = wheel_contact_drop(xacro_text)
        offset, optical_world = self._optical_center()
        kinematic_height = float(offset[2] + contact_drop)
        expected_fx = (self.info.width * 0.5) / math.tan(1.22 * 0.5)
        result={'mode':'mars' if mars else 'original','simulator':'gazebo-harmonic',
                'camera_dimensions':[self.info.width,self.info.height],
                'rgb_dimensions':[rgb.width, rgb.height],
                'depth_dimensions':[self.depth.width, self.depth.height],
                'camera_frame':self.info.header.frame_id,
                'depth_frame':self.depth.header.frame_id,
                'camera_hz_sim':sim_hz,'camera_hz_wall':wall_hz,
                'camera_k':k,'expected_fx_at_provisional_hfov':expected_fx,
                'depth_encoding':self.depth.encoding,
                'gazebo_boxes_received':self.boxes is not None,
                'initial_rover_xyz':start_xyz,'drive_displacement_m':distance,
                'base_link_z_m':float(self.odom.pose.pose.position.z),
                'optical_offset_base_m':offset.tolist(),
                'optical_center_above_contact_m':kinematic_height,
                'optical_center_world_z_m':float(optical_world[2]),
                'expected_optical_center_height_m':expected_height,
                'errors':[]}
        if distance<.06:
            result['errors'].append('Controller drive produced insufficient displacement')
        if self.depth.encoding!='32FC1':
            result['errors'].append('Depth is not 32FC1')
        if [self.info.width, self.info.height] != [1920, 1080] or [rgb.width, rgb.height] != [1920, 1080]:
            result['errors'].append('RGB or camera_info is not 1920x1080')
        if (self.depth.width, self.depth.height) != (rgb.width, rgb.height):
            result['errors'].append('Depth resolution does not match RGB')
        if self.depth.header.frame_id != rgb.header.frame_id:
            result['errors'].append('RGB and depth frame_ids differ')
        if sim_hz < 24.0:
            result['errors'].append(f'Simulation-time camera rate {sim_hz:.2f} Hz is below the 30 Hz setting')
        if abs(k[0] - expected_fx) / expected_fx > 0.02:
            result['errors'].append('camera_info fx does not match the provisional 1.22 rad horizontal FOV')
        if abs(kinematic_height - OPTICAL_CENTER_HEIGHT_M) > 0.005 or abs(expected_height - OPTICAL_CENTER_HEIGHT_M) > 1e-6:
            result['errors'].append(
                f'Optical center is {kinematic_height:.4f} m above the wheel contact, not {OPTICAL_CENTER_HEIGHT_M} m')
        if abs(float(optical_world[2]) - OPTICAL_CENTER_HEIGHT_M) > 0.03:
            result['errors'].append(
                f'Optical center world z is {float(optical_world[2]):.4f} m on level ground')
        result['wall_seconds']=time.monotonic()-wall
        return result

    def _optical_center(self):
        deadline = time.monotonic() + 10.0
        transform = None
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            try:
                transform = self.tf_buffer.lookup_transform('base_link', 'camera_optical_frame', rclpy.time.Time())
                break
            except TransformException:
                transform = None
        if transform is None:
            raise TimeoutError('camera_optical_frame TF was not published')
        offset = np.array([
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z,
        ])
        pose = self.odom.pose.pose
        rotation = quaternion_to_matrix(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
        base = (pose.position.x, pose.position.y, pose.position.z)
        world = tuple(base[i] + mat_vec(rotation, tuple(offset))[i] for i in range(3))
        return offset, world


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',type=Path,help='Omit to verify the original maze in Harmonic')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.scene:
        cmd=['ros2','launch','mars_training_environment','mars_world.launch.py',f'scene:={args.scene.resolve()}','gui:=false']
    else:
        cmd=['ros2','launch','four_wheel_sim','gz_harmonic.launch.py','gui:=false']
    # This process joins the same container-local ROS graph as the simulator.
    os.environ.setdefault('ROS_AUTOMATIC_DISCOVERY_RANGE','LOCALHOST')
    report={'errors':['not run']}
    with args.output.with_suffix('.log').open('w') as log, simulator_process(cmd,log):
        rclpy.init()
        node=RuntimeCheck()
        try:
            report=node.verify(bool(args.scene))
            write_json(args.output,report)
            print(json.dumps(report,indent=2))
        finally:
            node.cmd.publish(Twist())
            node.destroy_node()
            rclpy.shutdown()
    raise SystemExit(bool(report['errors']))


if __name__ == '__main__':
    main()
