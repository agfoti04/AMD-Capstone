"""Launch the original maze or a Mars world in Harmonic and verify camera, boxes and driving."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import Detection2DArray

from .common import write_json
from .dataset import simulator_process


class RuntimeCheck(Node):
    def __init__(self):
        super().__init__('mars_runtime_check')
        self.odom = None
        self.images = []
        self.depth = None
        self.info = None
        self.boxes = None
        self.create_subscription(Odometry,'/odom',lambda m:setattr(self,'odom',m),qos_profile_sensor_data)
        self.create_subscription(Image,'/oak/image_raw',lambda m:self.images.append(m),qos_profile_sensor_data)
        self.create_subscription(Image,'/oak/depth/image_raw',lambda m:setattr(self,'depth',m),qos_profile_sensor_data)
        self.create_subscription(CameraInfo,'/oak/camera_info',lambda m:setattr(self,'info',m),qos_profile_sensor_data)
        self.create_subscription(Detection2DArray,'/oak/boxes',lambda m:setattr(self,'boxes',m),qos_profile_sensor_data)
        self.cmd = self.create_publisher(Twist,'/cmd_vel',10)

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
        times=[m.header.stamp.sec+m.header.stamp.nanosec*1e-9 for m in self.images]
        sim_hz=(len(times)-1)/(times[-1]-times[0])
        result={'mode':'mars' if mars else 'original','simulator':'gazebo-harmonic',
                'camera_dimensions':[self.info.width,self.info.height],
                'camera_frame':self.info.header.frame_id,'camera_hz_sim':sim_hz,'depth_encoding':self.depth.encoding,
                'gazebo_boxes_received':self.boxes is not None,
                'initial_rover_xyz':start_xyz,'drive_displacement_m':distance,'errors':[]}
        if distance<.06:
            result['errors'].append('Controller drive produced insufficient displacement')
        if self.depth.encoding!='32FC1':
            result['errors'].append('Depth is not 32FC1')
        result['wall_seconds']=time.monotonic()-wall
        return result


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
