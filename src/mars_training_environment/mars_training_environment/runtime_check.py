"""Launch the original or Mars world, verify camera/control, and drop collision probes."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import rclpy
from gazebo_msgs.srv import DeleteEntity, GetEntityState, SpawnEntity
from gazebo_msgs.msg import ContactsState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from .common import rotation, write_json
from .dataset import simulator_process
from .terrain import Terrain


class RuntimeCheck(Node):
    def __init__(self):
        super().__init__('mars_runtime_check')
        self.odom = None
        self.images = []
        self.depth = None
        self.info = None
        self.contacts = []
        self.create_subscription(Odometry,'/odom',lambda m:setattr(self,'odom',m),qos_profile_sensor_data)
        self.create_subscription(Image,'/oak/image_raw',self.images.append,qos_profile_sensor_data)
        self.create_subscription(Image,'/oak/depth/image_raw',lambda m:setattr(self,'depth',m),qos_profile_sensor_data)
        self.create_subscription(CameraInfo,'/oak/camera_info',lambda m:setattr(self,'info',m),qos_profile_sensor_data)
        self.cmd = self.create_publisher(Twist,'/cmd_vel',10)
        self.spawn = self.create_client(SpawnEntity,'/spawn_entity')
        self.delete = self.create_client(DeleteEntity,'/delete_entity')
        self.get_state = self.create_client(GetEntityState,'/gazebo/get_entity_state')
        self.create_subscription(ContactsState,'/mars_probe/contacts',self.on_contacts,10)

    def on_contacts(self,msg):
        for state in msg.states:
            self.contacts.extend([state.collision1_name,state.collision2_name])

    def wait(self,check,timeout=120):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            rclpy.spin_once(self,timeout_sec=.03)
            value=check()
            if value:
                return value
        raise TimeoutError('Runtime verification timed out')

    def sim_time(self):
        return self.odom.header.stamp.sec+self.odom.header.stamp.nanosec*1e-9

    def call(self,client,req):
        self.wait(client.service_is_ready)
        future=client.call_async(req)
        self.wait(future.done)
        response=future.result()
        if not response.success:
            raise RuntimeError(str(response))
        return response

    def probe(self,name,xyz):
        self.contacts.clear()
        req=SpawnEntity.Request()
        req.name=name
        req.xml=f'''<sdf version="1.6"><model name="{name}"><link name="body">
<inertial><mass>0.2</mass><inertia><ixx>0.0000533</ixx><iyy>0.0000533</iyy><izz>0.0000533</izz></inertia></inertial>
<collision name="collision"><geometry><box><size>0.04 0.04 0.04</size></box></geometry>
<surface><friction><ode><mu>2</mu><mu2>2</mu2></ode></friction></surface></collision>
<visual name="visual"><geometry><box><size>0.04 0.04 0.04</size></box></geometry></visual>
<sensor name="contact_sensor" type="contact"><always_on>true</always_on><update_rate>100</update_rate>
<contact><collision>collision</collision></contact><plugin name="bumper" filename="libgazebo_ros_bumper.so">
<ros><remapping>bumper_states:=/mars_probe/contacts</remapping></ros><frame_name>world</frame_name>
</plugin></sensor>
</link></model></sdf>'''
        req.initial_pose.position.x,req.initial_pose.position.y,req.initial_pose.position.z=map(float,xyz)
        req.initial_pose.orientation.w=1.
        self.call(self.spawn,req)
        start=self.sim_time()
        self.wait(lambda:self.sim_time()>start+1.5)
        query=GetEntityState.Request()
        query.name=name
        query.reference_frame='world'
        result=self.call(self.get_state,query).state
        delete=DeleteEntity.Request()
        delete.name=name
        self.call(self.delete,delete)
        return [result.pose.position.x,result.pose.position.y,result.pose.position.z],sorted(set(self.contacts))

    def verify(self,scene=None,bundle=None):
        wall=time.monotonic()
        self.wait(lambda:len(self.images)>10 and self.odom is not None and self.depth is not None and self.info is not None)
        beginning=self.sim_time()
        start_xyz=[self.odom.pose.pose.position.x,self.odom.pose.pose.position.y,self.odom.pose.pose.position.z]
        # Let controller spawners finish, then command a short straight drive.
        self.wait(lambda:self.sim_time()>beginning+2.)
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
        result={'mode':'mars' if scene else 'original','camera_dimensions':[self.info.width,self.info.height],
                'camera_frame':self.info.header.frame_id,'camera_hz_sim':sim_hz,'depth_encoding':self.depth.encoding,
                'initial_rover_xyz':start_xyz,'drive_displacement_m':distance,
                'camera_topics':['/oak/image_raw','/oak/depth/image_raw','/oak/camera_info'],
                'bridges_required':False,'collision_probes':[],'errors':[]}
        if distance<.03:
            result['errors'].append('Controller drive produced insufficient displacement')
        if scene:
            field=Terrain(scene['terrain']['heights'],scene['terrain']['size_m'])
            for i,(x,y) in enumerate([(0.,-.7),(3.2,3.1),(-3.1,-3.3)]):
                xyz,contacts=self.probe(f'mars_check_ground_{i}',[x,y,float(field.height(x,y))+.25])
                residual=xyz[2]-.02-float(field.height(*xyz[:2]))
                result['collision_probes'].append({'kind':'terrain','xyz':xyz,'bottom_surface_error_m':residual,'contacts':contacts})
                if abs(residual)>.055:
                    result['errors'].append(f'Terrain probe {i} disagrees with terrain elevation')
            if scene['objects']:
                obj=max(scene['objects'],key=lambda o:o['scale'])
                assets=json.loads((bundle/'assets/catalog.json').read_text())
                asset=next(a for a in assets if a['id']==obj['asset_id'])
                v=np.array(json.loads((bundle/asset['geometry']).read_text())['vertices'])
                v=v @ rotation(obj['rpy']).T*obj['scale']+obj['xyz']
                peak=v[np.argmax(v[:,2])]
                # Drop onto a patch close to the top; a collision must keep it
                # significantly above the terrain, even with simplified geometry.
                xyz,contacts=self.probe('mars_check_rock',[peak[0],peak[1],peak[2]+.15])
                above=xyz[2]-.02-float(field.height(*xyz[:2]))
                result['collision_probes'].append({'kind':'rock','asset_id':obj['asset_id'],'xyz':xyz,'height_above_terrain_m':above,'contacts':contacts})
                if not any(obj['name']+'::' in name for name in contacts):
                    result['errors'].append('Rock probe registered no contact with the target obstacle')
        result['wall_seconds']=time.monotonic()-wall
        return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',type=Path,help='Omit to verify the original maze launch')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    scene=json.loads(args.scene.read_text()) if args.scene else None
    if scene:
        cmd=['ros2','launch','mars_training_environment','mars_world.launch.py',f'scene:={args.scene.resolve()}','gui:=false']
    else:
        # The maze launch has no gui argument and always starts its own client.
        cmd=['ros2','launch','four_wheel_sim','gazebo.launch.py']
    with args.output.with_suffix('.log').open('w') as log, simulator_process(cmd,log):
        rclpy.init()
        node=RuntimeCheck()
        try:
            report=node.verify(scene,args.scene.resolve().parents[2] if args.scene else None)
            write_json(args.output,report)
            print(json.dumps(report,indent=2))
        finally:
            node.cmd.publish(Twist())
            node.destroy_node()
            rclpy.shutdown()
    raise SystemExit(bool(report['errors']))


if __name__ == '__main__':
    main()
