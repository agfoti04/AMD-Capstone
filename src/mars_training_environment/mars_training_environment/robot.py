"""Apply camera-only overrides to an in-memory copy of the existing rover."""
import xml.etree.ElementTree as ET


def robot_description(config):
    import xacro
    from rover_training.randomize import default_xacro
    root = ET.fromstring(xacro.process_file(str(default_xacro())).toxml())
    camera = config['camera']
    joint = root.find("joint[@name='camera_joint']/origin")
    joint.set('xyz',' '.join(map(str,camera['mount_xyz'])))
    joint.set('rpy',f"0 {camera['tilt_rad']} 0")
    sensor = root.find("gazebo[@reference='camera_link']/sensor")
    changes = {'update_rate':camera['rate_hz'],'camera/horizontal_fov':camera['horizontal_fov_rad'],
               'camera/image/width':camera['width'],'camera/image/height':camera['height'],
               'camera/clip/near':camera['near_m'],'camera/clip/far':camera['far_m'],
               'plugin/min_depth':camera['near_m'],'plugin/max_depth':camera['far_m']}
    for path,value in changes.items():
        sensor.find(path).text = str(value)
    return ET.tostring(root,encoding='unicode')
