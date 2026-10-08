"""Apply camera-only overrides to an in-memory copy of the existing rover (Harmonic)."""
import copy
import xml.etree.ElementTree as ET

BOXES_GZ_TOPIC = 'oak_boxes'


def robot_description(config):
    import xacro
    from rover_training.randomize import default_xacro
    root = ET.fromstring(xacro.process_file(str(default_xacro()), mappings={'sim': 'harmonic'}).toxml())
    camera = config['camera']
    joint = root.find("joint[@name='camera_joint']/origin")
    joint.set('xyz',' '.join(map(str,camera['mount_xyz'])))
    joint.set('rpy',f"0 {camera['tilt_rad']} 0")
    holder = root.find("gazebo[@reference='camera_link']")
    sensor = holder.find('sensor')
    changes = {'update_rate':camera['rate_hz'],'camera/horizontal_fov':camera['horizontal_fov_rad'],
               'camera/image/width':camera['width'],'camera/image/height':camera['height'],
               'camera/clip/near':camera['near_m'],'camera/clip/far':camera['far_m'],
               'camera/depth_camera/clip/near':camera['near_m'],'camera/depth_camera/clip/far':camera['far_m']}
    for path,value in changes.items():
        sensor.find(path).text = str(value)
    # Same mount, lens and rate as the RGB-D camera. Gazebo reports a visible 2D box
    # for every model tagged by the Label system; capture compares them with our labels.
    boxes = copy.deepcopy(sensor)
    boxes.set('name','oak_boxes')
    boxes.set('type','boundingbox_camera')
    boxes.find('topic').text = BOXES_GZ_TOPIC
    lens = boxes.find('camera')
    lens.remove(lens.find('depth_camera'))
    box_type = ET.Element('box_type')
    box_type.text = '2d'
    lens.insert(0, box_type)
    holder.append(boxes)
    return ET.tostring(root,encoding='unicode')
