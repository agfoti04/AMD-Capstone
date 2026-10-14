"""Apply camera-only overrides to an in-memory copy of the existing rover (Harmonic)."""
import copy
import xml.etree.ElementTree as ET

BOXES_GZ_TOPIC = 'oak_boxes'


def live_camera_config(config, width=960):
    """Reduce live rendering cost without changing the saved capture profile."""
    if width not in (640, 960, 1280, 1920):
        raise ValueError('live_width must be 640, 960, 1280 or 1920')
    result = copy.deepcopy(config)
    camera = result['camera']
    camera['height'] = max(2, round(camera['height'] * width / camera['width'] / 2) * 2)
    camera['width'], camera['rate_hz'] = width, 30
    return result


def sync_mast(root, mount_xyz):
    """Keep the visual mast between the chassis top and the camera origin.

    mount_xyz is the camera_joint offset in base_link. It is not the optical
    center's height above the ground. The mast has no collision.
    """
    mast = root.find("link[@name='base_link']/visual[@name='camera_mast']")
    chassis = root.find("link[@name='base_link']/visual[@name='chassis']/geometry/box")
    if mast is None or chassis is None:
        return
    half_z = float(chassis.get('size').split()[2]) * 0.5
    top = float(mount_xyz[2])
    length = top - half_z
    center = (top + half_z) * 0.5
    origin = mast.find('origin')
    if origin is None:
        origin = ET.SubElement(mast, 'origin')
    origin.set('xyz', f'{float(mount_xyz[0]):.12g} {float(mount_xyz[1]):.12g} {center:.12g}')
    origin.set('rpy', '0 0 0')
    mast.find('geometry/box').set('size', f'0.02 0.02 {length:.12g}')


def robot_description(config, live=False):
    import xacro
    from rover_training.randomize import default_xacro
    root = ET.fromstring(xacro.process_file(str(default_xacro()), mappings={'sim': 'harmonic'}).toxml())
    camera = config['camera']
    joint = root.find("joint[@name='camera_joint']/origin")
    joint.set('xyz',' '.join(map(str,camera['mount_xyz'])))
    joint.set('rpy',f"0 {camera['tilt_rad']} 0")
    sync_mast(root, camera['mount_xyz'])
    holder = root.find("gazebo[@reference='camera_link']")
    sensor = holder.find('sensor')
    # One rgbd_camera image size covers RGB and depth. There is no separate
    # depth resolution to update. The bounding-box camera is a copy of this
    # sensor, so it follows the same width, height, and rate.
    changes = {'update_rate':camera['rate_hz'],'camera/horizontal_fov':camera['horizontal_fov_rad'],
               'camera/image/width':camera['width'],'camera/image/height':camera['height'],
               'camera/clip/near':camera['near_m'],'camera/clip/far':camera['far_m'],
               'camera/depth_camera/clip/near':camera['near_m'],'camera/depth_camera/clip/far':camera['far_m']}
    for path,value in changes.items():
        sensor.find(path).text = str(value)
    if live:
        # Live detection needs only RGB. Plain cameras publish on their exact
        # topic, whereas the RGB-D sensor appends /image to its base topic.
        sensor.set('type', 'camera')
        sensor.find('topic').text = '/oak/image'
        sensor.find('camera').remove(sensor.find('camera/depth_camera'))
        return ET.tostring(root, encoding='unicode')
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
