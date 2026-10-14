"""Real rover Xacro expansion for capture and the lighter browser-driving camera."""
import copy
import sys
import types
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

from mars_training_environment.common import read_config
from mars_training_environment.robot import live_camera_config, robot_description


class LiveCameraTests(unittest.TestCase):
    def test_live_resolution_keeps_fov_and_saved_capture_config(self):
        original = read_config()
        saved = copy.deepcopy(original)
        live = live_camera_config(original)
        self.assertEqual((live['camera']['width'], live['camera']['height']), (960, 540))
        self.assertEqual(live['camera']['horizontal_fov_rad'], original['camera']['horizontal_fov_rad'])
        self.assertEqual(live['camera']['mount_xyz'], original['camera']['mount_xyz'])
        self.assertEqual(original, saved)
        self.assertEqual(live_camera_config(original, 640)['camera']['height'], 360)
        with self.assertRaises(ValueError):
            live_camera_config(original, 100)

    def test_real_robot_keeps_capture_sensors_and_uses_only_rgb_for_live(self):
        root = Path(__file__).resolve().parents[3]
        packages = types.ModuleType('ament_index_python.packages')
        packages.get_package_share_directory = lambda name: str(root / 'src' / name)
        with patch.dict(sys.modules, {'ament_index_python': types.ModuleType('ament_index_python'),
                                     'ament_index_python.packages': packages}):
            config = read_config()
            capture = ET.fromstring(robot_description(config))
            live = ET.fromstring(robot_description(live_camera_config(config), live=True))
        capture_sensors, live_sensors = capture.findall('.//sensor'), live.findall('.//sensor')
        self.assertEqual([s.get('type') for s in capture_sensors], ['rgbd_camera', 'boundingbox_camera'])
        self.assertEqual([s.get('type') for s in live_sensors], ['camera'])
        rgb = live_sensors[0]
        self.assertEqual(rgb.findtext('topic'), '/oak/image')
        self.assertEqual(rgb.findtext('camera/image/width'), '960')
        self.assertEqual(rgb.findtext('camera/image/height'), '540')
        self.assertIsNone(rgb.find('camera/depth_camera'))
        self.assertEqual(capture_sensors[0].findtext('camera/image/width'), str(config['camera']['width']))
        self.assertEqual([n.get('name') for n in capture.findall('joint')], [n.get('name') for n in live.findall('joint')])
        self.assertEqual(ET.tostring(capture.find('ros2_control')), ET.tostring(live.find('ros2_control')))


if __name__ == '__main__':
    unittest.main()
