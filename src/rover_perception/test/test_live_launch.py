"""Verify exclusive command ownership in the deployed ROS launch configuration."""
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from rover_perception.drive_mode import GAZEBO_DRIVE_TOPIC


class Configuration:
    def __init__(self, name):
        self.name = name

    def perform(self, context):
        return context[self.name]


class LaunchOwnershipTests(unittest.TestCase):
    def test_each_mode_launches_only_its_command_source(self):
        modules = {name: types.ModuleType(name) for name in (
            'ament_index_python', 'ament_index_python.packages', 'launch', 'launch.actions',
            'launch.launch_description_sources', 'launch.substitutions', 'launch_ros', 'launch_ros.actions')}
        modules['ament_index_python.packages'].get_package_share_directory = lambda name: '/share/' + name
        modules['launch'].LaunchDescription = list
        for name in ('DeclareLaunchArgument', 'OpaqueFunction'):
            setattr(modules['launch.actions'], name, lambda *args, **kw: (args, kw))
        modules['launch.actions'].IncludeLaunchDescription = lambda source, **kw: dict(kind='world', **kw)
        modules['launch.launch_description_sources'].PythonLaunchDescriptionSource = lambda path: path
        modules['launch.substitutions'].LaunchConfiguration = Configuration
        modules['launch_ros.actions'].Node = lambda **kw: dict(kind='node', **kw)
        root = Path(__file__).resolve().parents[3]
        source = root / 'src/mars_training_environment/launch/mars_live.launch.py'
        spec = importlib.util.spec_from_file_location('tested_mars_live', source)
        launch = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(launch)
        with tempfile.TemporaryDirectory() as folder:
            scene = Path(folder) / 'scene.json'
            scene.write_text(json.dumps({'config': {'camera': {'image_topic': '/oak/image_raw'}}}))
            context = dict(scene=str(scene), bridge_port='8770', camera_timeout='3.0', gui='true')
            for mode in ('browser', 'gazebo'):
                with self.subTest(mode=mode):
                    actions = launch.setup(dict(context, control_mode=mode))
                    camera = next(a for a in actions if a.get('executable') == 'live_bridge')
                    self.assertEqual(camera['parameters'][0]['control_mode'], mode)
                    self.assertFalse(camera['parameters'][0]['use_sim_time'])
                    self.assertEqual(dict(actions[0]['launch_arguments'])['capture'], 'false')
                    self.assertEqual(dict(actions[0]['launch_arguments'])['live_camera'], 'true')
                    native = [a for a in actions if a.get('executable') == 'gazebo_drive']
                    bridge = [a for a in actions if a.get('executable') == 'parameter_bridge']
                    self.assertEqual(len(native), int(mode == 'gazebo'))
                    self.assertEqual(len(bridge), int(mode == 'gazebo'))
                    if native:
                        self.assertFalse(native[0]['parameters'][0]['use_sim_time'])
                        self.assertEqual(bridge[0]['arguments'], [
                            GAZEBO_DRIVE_TOPIC + '@geometry_msgs/msg/Twist[gz.msgs.Twist'])
            with self.assertRaises(ValueError):
                launch.setup(dict(context, control_mode='both'))


if __name__ == '__main__':
    unittest.main()
