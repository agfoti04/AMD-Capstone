from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'four_wheel_sim'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'config'), glob('config/*.rviz')),
        (os.path.join('share', package_name, 'config', 'bt'), glob('config/bt/*.xml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='AMD Capstone Team',
    maintainer_email='agfoti04@tamu.edu',
    description='Four-wheel rover simulation, mapping, and navigation',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'my_teleop = four_wheel_sim.my_teleop:main',
            'maze_waypoints = four_wheel_sim.maze_waypoints:main',
            'odom_gt = four_wheel_sim.odom_gt:main',
            'cmd_filter = four_wheel_sim.cmd_filter:main',
            'head_scanner = four_wheel_sim.camera_control:main',
            'rover_mission = four_wheel_sim.rover_mission:main',
        ],
    },
)
