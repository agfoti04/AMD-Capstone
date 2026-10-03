from setuptools import find_packages, setup
import os
from glob import glob

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
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='x',
    description='Sim testing package for basic control theory',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'base_controller = four_wheel_sim.base_controller:main',
            'my_teleop = four_wheel_sim.my_teleop:main',
            'path_runner = four_wheel_sim.path_runner:main',
        ],
    },
)
