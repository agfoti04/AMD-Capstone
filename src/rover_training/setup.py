import os
from glob import glob
from pathlib import Path

from setuptools import find_packages, setup

package_name = "rover_training"


def model_files():
    entries = []
    for path in Path("models").rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        destination = Path("share") / package_name / path.parent
        entries.append((destination.as_posix(), [path.as_posix()]))
    return entries


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    ] + model_files(),
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Ian Benge",
    maintainer_email="ianbenge37@gmail.com",
    description="Synthetic rock scenes and YOLO labels for the rover camera.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "generate_scenes = rover_training.randomize:main",
            "list_scenes = rover_training.randomize:list_main",
            "preview_labels = rover_training.preview:main",
            "capture_dataset = rover_training.capture_node:main",
        ],
    },
)
