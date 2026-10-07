# Dependencies

The simulation target is ROS 2 Humble on Ubuntu 22.04 with Gazebo Classic 11.
Ubuntu 24.04's default is ROS 2 Jazzy and Gazebo Harmonic. This repository
does not use those. The Docker image in `docker/Dockerfile` is the isolated
22.04 environment for a 24.04 host, including the Halo.

## Apt packages inside the image

- `ros-humble-ros-base` (from the official image `ros:humble-ros-base`, Ubuntu 22.04)
- `ros-humble-xacro`
- `ros-humble-robot-state-publisher`
- `ros-humble-gazebo-ros-pkgs` (Gazebo Classic 11 and `gazebo_ros`)
- `ros-humble-gazebo-ros2-control`
- `ros-humble-controller-manager`
- `ros-humble-joint-state-broadcaster`
- `ros-humble-diff-drive-controller`
- `ros-humble-topic-tools`
- `ros-humble-cv-bridge`
- `ros-humble-vision-msgs`
- `ros-humble-image-transport`
- `python3-colcon-common-extensions`
- `python3-numpy`
- `python3-opencv`
- `xvfb`, `mesa-utils`, `libgl1-mesa-dri` for headless software rendering

Optional later, only if you launch `rtab_demo.py`: `ros-humble-rtabmap-launch`,
`ros-humble-rviz2`. The physical camera launch in `custom_launch/` also needs
`depthai_ros_driver_v3`, which is not in this image.

`four_wheel_sim/package.xml` originally omitted the exec dependencies its
launch file imports. Those names were added. The launch files themselves were
not edited.

## Not installed

Do not install PyTorch, Ultralytics, or a CUDA wheel on this laptop. The GPU
seen here is an AMD Radeon with no NVIDIA driver. The Halo is an AMD Ryzen AI
machine. On that computer, check the AMD ROCm or Ryzen AI stack before
installing a training framework. The detector runs with `backend: none` until
a weights file exists.

## Camera topics

Humble `libgazebo_ros_camera` publishes `<camera_name>/image_raw`,
`<camera_name>/camera_info`, and, for a depth sensor,
`<camera_name>/depth/image_raw`, `<camera_name>/depth/camera_info`, and
`<camera_name>/points`. The rover sets `camera_name` to `oak` and does not set
a ROS namespace, so the expected simulator names are:

- `/oak/image_raw`
- `/oak/camera_info`
- `/oak/depth/image_raw`

`rtab_demo.py` subscribes to those three. Depth is `32FC1` meters along the
optical axis. Values outside the sensor near/far range (0.3 m to 8.0 m in the
xacro) are stored as infinities.

The physical DepthAI launch in `custom_launch/rtab_script.launch.py` uses
different names:

- `/oak/rgb/image_raw`
- `/oak/rgb/camera_info`
- `/oak/stereo/image_raw`

`rover_perception/config/sim.yaml` and `oak.yaml` select which image the
detector reads. Confirm the live names with `ros2 topic list` before a long
capture. This laptop could not start Gazebo, so those names are from the
Humble plugin source plus the launch files, not from a running simulator.

## Assets

See `src/rover_training/models/ATTRIBUTION.md` and `assets.json`.
