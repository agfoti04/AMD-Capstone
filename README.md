# AMD Capstone Rover

ROS 2 workspace for a four-wheel rover prototype. The project includes a
Gazebo Classic simulation, robot description and controllers, several test
worlds, RTAB-Map mapping/localization, and rover navigation/mission nodes.
There is also a separate launch file for using an OAK camera with a
host-computed RTAB-Map visual odometry pipeline.

The ROS package in this workspace is `four_wheel_sim`.

## Requirements

The current launch files target **ROS 2 Humble on Ubuntu 22.04** and use
Gazebo Classic. Install ROS 2 Humble and initialize `rosdep` using the
[official ROS 2 installation instructions](https://docs.ros.org/en/humble/Installation.html)
before installing the project dependencies.

### ROS 2 and system packages

```bash
sudo apt update
sudo apt install -y \
  python3-colcon-common-extensions \
  python3-rosdep \
  python3-numpy \
  ros-humble-gazebo-ros-pkgs \
  ros-humble-robot-state-publisher \
  ros-humble-rviz2 \
  ros-humble-xacro \
  ros-humble-ros2-control \
  ros-humble-ros2-controllers \
  ros-humble-navigation2 \
  ros-humble-nav2-bringup \
  ros-humble-nav2-simple-commander \
  ros-humble-rtabmap-ros \
  ros-humble-robot-localization \
  ros-humble-ament-copyright \
  ros-humble-ament-flake8 \
  ros-humble-ament-pep257 \
  python3-pytest
```

The ROS 2 Humble desktop/base installation supplies the core ROS client
libraries, messages, TF, and launch system used by the package. `python3-numpy`
is used by the depth-image rover mission.

If `rosdep` has not been initialized on this machine yet:

```bash
sudo rosdep init
rosdep update
```

### Physical OAK camera

The hardware launch at `custom_launch/rtab_script.launch.py` additionally
expects the `depthai_ros_driver_v3` ROS package to be installed and discoverable
in the sourced ROS environment. It is not part of this repository; install and
build the compatible driver separately according to the driver's instructions.
The launch expects a driver parameter YAML file with
`/oak/pipeline_gen/i_enable_vio: false`, because RTAB-Map computes RGB-D
odometry on the host instead.

## Build the workspace

Open a new terminal for each ROS session and source both the ROS installation
and this workspace:

```bash
cd /path/to/AMD-Capstone
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

After rebuilding in the same terminal:

```bash
source install/setup.bash
```

`rosdep` installs dependencies declared in package manifests. The explicit
package list above is included because some launch/runtime dependencies used
by this project are not currently declared in `package.xml`.

## Run the simulation

Start Gazebo Classic with the default `survey` world:

```bash
ros2 launch four_wheel_sim gazebo.launch.py
```

Choose another built-in world by setting `SIM_WORLD` before launch:

```bash
SIM_WORLD=maze ros2 launch four_wheel_sim gazebo.launch.py
SIM_WORLD=field ros2 launch four_wheel_sim gazebo.launch.py
SIM_WORLD=survey ros2 launch four_wheel_sim gazebo.launch.py
```

Start the integrated simulation, RTAB-Map mapping, and Nav2 stack:

```bash
ros2 launch four_wheel_sim slam.launch.py
```

The `viz` launch argument controls RTAB-Map visualization:

```bash
ros2 launch four_wheel_sim slam.launch.py viz:=false
```

For mapping without the Nav2 stack, run the simulation and RTAB-Map demo:

```bash
ros2 launch four_wheel_sim rtab_demo.py
```

The rover mission launch starts the simulated rover, mapping/localization, and
the configured rover mission:

```bash
ros2 launch four_wheel_sim rover.launch.py
```

The standalone localization launch is intended for a live OAK camera publishing
the configured RGB and depth topics:

```bash
ros2 launch four_wheel_sim localization.launch.py
```

These launch files use simulated time. To stop a launch, press `Ctrl+C` in its
terminal.

## Run individual nodes

The following console scripts are registered by `four_wheel_sim`:

```bash
ros2 run four_wheel_sim rover_mission
ros2 run four_wheel_sim maze_waypoints
ros2 run four_wheel_sim cmd_filter
ros2 run four_wheel_sim odom_gt
```

`maze_waypoints` expects Nav2 to already be running. `rover_mission` expects a
depth image on `/oak/depth/image_raw` and a usable transform from its configured
world frame to `base_link`; it is normally started as part of `rover.launch.py`.
The filter and odometry relay are normally started by the Gazebo launch and do
not need to be run separately.

## OAK camera and host RTAB-Map

With the separately installed `depthai_ros_driver_v3`, a driver parameter YAML
file, and the ROS workspace sourced, launch the camera and host RGB-D mapping:

```bash
cd /path/to/AMD-Capstone
ros2 launch "$PWD/custom_launch/rtab_script.launch.py" \
  params_file:=/path/to/oak_driver_params.yaml \
  name:=oak \
  frame_id:=oak_parent_frame
```

Set `params_file` to the parameter YAML used by the camera driver and ensure
that its `i_enable_vio` setting is `false`. The `frame_id` argument must match
the base frame published by the driver. The launch starts the camera driver,
`rtabmap_odom/rgbd_odometry`, RTAB-Map SLAM, and RTAB-Map visualization.

## Useful ROS 2 commands

Run these in a terminal where `/opt/ros/humble/setup.bash` and
`install/setup.bash` have been sourced:

```bash
# List ROS packages, nodes, and launch files
ros2 pkg list
ros2 pkg executables four_wheel_sim
ros2 launch four_wheel_sim <launch-file>

# Inspect active nodes and topics
ros2 node list
ros2 topic list
ros2 topic echo /odom
ros2 topic echo /map
ros2 topic echo /cmd_vel

# Send a simple velocity command (stop with Ctrl+C)
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  '{linear: {x: 0.1}, angular: {z: 0.0}}'

# Run package tests and display their results
colcon test --packages-select four_wheel_sim
colcon test-result --verbose
```

Use `ros2 topic list` first when working with a different camera or launch
configuration; topic names can differ between the simulated and hardware
setups.

## Project layout

```text
src/four_wheel_sim/
  config/       Controller, EKF, Nav2, and world-generation configuration
  four_wheel_sim/ ROS 2 nodes and world definitions
  launch/       Gazebo, mapping, localization, and mission launch files
  urdf/         Xacro robot description
custom_launch/  Hardware-oriented OAK/RTAB-Map launch
testing/       Separate projector/device experiments and Python requirements
```

The simulation world is generated at launch from the selected world definition.
`gazebo.launch.py` defaults to `survey`; `maze`, `field`, and `survey` are
implemented in `four_wheel_sim`.

## Configuration note

`config/nav2_params.yaml` currently refers to the custom behavior tree using an
absolute path under `/root/Capstone/AMD-Capstone`. If your checkout is in a
different location and you use `slam.launch.py`, update
`default_nav_to_pose_bt_xml` in that file to the local path of
`src/four_wheel_sim/config/bt/nav_bt.xml` before launching.
