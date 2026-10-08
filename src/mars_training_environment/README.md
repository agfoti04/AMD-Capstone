# Martian training environment

Additive world and dataset tools for the existing Humble / Gazebo Classic rover.
`four_wheel_sim` is not edited. The maze launch still builds the maze, and it
still has no world argument. This package starts `gzserver` itself and spawns
the same rover, controllers, `/cmd_vel` relay, and `/odom` ground-truth relay.

The default world is 20 m by 20 m. Size, relief, rock counts, and camera
settings come from `config/sample.json`. Earth gravity (`0 0 -9.81`) stays on,
because the physical rover is tested on Earth.

## Terrain and rocks

The default terrain backend is `mesh`, so the picture, the collision, and the
labeler share one triangle mesh (`scenes/<id>/terrain.dae`). Gazebo Classic
can load a heightmap instead (`"backend": "heightmap"`). That path writes the
same image, size, and position into the visual and the collision. Classic has
a history of mirroring those two heightmaps, so the sample config uses the mesh.

The ground texture is a seeded mix of dusty red, tan, and gray. It is copied
next to the Ogre material script so Gazebo finds it. The height field has
broad slopes, ridges, and shallow bowls, then a flat disk at the origin so
the rover still spawns at `z = 0.09`. Slope is limited to `max_slope_deg`
(10 degrees in the sample).

Rock shapes:

- 16 procedural meshes (`proc_00` through `proc_15`), generated in this
  package, Apache-2.0.
- 4 Open Robotics Fuel models already stored in `rover_training`
  (`falling_rock1` through `falling_rock4`), CC-BY-4.0. See
  `src/rover_training/models/ATTRIBUTION.md`.

`proc_12`, `proc_13`, `proc_14`, `proc_15`, and `falling_rock4` are reserved
for test scenes. A train or val scene never instances them, and a test scene
never instances the other shapes. Each mesh is a few hundred triangles, and
collision uses those same triangles.

Sample rocks are 0.07–0.20 m on the longest side. Obstacle rocks are
0.25–0.70 m. Those ranges are placeholders until the arm and gripper limits
are known; change `sampling_size_m` and `obstacle_size_m`. Burial is
0–24% of the rock's height (`burial_fraction`). Rocks are static. Soil
deformation and grasping are not modeled.

## Labels

Classic 11 has no bounding-box camera. Labels are not detector output. A CPU
rasterizer projects the same visual triangles the world loads, including the
terrain, and keeps the visible pixels.

| Case | What is written |
| --- | --- |
| Fully outside the image, or only a sub-pixel fragment | No box |
| Fully hidden by terrain or another rock | No box |
| Visible fraction below `min_visible_fraction` (0.30) | No box |
| Visible area below `min_visible_pixels` (24) or box side below `min_box_side_px` (5) | No box |
| Partly in frame | Box clipped to the image |
| Empty scene | Empty `.txt` file, which is a valid YOLO label |

Class id `0` is `rock`. Ground texture is background. Clean images are written
under `images/`. Boxes are drawn only under `previews/`.

Each accepted frame must also match the live depth image. The capture refuses
the frame when the median depth error on rock pixels is above 8 cm. That
check uses `/oak/depth/image_raw`. Depth files are saved only when
`save_depth` is true.

The rover body is not in the rasterizer. The camera looks forward from the
front of the body, so the body rarely covers a labeled rock. This is still a
limitation for rocks directly against the chassis.

## Camera values are not OAK-D calibration

`config/sample.json` copies the camera already in `robot.urdf.xacro`. Do not
treat these as measured intrinsics.

| Setting | Sample value | Replace with |
| --- | --- | --- |
| `mount_xyz` | `0.2 0 0.12` m from `base_link` | Measured mount |
| `tilt_rad` | `0.1` rad nose-down | Measured tilt |
| `width`, `height` | 640 x 400 | OAK-D RGB mode |
| `horizontal_fov_rad` | 1.22 rad | Value from `camera_info`, not a guess |
| `near_m`, `far_m` | 0.3 m, 8.0 m | The mode you actually run |
| `rate_hz` | 10 | The capture rate you want |

The launch applies those overrides to an in-memory copy of the rover. The
xacro file on disk is unchanged. Capture reads `/oak/camera_info` for the
actual `K` matrix, so a later calibration change is picked up from the sensor
message rather than from this file.

`/odom` in this stack is the ground-truth relay (`odom_gt` republishes the
Gazebo `libgazebo_ros_p3d` pose). It is not wheel odometry. Labels use that
pose, interpolated to the image timestamp.

## Splits

Every frame from one seed stays in one split. The split is a hash of the seed,
so adding seeds does not move an existing scene. The sample seeds 100–119
produce 14 train, 2 val, and 4 test scenes, 10 views each, 200 images.
Seeds 100, 110, and 117 are empty. Seed 100 is an empty test scene.

`dataset/mars_sample/data.yaml` points Ultralytics at `images/train`,
`images/val`, and `images/test` with `0: rock`. Nothing in this package starts
a training run.

## Commands

Run these inside the `rover-humble` container, from `/work`, after Docker
Desktop is up. From PowerShell in `rover_ws`, a headless container is:

```powershell
docker run --rm -it -v ${PWD}:/work -w /work rover-humble bash
```

To open the Gazebo window, use the display mounts instead:

```powershell
docker run --rm -it -v ${PWD}:/work -w /work -v /run/desktop/mnt/host/wslg/.X11-unix:/tmp/.X11-unix -v /run/desktop/mnt/host/wslg:/mnt/wslg -e DISPLAY=:0 -e WAYLAND_DISPLAY=wayland-0 -e XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir rover-humble bash
```

Only one Gazebo can run in that container. Stop the other with Ctrl+C before
launching again.

```bash
source /opt/ros/humble/setup.bash
cd /work
colcon build --packages-select mars_training_environment
source install/setup.bash
export LIBGL_ALWAYS_SOFTWARE=1
```

Unit tests do not need Gazebo:

```bash
python3 -m unittest discover -s src/mars_training_environment/test
```

Generate the 20 sample scenes (200 planned images):

```bash
ros2 run mars_training_environment mars_generate -- --output /work/dataset/mars_scenes
```

The same config and seed rewrite the same scene. A different config in an
existing output directory is rejected. `scene.json` stores the seed, config
hash, asset ids, and poses.

Look at one world:

```bash
ros2 launch mars_training_environment mars_world.launch.py \
  scene:=/work/dataset/mars_scenes/scenes/scene_000101/scene.json gui:=true
```

Capture the sample dataset. This starts an offscreen Gazebo per scene, writes
clean images and labels, and validates the set:

```bash
ros2 run mars_training_environment mars_dataset -- \
  --bundle /work/dataset/mars_scenes --output /work/dataset/mars_sample
```

One scene is enough for a smoke test:

```bash
ros2 run mars_training_environment mars_dataset -- \
  --bundle /work/dataset/mars_scenes \
  --output /work/dataset/mars_sample --scene scene_000101
```

Resume after a stopped run:

```bash
ros2 run mars_training_environment mars_dataset -- \
  --bundle /work/dataset/mars_scenes \
  --output /work/dataset/mars_sample --resume
```

Check labels again, and rebuild the contact sheet:

```bash
ros2 run mars_training_environment mars_validate -- \
  --dataset /work/dataset/mars_sample --bundle /work/dataset/mars_scenes
```

Previews with boxes are `dataset/mars_sample/previews/<split>/`. The clean
training images are `dataset/mars_sample/images/<split>/`.

Collision and the original maze launch:

```bash
ros2 run mars_training_environment mars_verify_runtime -- \
  --output /work/dataset/mars_sample/reports/original_runtime.json
ros2 run mars_training_environment mars_verify_runtime -- \
  --scene /work/dataset/mars_scenes/scenes/scene_000101/scene.json \
  --output /work/dataset/mars_sample/reports/mars_runtime.json
```

The original command is `ros2 launch four_wheel_sim gazebo.launch.py`. It
always starts the maze and a Gazebo client, and it does not accept `gui:=`.

A larger set is the same tools with a different seed list. Copy
`config/sample.json`, append seeds, and keep the held-out mesh ids. Then:

```bash
ros2 run mars_training_environment mars_generate -- \
  --config /work/my_mars.json --output /work/dataset/mars_large_scenes
ros2 run mars_training_environment mars_dataset -- \
  --bundle /work/dataset/mars_large_scenes --output /work/dataset/mars_large
```

Read `validation.json` before growing the set. It records images per wall
second, annotation time, and the simulation real-time factor. Scene generation
is about one second per scene. Labeling one frame is under a second. A
headless scene of 10 frames took about 80 seconds on this laptop with
software rendering, at a real-time factor near 0.8, so the 200-image sample
is about half an hour. The median rock-depth error on that check was 1.5 mm.
`dataset/` is gitignored; do not commit images or generated worlds.

Viewpoints are placed so the lens, which sits 0.2 m ahead of `base_link`,
stays outside the 0.3 m near plane of every rock. A rock closer than that is
clipped by the camera and shows a hole.

## Halo Strix

Keep Humble and Gazebo Classic. Do not install Jazzy or Gazebo Harmonic on the
24.04 host. Build `docker/Dockerfile` there and run the same commands. The
image already contains `xvfb`, `libgl1-mesa-dri`, and `LIBGL_ALWAYS_SOFTWARE`
is the safe default when the container cannot see a GPU.

If `/dev/dri` is passed into the container and `glxinfo` shows a non-software
renderer, unset `LIBGL_ALWAYS_SOFTWARE` and compare one scene's
`real_time_factor` in `reports/scene_*.json` before changing the sample size.
Classic's renderer is OpenGL. A ROCm or PyTorch install does not speed up
Gazebo, and this package does not start training.

## What has been run

Unit tests: 18 passed, without Gazebo.

Inside `rover-humble`, scene 101 was captured with the mesh terrain. The ground
is reddish rather than flat green, clean images have no boxes, and the median
difference between the depth image and the label mesh was about 1.5 mm.
`dataset/mars_sample` currently has 40 frames (scenes 100–103), including the
empty test scene. Finish the other 16 scenes with `--resume` after Docker has
free disk space. The software renderer sometimes stops publishing the camera
while physics keeps going; the capture retries each scene up to three times.

The original maze launch published `/oak/image_raw` at 640×400 and
`/oak/depth/image_raw` as `32FC1`. An automated one-second drive in that same
check barely moved, so do not treat that run as proof of the controllers.
An earlier probe on this terrain rested on the mesh within a millimeter and
did contact a rock; a later probe timed out after the collision meshes were
made identical to the visual meshes, which are heavier.

## Limitations

- Camera numbers above are the current simulated camera, not a bench calibration.
- Rock sizes are not yet tied to a real gripper.
- Rocks do not move, and the ground does not deform.
- Labels ignore chassis occlusion.
- `backend: heightmap` is implemented and tested for matching visual/collision
  XML, but it is not the default.
- The `odom_gt` process can print `rcl_shutdown already called` while the
  launch is exiting. The capture has already finished when that appears.
- Two Gazebo processes in one container fail with `gzserver` exit 255.
- The capture keeps an image only when RGB, camera info, depth, and
  ground-truth odometry line up. Under software rendering the odometry
  bracket is at least 0.2 s, even if `max_pose_gap_sec` is smaller.
  `--resume` replaces a scene that stopped halfway.
