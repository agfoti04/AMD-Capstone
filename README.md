# Rover perception workspace

This folder is a separate copy of the rover simulation. The original archive
is unchanged. `main` is a local baseline commit of that simulation. Branches,
each built on the one before it:

| Branch | Adds |
| --- | --- |
| `perception/rock-dataset` | Rock scenes, YOLO labels, and a detector in Gazebo Classic |
| `perception/mars-training-environment` | The Martian terrain and the 200-image capture tools |
| `perception/gz-harmonic-migration` | The rover and the Mars environment in Gazebo Harmonic (ROS 2 Jazzy) |
| `perception/mars-sampling-sites` | A 1080p camera 30 in above the ground, and sampling-site analogues |

The sections below cover the original Humble and Gazebo Classic workflow. The
Mars environment runs in Jazzy and Harmonic. Its short version is in
[Mars environment in Gazebo Harmonic](#mars-environment-in-gazebo-harmonic),
and the full commands are in `src/mars_training_environment/README.md`.

The rover description, maze launch, and controllers are the original files.
`four_wheel_sim/launch/gazebo.launch.py` does not take a world argument: it
always writes the maze and passes that file into Gazebo. Rock worlds live in
`rover_training` and start `gzserver` directly so the maze launch stays as it
was.

Commands below run inside the container, unless a step says PowerShell.

## One-time container setup

Start Docker Desktop and wait until it says the engine is running. In
PowerShell, from this folder (`rover_ws`):

```powershell
docker build -t rover-humble -f docker/Dockerfile .
docker run --rm -it -v ${PWD}:/work -w /work rover-humble bash
```

The image is several gigabytes. If the build fails for lack of disk space,
free space on C: or build the image on the Halo instead.

Inside the container, the first line loads Humble. The second compiles only
the source packages and ignores any old `build/`, `install/`, or `log/`
directory from the zip. The third line lets you run the packages you just
built.

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select four_wheel_sim rover_training rover_perception
source install/setup.bash
```

Run those two `source` lines again in every new terminal. `install/setup.bash`
also loads Humble if you sourced Humble before building.

## Original maze

This opens Gazebo, spawns the rover, and starts the wheel controllers. It
needs a display. On the Halo desktop, run it as-is. In the laptop container
there is no window, so try `gui:=false` first.

```bash
ros2 launch four_wheel_sim gazebo.launch.py gui:=false
```

The maze file is written to `/tmp/four_wheel_maze.world`. In the Gazebo log,
that path should appear. If the log loads `empty.world` instead, say so before
changing the maze launch: the world path is handed to `gazebo.launch.py`, and
an older Gazebo ROS package might not pass it through.

In a second container terminal (`docker exec -it <container> bash`, then the
same two `source` lines), check the camera. The xacro names the sensor `oak`
and the Humble camera plugin publishes these topics when the namespace is `/`:

```bash
ros2 topic list
ros2 topic hz /oak/image_raw
ros2 topic echo /oak/camera_info --once
ros2 topic hz /oak/depth/image_raw
```

You want a camera info message whose width is 1920 and height is 1080, and a
simulation-time image rate near 30 Hz. Wall-clock rate is lower when the
container is software-rendering. The horizontal field of view, 1.22 rad, is
provisional until an OAK-D calibration is measured. If the topic names differ, do not edit the rover xacro. Pass the
real names to the capture node (`image_topic`, `depth_topic`,
`camera_info_topic`) and edit `src/rover_perception/config/sim.yaml` to match,
then rebuild.

`ros2 launch four_wheel_sim rtab_demo.py` also starts RTAB-Map and expects
those three topics. It opens RViz with `config/demo.rviz`, which is not in
the package. Skip RViz with:

```bash
ros2 launch four_wheel_sim rtab_demo.py rviz:=false
```

`sim.launch.py` is a separate, older launch. It reads `urdf/robot.urdf` and
starts `kinematic_sim`, and `setup.py` points at `my_teleop`. Those files are
not in the source tree (only old `.pyc` files were in the archive). Use
`gazebo.launch.py`, not `sim.launch.py`.

## Three rocks

`scene_three` places three training rocks in front of the rover and leaves
the maze launch alone.

```bash
export LIBGL_ALWAYS_SOFTWARE=1
ros2 launch rover_training rocks_world.launch.py gui:=false scene_id:=scene_three
```

Other built-in scenes: `scene_empty` (no rocks), `scene_clip` (a rock on the
image border), `scene_aligned` (one rock behind another), `scene_holdout`
(only `falling_rock4`, which is reserved for test images).

With a display, drop `gui:=false` or pass `gui:=true`.

## Capture and preview

One scene, while you watch the logs:

```bash
export LIBGL_ALWAYS_SOFTWARE=1
xvfb-run -a ros2 launch rover_training capture_scene.launch.py \
  gui:=false scene_id:=scene_three output_dir:=/work/dataset/rocks
```

That starts Gazebo, moves the rover through the scene's viewpoints, writes
images and labels, and then shuts Gazebo down. `xvfb-run` provides an
offscreen display so the Classic camera can render without a monitor.
`LIBGL_ALWAYS_SOFTWARE=1` forces Mesa's software renderer. It is slow. It is
the right setting in this container because the laptop GPU is not passed
through.

Every scene for seed 7, then a contact sheet of the boxes:

```bash
bash scripts/capture_plan.sh 7 /work/dataset/rocks
```

Or draw boxes on a dataset you already saved:

```bash
ros2 run rover_training preview_labels -- --dataset /work/dataset/rocks
```

Look in `dataset/rocks/previews/`. Check four things before generating a large
set: boxes sit on the rocks, a rock cut by the image border is clipped to the
frame, a rock hidden behind another is dropped or kept according to the depth
buffer, and `scene_empty` has an empty `.txt` file (a valid YOLO background).

Layout:

```text
dataset/rocks/images/<train|val|test>/
dataset/rocks/labels/<train|val|test>/
dataset/rocks/calibration/
dataset/rocks/depth/          # optional 32-bit depth, PFM
dataset/rocks/previews/
dataset/rocks/data.yaml       # class 0 is rock
dataset/rocks/manifest.jsonl
```

`dataset/` is gitignored. Neighboring viewpoints of one scene share a split.
`falling_rock4` appears only in test scenes, never next to a training rock.

The label is the tightest image box around the rock's mesh vertices, projected
with the camera calibration and the simulator pose. If part of the rock is
behind the camera, the corners of its 3D box are used instead. A depth check measures how much of the rock is visible.
Rocks less than 30% visible are not labeled (`min_visible_fraction`), so
fully hidden rocks and slivers behind other rocks are skipped. Detector output is not used
as a label.

## Detector

Simulator:

```bash
ros2 launch rover_perception detector.launch.py
```

Physical OAK-D, after the DepthAI driver is publishing:

```bash
ros2 launch rover_perception detector.launch.py \
  params_file:=$(ros2 pkg prefix --share rover_perception)/config/oak.yaml
```

The node subscribes only to the image topic in that file (`/oak/image_raw` in
simulation, `/oak/rgb/image_raw` on the camera). It publishes
`/detections` as a `vision_msgs/Detection2DArray`. With `backend: none` the
array is empty. That is expected until a weights file exists. Do not point
this node at `/gazebo/model_states` or the label files.

## Tests that do not need Gazebo

On the Windows laptop, from `rover_ws`:

```powershell
$env:PYTHONPATH = "src\rover_training"
python -m unittest discover -s src\rover_training\test -v
```

These tests check projection, clipping, occlusion, empty labels, class id 0,
the scene split, and that the maze launch still has no world argument. They
do not start Gazebo.

The Mars environment tests also run without Gazebo:

```powershell
$env:PYTHONPATH = "src\mars_training_environment;src\rover_training"
python src\mars_training_environment\test\test_environment.py
```

## Mars environment in Gazebo Harmonic

The Mars worlds use the Jazzy image (`rover-jazzy`), not `rover-humble`. Build
it once from this folder in PowerShell:

```powershell
docker build -f docker/Dockerfile.jazzy -t rover-jazzy .
```

To see the Gazebo window, start the container with the WSLg display mounts.
Without them Gazebo exits with `could not connect to display`. Leave the
mounts off for headless capture.

```powershell
docker run --rm -it -e LIBGL_ALWAYS_SOFTWARE=1 -v "${PWD}:/work" -w /work -v /run/desktop/mnt/host/wslg/.X11-unix:/tmp/.X11-unix -v /run/desktop/mnt/host/wslg:/mnt/wslg -e DISPLAY=:0 -e WAYLAND_DISPLAY=wayland-0 -e XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir rover-jazzy bash
```

Inside the container, build into the Jazzy folders, generate one scene, and
open it:

```bash
source /opt/ros/jazzy/setup.bash
cd /work
colcon --log-base log_jazzy build --build-base build_jazzy --install-base install_jazzy \
  --packages-select four_wheel_sim rover_training mars_training_environment
source /work/install_jazzy/setup.bash
ros2 run mars_training_environment mars_generate \
  --config /work/src/mars_training_environment/config/sample.json \
  --output /work/dataset/mars_sites_scenes --seed 101 --asset-cache /work/assets_cache
ros2 launch mars_training_environment mars_world.launch.py \
  scene:=/work/dataset/mars_sites_scenes/scenes/scene_000101/scene.json gui:=true
```

Rock and ground scans come from `mars_assets` into `assets_cache/`. See the
package README if that folder is empty.

For a labeled 10-image set from that scene, without the window:

```bash
ros2 run mars_training_environment mars_dataset \
  --bundle /work/dataset/mars_sites_scenes --output /work/dataset/mars_sites_10 --scene scene_000101
```

Saving `sample.json` does not change a running world. Generate into a new
output folder and relaunch. The generator refuses to reuse a folder made with
another config.

### Camera

The Harmonic camera captures 1920 × 1080 RGB at 30 Hz in simulation time.
Its optical center is 0.762 m (30 in) above level ground. `base_link` rests
0.08 m off the ground, so the camera joint is 0.682 m above it, with a 0.1 rad
nose-down pitch. The 1.22 rad horizontal field of view is provisional until
the OAK-D is calibrated. `mars_verify_runtime` checks the size, rate,
intrinsics, and height in a running world.

### Sampling-site analogues

`candidates` in `sample.json` adds three targets on top of the sand and
ordinary rocks. Each has a lookalike control, so a detector cannot pick sites
by brightness alone.

| Target | Lookalike |
| --- | --- |
| `silica_soil`: an irregular pale soil patch with embedded stones | `pale_lookalike`: a smoother pale patch |
| `sulfur_rock`: a pale stone with yellow crystals on its broken face | `pale_stone`: an intact pale stone |
| `cheyava_rock`: mudstone with pale veins and dark-rimmed spots | `veined_lookalike`: veins, no spots |

They are built from the NASA photos in `Mars_Gazebo_Reference_Pack` but are
not copies of them. Category names are assigned ground truth, not chemical
measurements, and a camera cannot confirm composition. Cheyava spots are a few
millimetres, so at 1080p they are only a few pixels past about 1 m.
`exaggerated_spots` enlarges them for demos and tags those rocks.

Labels have seven classes: `rock` plus the six above. Silica patches are also
saved as `candidate_mask.png` in each scene folder. Seeds 100, 110, and 117
stay empty.

## Copying to the Halo

Copy this git repository, not `build/`, `install/`, `log/`, or `dataset/`.
On the Halo, install Docker or use an existing engine, then run the same
`docker build` and `colcon build` commands. If the Halo already has a
display, you can omit `xvfb-run` and set `gui:=true` when you want the window.

Before any training install, run the vendor GPU check that matches the Halo
image (`rocminfo`, or the Ryzen AI software stack). This laptop cannot tell
you which of those is installed on the Halo.

## Known gaps

- The Classic maze was not started here, so its camera topic names are not
  confirmed live. The Harmonic camera was checked with `mars_verify_runtime`.
- Sampling-site scene 101 has a validated 10-image capture after the terrain
  placement fix. The full 200-image sampling-site dataset has not been captured.
- `sim.launch.py`, `my_teleop`, and `demo.rviz` are missing from the archive.
- NASA meshes were not available as direct downloads. Fuel rocks are used
  instead. See `src/rover_training/models/ATTRIBUTION.md`.
- No model has been trained.
