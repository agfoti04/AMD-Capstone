# Martian training environment

World and dataset tools for the rover in ROS 2 Jazzy and Gazebo Harmonic.
The Gazebo Classic version of this package is on the
`perception/mars-training-environment` branch and is not maintained here.
`four_wheel_sim` keeps both simulators; this package uses its Harmonic path
(`robot.urdf.xacro` with `sim:=harmonic`, `gz_ros2_control`, `twist_stamper`,
and the `odom_gt` relay).

The default world is 20 m by 20 m. Size, relief, rock counts, lighting, and
camera settings come from `config/sample.json`. Earth gravity (`0 0 -9.81`)
stays on, because the physical rover is tested on Earth.

## Look: scanned rocks and ground

Rocks and ground textures are photogrammetry scans from
[Poly Haven](https://polyhaven.com), all CC0. They are downloaded by
`mars_assets` into a local cache and are never committed. The exact files
and their MD5 checksums are pinned in `config/assets.lock.json`, so every
machine builds the same worlds.

NASA's free 3D library was checked first. It has spacecraft and kilometre-scale
terrain, but no individual rock models that fit a rover camera, so none are
used.

Rocks (`config/assets.json`): `moon_rock_01` to `moon_rock_07`, `rock_07`,
`rock_09`, and `boulder_01`. Headless Blender reduces each scan to about
2,500 triangles for the picture and labels, and 400 for collision, keeping
UVs, and copies its albedo, normal, and roughness maps. Preparation refuses
any scan that is several separate stones (the Namaqualand boulder scans, and
`stone_01`), because one asset must be one labeled rock.
`namaqualand_boulder_03` looks like one light boulder in its preview, but the
mesh falls apart into fragments once it is simplified, so it is not used.
`boulder_01` and `rock_07` are the lighter stones in the set.

When a scene is generated, each scanned rock is recoated with that scene's
dust colour. Upward faces take more dust than undersides, and the material
colour is left white so the coat is not multiplied back down to black. The
dark lunar scans are what made the rocks look like coal.

`moon_rock_06` and `rock_09` are held out for test scenes. A train or val
scene never instances them, and a test scene never instances the others.

Ground: `red_sand` and `sand_03`, both fine sand with no gravel. Gravel,
cracked earth, and pebble photos were removed, because at 12 cm camera height
those stones are painted on and the detector learns them as rocks. Each scene
blends the two sands across the whole 20 m patch, then pulls the result toward
one Mars hue and multiplies by a slower brightness change several metres
across, so the sand's own 2–3 m repeat is not the largest pattern in the frame.
Only color variation along the Mars hue survives, so moss or other off-palette
tones in a scan do not show.

The driveable surface stays the coarse height field: 0.8 m of relief, slopes
capped at 10°, and a flat spawn pad. The picture and the labels use a finer
mesh (`visual_grid_size` 129, about 16 cm) with wind ripples a couple of
centimetres high and low ridges. Collision stays on the coarse mesh, so the
simulator does not pay for the ripples. Ripples are zero on the spawn pad.

The world uses physically based materials (`<pbr>`), the `ogre2` renderer, a
dusty butterscotch sky, a directional sun at a random azimuth and elevation,
and ambient light between 0.32 and 0.50 so shadowed rock faces stay readable.

Sample rocks are 0.02–0.12 m on the longest side. Obstacle rocks are
0.15–0.45 m. Those ranges are placeholders until the arm and gripper limits
are known; change `sampling_size_m` and `obstacle_size_m`. Burial is
22–44% of the rock's height. Rocks are static. Soil deformation and grasping
are not modeled.

## Sampling-site appearances

`candidates` in `config/sample.json` adds three visual analogues on top of the
sand, ripples, and dust. They are not copies of the NASA photographs in
`Mars_Gazebo_Reference_Pack`. Those photos are appearance references. Patch
sizes, crystal grain, and spot diameters are simulation choices, stored on
each scene record. Category names are assigned ground truth. They are not
chemical measurements, and they are not written into the RGB filenames.

| Category | What is generated | Lookalike control |
| --- | --- | --- |
| `silica_soil` | Irregular pale soil blended into the ground, with a broken edge and small embedded stones | `pale_lookalike`, a pale patch of similar brightness and a smoother surface |
| `sulfur_rock` | A pale stone with yellow crystals only on one fracture face | `pale_stone`, an intact pale stone with no yellow |
| `cheyava_rock` | A mudstone slab with pale veins and millimetre-scale pale spots that have dark rims | `veined_lookalike`, the same kind of slab with veins and no spots |

The silica patches are painted into the ground image and saved as
`candidate_mask.png` in the scene folder. That mask is not a camera image.
Labels for those patches come from the polygon draped on the terrain. Sulfur
and Cheyava targets are meshes, so they use the same box labeler as rocks.
`exaggerated_spots` multiplies the Cheyava spot diameter by 8 and tags that
rock as an exaggerated analogue. Leave it false to keep the photographed
millimetre scale. Rover abrasion circles and the drill hole are not
reproduced. The PIXL map's purple, pink, and green display colors are not
used as surface colors.

Empty seeds stay empty: no rocks and no sampling candidates. Other scenes mix
the candidates with the ordinary rocks. Set `candidates.enabled` to false to
generate the previous ground only. Remove a name from `categories` to drop
that target. The camera pose is unchanged: the optical center stays 0.762 m
above level ground.

The offline unit tests still use 16 procedural shapes and the four Open
Robotics Fuel rocks from `rover_training`, so they need no download. Set
`procedural_count` or `include_existing` in a config to mix those back in.

## Labels

Labels are not detector output. A CPU rasterizer projects the same visual
triangles the world loads, including the terrain, and keeps the visible
pixels.

| Case | What is written |
| --- | --- |
| Fully outside the image, or only a sub-pixel fragment | No box |
| Fully hidden by terrain or another rock | No box |
| Visible fraction below `min_visible_fraction` (0.30) | No box |
| Visible area below `min_visible_pixels` (24) or box side below `min_box_side_px` (5) | No box |
| Partly in frame | Box clipped to the image |
| Empty scene | Empty `.txt` file, which is a valid YOLO label |

Class ids are `rock`, `silica_soil`, `pale_lookalike`, `sulfur_rock`,
`pale_stone`, `cheyava_rock`, and `veined_lookalike`, in that order. Ordinary
ground stays background. Clean images are written under `images/`. Boxes are
drawn only under `previews/`.

Two independent checks run on every frame:

- **Depth.** The capture refuses the frame when the median difference between
  `/oak/depth/image_raw` and the label mesh on rock pixels is above 8 cm.
- **Gazebo's bounding-box camera.** Every rock carries Harmonic's `Label`
  plugin, and the rover has a `boundingbox_camera` with the same pose and
  optics as the RGB camera. Its boxes arrive on `/oak/boxes`
  (`vision_msgs/Detection2DArray`). They are compared one-to-one with our
  labels by IoU (`bbox_min_iou`, 0.5) and stored in each frame's metadata as
  `bbox_check`. They are never used as labels. A Gazebo box is expected when
  it matches a rock our policy rejected, or when it is already below the
  minimum box size. Anything else is listed in `unexplained_boxes`.
  `validation.json` sums this as `gazebo_bbox_check`.

With `require_bbox_check` true, a frame is not written until a Gazebo box
message near the image time has arrived.

The rover body is not in the rasterizer. The camera looks forward from a mast
at the front of the body, so the body rarely covers a labeled rock.

## Camera values are not OAK-D calibration

`config/sample.json` and `robot.urdf.xacro` carry the same camera. The launch
still applies the config to an in-memory copy of the rover, and capture reads
that copy back from `robot_capture.urdf`. Do not treat these numbers as a
bench calibration.

The optical center is 0.762 m (30 in) above level ground with the rover
resting on its wheels. The wheel collision spheres have radius 0.05 m and are
centered 0.03 m below `base_link`, so `base_link` rests 0.08 m off the ground.
`mount_xyz` z is the joint offset, 0.682 m, not 0.762 m. The camera body,
RGB-D sensor, and optical frame share that origin and the 0.1 rad nose-down
pitch. A visual mast fills the gap from the chassis top. It has no collision.

`horizontal_fov_rad` stays 1.22 rad and is provisional until a measured OAK-D
calibration exists. The Harmonic `rgbd_camera` has one image size, so the
depth image is also 1920 × 1080. Capture rejects a frame whose depth size or
frame id does not match the RGB image.

| Setting | Sample value | Replace with |
| --- | --- | --- |
| `mount_xyz` | `0.2 0 0.682` m from `base_link` | Measured joint offset |
| optical center | 0.762 m above level ground | Measured mount height |
| `tilt_rad` | `0.1` rad nose-down | Measured tilt |
| `width`, `height` | 1920 x 1080 | OAK-D RGB mode |
| `horizontal_fov_rad` | 1.22 rad, provisional | Value from `camera_info`, not a guess |
| `near_m`, `far_m` | 0.3 m, 8.0 m | The mode you actually run |
| `rate_hz` | 30 | Simulation-time rate. Wall-clock rate is lower under software rendering |

The launch applies those overrides to an in-memory copy of the rover. Capture
reads `/oak/camera_info` for the actual `K` matrix.

`/odom` in this stack is the ground-truth relay: `odom_gt` republishes
Harmonic's `OdometryPublisher` pose from `/odom_gt`. It is not wheel
odometry. Labels use that pose, interpolated to the image timestamp. Each
view is reached by teleporting the rover with `/world/mars_training/set_pose`.

## Splits

Every frame from one seed stays in one split. The split is a hash of the seed,
so adding seeds does not move an existing scene. The sample seeds 100–119
produce 14 train, 2 val, and 4 test scenes, 10 views each, 200 images.
Seeds 100, 110, and 117 are empty. Seed 100 is an empty test scene.

`data.yaml` points Ultralytics at `images/train`, `images/val`, and
`images/test`. `classes.txt` lists the class names above. Nothing in this
package starts a training run.

## Commands

Build the image once, from `rover_ws` in PowerShell:

```powershell
docker build -f docker/Dockerfile.jazzy -t rover-jazzy .
```

A headless shell:

```powershell
docker run --rm -it -v ${PWD}:/work -w /work rover-jazzy bash
```

To see the Gazebo window, add the WSLg display mounts:

```powershell
docker run --rm -it -v ${PWD}:/work -w /work -v /run/desktop/mnt/host/wslg/.X11-unix:/tmp/.X11-unix -v /run/desktop/mnt/host/wslg:/mnt/wslg -e DISPLAY=:0 -e WAYLAND_DISPLAY=wayland-0 -e XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir rover-jazzy bash
```

If another ROS container is running (for example `rover-humble`), start
capture containers with `--network none`. Each capture also sets its own
`GZ_PARTITION` and keeps ROS discovery on localhost, but two containers on
the same Docker network can still see each other's topics.

Inside the container, Jazzy builds go to separate folders so they do not
clash with Humble builds:

```bash
source /opt/ros/jazzy/setup.bash
cd /work
colcon --log-base log_jazzy build --build-base build_jazzy --install-base install_jazzy \
  --packages-select four_wheel_sim rover_training mars_training_environment
source install_jazzy/setup.bash
```

Unit tests do not need Gazebo or the asset download:

```bash
cd /work/src/mars_training_environment
PYTHONPATH=$PWD:/work/src/rover_training python3 -m unittest discover -s test -p 'test_*.py'
```

Download and prepare the scanned assets (about 100 MB, a minute of Blender).
This needs network access, so do not use `--network none` for this step:

```bash
ros2 run mars_training_environment mars_assets all --cache /work/assets_cache
```

Only after editing `config/assets.json`, rebuild the lock file and commit it:

```bash
ros2 run mars_training_environment mars_assets lock \
  --lock /work/src/mars_training_environment/config/assets.lock.json
```

Generate the 20 sample scenes:

```bash
ros2 run mars_training_environment mars_generate \
  --output /work/dataset/mars_scenes --asset-cache /work/assets_cache
```

Without `--asset-cache`, `$MARS_ASSET_CACHE` or `./assets_cache` is used. The
same config and seed rewrite the same scene. A different config in an
existing output directory is rejected.

Look at one world:

```bash
ros2 launch mars_training_environment mars_world.launch.py \
  scene:=/work/dataset/mars_scenes/scenes/scene_000101/scene.json gui:=true
```

Capture and validate. This starts a headless Gazebo per scene:

```bash
ros2 run mars_training_environment mars_dataset \
  --bundle /work/dataset/mars_scenes --output /work/dataset/mars_sample
```

Add `--scene scene_000101` for a smoke test, or `--resume` after a stopped
run. Check labels again and rebuild the contact sheet:

```bash
ros2 run mars_training_environment mars_validate \
  --dataset /work/dataset/mars_sample --bundle /work/dataset/mars_scenes
```

`dataset/` and `assets_cache/` are gitignored. Do not commit images, worlds,
or downloaded assets.

## What has been run

All in `rover-jazzy` on this laptop, software rendering, `--network none`,
after the camera change to 1920 × 1080 at 30 Hz and a 0.762 m optical center.
Older dataset folders were removed; they were captured with the previous camera.

- Unit tests: 22 in `rover_training`, 32 in this package.
- `mars_verify_runtime` on the flat maze: RGB and depth are both 1920 × 1080
  in `camera_optical_frame`, depth is `32FC1`, and `camera_info` focal length
  matches the provisional 1.22 rad field of view. Simulation-time rate was
  30.3 Hz. Wall-clock rate was 2.4 Hz. `base_link` rested at 0.080 m and the
  optical center was at 0.762 m. The rover drove 0.12 m.
- Scene 101, 10 frames, in `dataset/mars_look_1080p_10`: no validation errors.
  94 labels, every one matched by a Gazebo box, median IoU 0.96. Median rock
  depth error under 0.01 mm. 9 of 10 frames agree fully. The other frame has
  one 16 × 5 px Gazebo box on a rock our policy rejected as mostly hidden
  (visible fraction 0.12). Its IoU with our visible pixels is 0.46, just under
  the 0.50 cutoff, so it is listed in `unexplained_boxes` rather than counted
  as an expected rejection. It is not a shifted label.
- Speed for that scene: about 27 s of wall time per image including Gazebo
  start-up, a simulation real-time factor near 0.05, and 3.8 s of labeling
  per frame.

Not yet run with this camera: `mars_world.launch.py gui:=true`, and the full
20-scene sample.

## Halo Strix

Build `docker/Dockerfile.jazzy` there and run the same commands. Harmonic
renders with `ogre2`. Without a GPU in the container, keep
`LIBGL_ALWAYS_SOFTWARE=1` (the capture sets it). If `/dev/dri` is passed in
and hardware rendering works, compare one scene's real-time factor in
`reports/scene_*.json` before growing the dataset. A ROCm or PyTorch install
does not speed up Gazebo, and this package does not start training.

## Limitations

- The simulated camera is 1920 × 1080 at 30 Hz in simulation time. Its
  horizontal field of view, 1.22 rad, is provisional, not a bench calibration.
  Depth uses the same resolution as RGB.
- The optical center is 0.762 m above level ground.
- Rock sizes are not yet tied to a real gripper.
- Ten rock shapes, two held out. More single-rock scans would add variety.
- Rocks do not move, and the ground does not deform.
- Labels ignore chassis occlusion.
- The `odom_gt` and `twist_stamper` processes can print
  `rcl_shutdown already called` while the launch is exiting. The capture has
  already finished when that appears.
- The capture keeps an image only when RGB, camera info, depth, Gazebo boxes,
  and ground-truth odometry line up. `--resume` replaces a scene that stopped
  halfway.
