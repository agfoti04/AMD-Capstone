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

## Latest verified capture: October 12, 2026

The Halo v2 pilot completed all 50 images across ten scenes with `errors: []`,
split 30 train / 10 val / 10 test. All 324 exported mesh labels matched a
Gazebo box, with no missing renderer matches and median IoU 0.9336. Thirteen
frames retain 18 unexplained renderer boxes as review flags; 37 frames have
full box agreement. Ground patches use separate depth/paint-mask checks.

The capture is complete. Manual review and a corrected export come next;
no completed v2 model training or evaluation result is recorded yet. The
[main README](../../README.md) provides the current build, capture, review,
and training workflow. The verification history later in this document
describes the individual fixes leading to this completed run.

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

The detector uses three broad visual classes. Appearance names in
`candidates.categories` select the original meshes and materials; the exporter
maps those appearances to the labels below. Pale stones and both soil looks
remain in the simulation, with separate instances and their existing textures.

| Class ID | Detection class | Included appearances |
| --- | --- | --- |
| 0 | `rock` | Ordinary scanned rocks, intact `pale_stone` rocks, and the `cheyava_rock` / `veined_lookalike` mudstone appearances |
| 1 | `pale_soil_patch` | Irregular `silica_soil` patches with embedded stones and smoother `pale_lookalike` patches |
| 2 | `yellow_crystal_rock` | `sulfur_rock`, a pale fractured stone with yellow crystals on one face |

These are appearance labels, not chemical measurements. The NASA photographs
in `Mars_Gazebo_Reference_Pack` are visual references, not textures copied into
the simulator. Original appearance names are retained in scene records and in
each annotation's `appearance_name` for later analysis.

The ordinary-rock count stays at 38–55 per populated scene. Mixed scenes also
keep 1–2 of each pale-patch appearance, 1–2 yellow crystal rocks, 2–4 pale
stones, and one of each mudstone appearance. Rock sizes, burial, orientation,
dust coating, terrain, and lighting variation retain their existing settings.
The mudstone slabs retain their terrain-following geometry and textures but
are labeled as ordinary `rock`, without separate vein or spot classes.

Pale soil is painted into the ground texture and saved as `candidate_mask.png`.
That mask records appearance variants, not YOLO class IDs or camera pixels.
Patch boxes come from visible terrain ray hits within the region polygons
and painted category mask. Yellow crystal rocks
and pale stones use the same mesh labeler as ordinary rocks.

`dataset.rocks_only_seeds` produces ordinary rocks and pale stones without
soil patches, yellow crystal rocks, or mudstone sampling-site appearances.
These scenes still have class-0 boxes.
`dataset.empty_seeds` has no rocks or candidate patches and produces empty
label files. The two seed lists must be disjoint. Remaining seeds mix all
enabled appearances. Set `candidates.enabled` to false to keep ordinary rocks
and terrain only. The optical center stays 0.762 m above level ground.

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

Class IDs are `0: rock`, `1: pale_soil_patch`, and `2: yellow_crystal_rock`.
Ordinary ground stays background. Clean images are written under `images/`.
Boxes are drawn only under `previews/`. A scene manifest uses schema version 5,
and a bundle plan uses version 3; both carry the exact ordered `class_names`.
Older bundles and dataset indexes cannot be resumed with this label mapping:
use fresh output directories when generation and capture are next requested.
Existing datasets remain unchanged; the preview tool reads their own
`classes.txt` so old numeric IDs retain their original names.

The following checks run on every frame:

- **Depth.** The capture refuses the frame when the median difference between
  `/oak/depth/image_raw` and the label mesh on rock pixels is above 8 cm.
- **Each retained object's visibility.** At least 90% of its predicted visible
  instance pixels must have valid sensor depth within 3 cm of the expected
  surface. Invalid depth counts against support. Ground patches compare with
  actual terrain depth and must also have at least 85% agreement with the
  painted appearance mask. A mask containing only the uniform sky color is
  rejected even if the other geometry checks pass.
- **Gazebo's bounding-box camera.** Every rock carries Harmonic's `Label`
  plugin, and the rover has a `boundingbox_camera` with the same pose and
  optics as the RGB camera. Its boxes arrive on `/oak/boxes`
  (`vision_msgs/Detection2DArray`). They are compared one-to-one with our
  labels by IoU (`bbox_min_iou`, 0.5) and stored in each frame's metadata as
  `bbox_check`. They are never used as labels. A Gazebo box is expected when
  it matches a rock our policy rejected, or when it is already below the
  minimum box size. Anything else is listed in `unexplained_boxes`.
  `validation.json` sums this as `gazebo_bbox_check`.

The per-object checks are stored under `depth_quality.visibility_check`.
When a mesh's projected pixels match uniform sky and have no valid sensor
surface depth, its label is removed as `renderer_invisible_sky`. Its original
failed checks remain in `visibility_check.rejected_objects`, and the review
page flags the removed object by name. This does not remove finite-depth
mismatches, ground-patch failures, or a visible mesh with a sky-colored material.
If the removed object is the primary target, the framing check still fails.
Frames that fail visibility or intended-target framing are stopped before an
image/label pair is exported. RGB, box overlays, sensor/expected depth,
instance masks, pose, and failure reasons are saved under
`diagnostics/<scene>/<view>/`. They appear in the review page as diagnostic
frames, separate from accepted training images. These checks detect unsupported
labels; the failed examples still need inspection to determine the rendering
or geometry cause.

## Target framing

Capture positions are 1.8–3.2 m from the lens to the target. Camera height,
tilt, original 1920×1080 resolution, object counts, and appearance randomization
are unchanged. The planner projects the actual target mesh and rasterizes a
candidate view to test terrain/rock occlusion before accepting the pose.

Four views in each group of five must show the primary target fully within a
10% image margin, with at least 70% visible and a shorter box side of at least
24 pixels after resizing to `capture.training_imgsz` (960). One view in each
group is deliberately more difficult: it allows edge clipping, requires 30%
visibility, and uses a 12-pixel minimum. The rules apply only to the intended
target; small incidental rocks retain their labels. Empty scenes remain valid
background examples. Target identity and planned framing are stored in each
view; capture checks the actual settled pose again before writing a frame.

The default thresholds are starting values, not a claim about model accuracy.
An infeasible target or failed rendered frame produces diagnostics rather than
silently weakening the policy or exporting a bad example.

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
Seeds 102, 107, 108, and 114 contain only the rock class, covering train, val,
and test. The default plan has 13 mixed scenes, 4 rock-only scenes, and 3 empty
scenes. This is a configured plan; changing the config does not capture it.

For an initial 50-image run, pass `--config` with `config/pilot50.json` to
`mars_generate` (or to `mars_dataset --generate`). This overrides only the
scene seeds and views per scene: 10 scenes with 5 views each, giving 30 train,
10 val, and 10 test images. Six scenes are mixed, three contain ordinary rocks
and pale stones, and one is empty. Both val and test include a mixed scene
and a rock-only scene; train includes all three scene types. Clutter and
appearance settings are inherited from `sample.json`.

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
  --base-paths src --packages-select four_wheel_sim rover_training mars_training_environment
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

## Automatic review and corrections

Every capture creates `review/index.html` and `review/index.json` automatically,
including partial runs and rejected-frame diagnostics. The HTML can be opened
directly for a read-only review. With the review server running, new frames
appear every five seconds and corrections can be saved.

On the Halo host, in the existing `rover-yolo` Conda environment, run from the
copied project folder (outside the ROS container):

```bash
conda activate rover-yolo
PYTHONPATH="$PWD/src/mars_training_environment:$PWD/src/rover_training" \
  python -m mars_training_environment.review --dataset dataset/pilot50_v2
```

The server listens on `127.0.0.1:8765`. In a VS Code Remote-SSH window, use
the Ports panel to forward port 8765 and open its local browser address.
Alternatively, from Windows open a dedicated SSH tunnel:

```powershell
ssh -N -L 8765:127.0.0.1:8765 ian@halo-strix
```

Then open `http://127.0.0.1:8765` locally. No Gazebo GUI or remote desktop is
needed to review the saved images.

The page supports split/class/status filters, 640/960/1280 previews, flags,
target information, drawing/removing boxes, coordinate and class corrections,
review notes, and excluding/re-including frames. **Restore original capture**
restores the original labels and includes the frame again. Unsaved edits are
preserved during automatic refresh; stale edits are rejected if another editor
has changed that frame. If a partial scene is recaptured, old corrections are
preserved but flagged as stale rather than applied to the replacement image.
Review or restore that frame before exporting.

Corrections are reversible sidecars in `review/edits/`, with an audit history
in `review/history.jsonl`. Raw capture images, YOLO labels, and metadata stay
unchanged. **Export reviewed training data** writes a new dataset under
`review/exports/<revision>/`, retaining the original scene splits and omitting
excluded frames. It returns a `data.yaml` with the host's actual paths and a
`review_export.json` recording source revisions, reviewed labels and file hashes.
Export requires a complete, validated capture and at least one included image
in every split. Exclusion can reduce the reviewed image count below 50.

Later, YOLO predictions saved as `predictions/<split>/<frame>.txt` can be
displayed as dashed boxes. Standard normalized YOLO xywh rows with an optional
sixth confidence value are supported. The page flags wrong classes, missing
detections, and extra detections using one-to-one box matching at IoU 0.5.
An empty prediction file represents zero detections. These are review hints;
full performance measurement still uses held-out validation.

To regenerate a page for an existing three-class capture, or export from the
command line:

```bash
PYTHONPATH="$PWD/src/mars_training_environment:$PWD/src/rover_training" \
  python -m mars_training_environment.review --dataset dataset/pilot50 --build-only

PYTHONPATH="$PWD/src/mars_training_environment:$PWD/src/rover_training" \
  python -m mars_training_environment.review --dataset dataset/pilot50_v2 --export
```

Old captures can be reviewed; their missing per-object visibility audit is
flagged. New captures require a fresh version-3 bundle and output folder.

## Preliminary training on the Halo

`config/train960.yaml` sets `imgsz=960`, `batch=4`, 30 epochs, square
letterboxing, and seed 0. It does not start training automatically. After
review/export, supply the returned YAML and the model used for the initial
Ultralytics smoke test:

```bash
conda activate rover-yolo
yolo train cfg=src/mars_training_environment/config/train960.yaml \
  model=/path/to/pretrained-model.pt data=/path/from/review/export/data.yaml device=0
```

Keep the original high-resolution captures. Review pixel sizes are measured
after resizing, and higher resolutions do not fix missing or clipped targets.

## What has been run

The earlier three-class revision passed 24 offline tests in `rover_training`
and 47 in this package. A 50-image pilot was subsequently captured on the
Halo (30 train / 10 val / 10 test). Its geometry checks passed, but visual
review found sky-only boxes and poor target framing, motivating the current
visibility and review revision. The first v2 Halo run completed scene 101 in
26.8 seconds, then correctly identified an unsupported sky-only `rock_006` label
in scene 102/view 001. The follow-up change removes only confirmed invisible
mesh labels while preserving their failed evidence and the other acceptance
checks. The Halo retry then reached scene 102/view 004, whose target measured
11 pixels at training size despite the coarse planner predicting 12.5 pixels.
The framing correction adds conservative coarse bounds and a full-resolution
planning pass. Existing plans can recover a size-only failure using at most
three closer, collision-checked poses within the configured distance range.
Every new image must pass the unchanged capture checks. Corrections are saved
in metadata, flagged as `target_reframed` in review, and failed attempts remain
available as diagnostics. The historical
captures below predate the three-class change.

The prior framing revision had 90 passing tests (89 in the full
run, followed by the final 35-test quality/review suite after one extra test).
The earlier 24 rover tests also passed; rover code was unchanged by this fix.
The original scene 102 was recaptured locally in headless Gazebo after the
framing correction: five images, `errors: []`, 175.1 seconds. View 004 recovered
from 11 to 13 pixels after one closer pose. All 51 labels matched renderer
boxes; one extra box near rock_014 remained flagged for manual review. The
user subsequently completed scenes 102
and 103 on the Halo; scene 104 exposed an off-axis far-clipping mismatch.
Ogre2 RGB-D clips at radial range 8 m, but the annotation raster previously
clipped only forward Z. It now matches the sensor's radial far/forward near
rule. Ground labels also use the actual terrain ray hits and paint-mask
categories, replacing lifted proxy surfaces. Earlier accepted ground-region
frames are flagged `legacy_ground_region_projection` for manual review.
No sensor-depth, sky, or primary-target acceptance threshold was lowered.
The corrected original scene 104 passed a local rendered capture: five images,
`errors: []`, 158.4 seconds. The first view's primary ground patch had 100%
depth and paint-mask support; the nonexistent off-axis patch was not labeled.
All 39 mesh labels matched renderer boxes; two extra boxes in view 001 remain
flagged for manual review. 102 distinct environment/review tests are verified:
101 passed in the final full run; an old size-regression fixture was then moved
inside the corrected sensor range, and all 36 quality/review tests passed.
No production code changed after the full run. The subsequent Halo deployment
completed all 50 v2 frames with `errors: []`; see the latest capture summary
above for its remaining review flags. The clipping rule matches the installed Ogre2 shader:
https://github.com/gazebosim/gz-rendering/blob/gz-rendering8/ogre2/src/media/materials/programs/GLSL/depth_camera_fs.glsl
The review editor was exercised in a browser with synthetic fixtures, including
class edits, box drawing/removal, exclusion/restoration, and curated export.
The 50-view plan was also checked against the actual cached scanned assets in
temporary folders, preserving 30 train / 10 val / 10 test views. The final
Halo run passed the capture visibility checks on all accepted frames.
Additional regression checks cover sky-only label removal, preservation of
real sensor/geometry failures, primary-target rejection, review flags, and
surfacing capture exceptions in the outer dataset log.

All in `rover-jazzy` on this laptop with software rendering, after the camera
change to 1920 × 1080 at 30 Hz and a 0.762 m optical center. Earlier checks
used `--network none`; the October 9 terrain-placement check used the
`rover-gazebo` container on its default bridge network.
Older dataset folders were removed; they were captured with the previous camera.

- Unit tests: 22 in `rover_training`, 42 in this package, including mudstone
  ground-contact and matching render/label geometry checks.
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

- October 9 terrain-following sampling sites: scene 101, ten frames in
  `dataset/mars_sites_flush_10`, with no validation errors. All 83 mesh labels
  matched Gazebo boxes, all ten frames agreed, and median IoU was 0.941.
  The repaired mudstone undersides are buried 20 mm across their footprints;
  tops are 1–8 mm above the visual terrain. Capture took about 239 s including
  launch and shutdown. These generated files remain gitignored.

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
