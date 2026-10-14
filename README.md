# Rover perception and Mars training environment

This branch contains the synthetic dataset capture and review workflow for the
Mars rover simulation in ROS 2 Jazzy and Gazebo Harmonic. It extends
`perception/mars-sampling-sites` with three training classes, stronger framing
and visibility checks, resumable capture, a browser review editor, and a
960-pixel training preset.

**October 12, 2026 milestone:** the Halo/Strix completed the v2 pilot with
**50 images, no validation errors, and 30 train / 10 validation / 10 test images**.
The reviewed export has completed its first 30-epoch training run. The completed capture came from the existing v2 scene bundle through
resumes; newer plans also use full-resolution framing checks.

**October 13, 2026 milestone:** the balanced 500-image capture, browser review,
30-epoch training and held-out synthetic test evaluation are complete. The
500-image model reached test mAP50 **0.937** and mAP50-95 **0.735** across 100 images.
Per-class test mAP50 was 0.933 rock, 0.901 pale soil and 0.977 yellow crystal.
The Halo has also run live predictions while manually driving in new scenes.

Branch: `perception/mars-pilot50-v2-review`.

## Live driving with trained vision

The [Halo live-vision guide](scripts/HALO_LIVE_VISION.md) prepares a separate
interactive Gazebo session, three new evaluation scenes, and a browser camera
view with predictions for all three classes. Browser driving uses held WASD,
arrow keys or buttons, with speed sliders up to 0.60 m/s and 1.50 rad/s.
Persistent control and video connections keep commands independent of video
delivery; slow viewers receive the newest frame instead of accumulating a queue.
The live launcher now defaults to headless, RGB-only 960 x 540 rendering and
targets up to 30 processed FPS. The dashboard reports actual display FPS,
camera age, command latency and simulation speed. Capture retains its full
RGB-D/bounding-box sensors and saved configuration. ROS stays in Docker while
inference uses the host's ROCm environment, with aiohttp added for WebSockets.
Command expiry, release/focus handling and camera freshness checks stop stale
movement. Local tests do not replace measuring actual Gazebo/ROCm FPS on the Halo.

## Balanced class coverage for the next dataset

`config/coverage500_balanced.json` in the environment package defines 50 new
scenes and 500 views, split 300 train / 100 val / 100 test. Mixed scenes require
4 rock, 3 pale-soil and 3 yellow-crystal primary targets per ten views. Five
rock-only scenes and five background scenes are retained. Expected soil and
crystal targets are 72 each in training and 24 each in val/test.

The planner chooses the class first and retries only targets of that class.
Infeasible mixed layouts can be resampled deterministically within the same
physical ranges and split; visibility/framing thresholds are preserved.
Per-scene progress and a quota audit prevent a low-crystal plan from silently
passing as balanced. See the [balanced Halo generation/capture guide](scripts/HALO_COVERAGE500_BALANCED.md).

The original pilot has now completed 30 training epochs on the Halo. Its final
validation mAP50 was 0.573 and mAP50–95 was 0.425, with only three soil and one
crystal validation instances. The separate test evaluation stopped on an
incorrect checkpoint path, so no held-out test result is recorded yet. These
limited minority-class observations motivated the balanced expansion.

## What has been implemented

- **Scene appearances and geometry.** Scanned rocks, fine sand, wind ripples,
  terrain-following mudstone exposures, pale soil patches, yellow crystal
  appearances, and lookalike controls. Size, burial, orientation, dust, lighting,
  physical clutter, and held-out mesh variation are retained.
- **Training labels.** Three canonical classes preserve original appearance
  names and asset provenance in metadata. Ground patches use visible terrain
  ray hits and the painted category mask; physical rocks use their visual meshes.
- **Target framing.** Capture distances of 1.8–3.2 m, ordinary-view margins,
  minimum target visibility, and pixel size measured at training resolution.
  One view per five is deliberately difficult; background scenes remain empty.
  New plans use conservative coarse checks followed by capture-resolution checks.
- **Capture validation and recovery.** Synchronized RGB, depth, camera info,
  measured pose, per-object depth support, RGB sky checks, and renderer-box
  comparisons. Failed frames save diagnostics. A size-only failure can retry
  from a closer pose within distance, collision, slope, and boundary limits.
  Completed scenes survive `--resume`; incomplete scenes are recaptured.
- **Automatic review and corrected exports.** Live updates, split/class/status
  filters, 640/960/1280 previews, box and class edits, drawing/removing boxes,
  notes, reversible exclusions, original-label restoration, and separate
  versioned dataset exports. Raw captures are preserved.
- **Training preparation.** `train960.yaml` supplies detection training at 960
  pixels, batch 4, 30 epochs, and seed 0. Optional prediction overlays support
  inspection of misses, wrong classes, and extra detections after training.

These labels describe appearance rather than chemical composition:

| ID | Training class | Included appearances |
| --- | --- | --- |
| 0 | `rock` | Ordinary rocks, pale stones, and mudstone/veined lookalikes |
| 1 | `pale_soil_patch` | Pale soil and its visual lookalikes |
| 2 | `yellow_crystal_rock` | Sulfur-like yellow crystal appearances |

The camera captures 1920 × 1080 RGB and depth at 30 Hz in simulation time.
Its optical center is 0.762 m above level ground. The 1.22 rad horizontal field
of view remains provisional until the physical OAK-D is calibrated.

## Capture issues resolved during the pilot

The original geometry checks could label objects that the RGB-D renderer did
not display. The v2 work added object-level depth/sky checks and retained failure
evidence instead of exporting unsupported labels.

Two specific discrepancies were reproduced and corrected:

1. **Coarse target-size estimates.** Scene 102/view 004 measured 12.5 pixels in
   the coarse plan but 11 pixels at training size in the captured image, below
   its 12-pixel difficult-view minimum. Planning now confirms full-resolution
   bounds. Existing plans can reframe a size-only failure while keeping all
   acceptance checks.
2. **RGB-D clipping and ground projection.** The annotation raster used forward
   depth for the far limit, while Ogre2 clips by total camera-to-surface distance.
   Scene 104's incidental ground patch was 7.6–8.0 m forward but 8.2–8.7 m away,
   beyond the 8 m sensor range. The raster now matches the sensor's radial far
   and forward near clipping. Ground labels also follow the real terrain surface
   rather than a lifted proxy mesh.

The clipping behavior is defined in [Ogre2's depth-camera shader](https://github.com/gazebosim/gz-rendering/blob/gz-rendering8/ogre2/src/media/materials/programs/GLSL/depth_camera_fs.glsl).
Depth, sky, and primary-target thresholds were retained throughout these fixes.

## Latest Halo capture result

The user-reported final validation contains:

| Measure | Result |
| --- | --- |
| Captured images | 50 across 10 scenes |
| Splits | 30 train / 10 val / 10 test |
| Validation errors | None (`errors: []`) |
| Background frames | 5 |
| Exported mesh labels matched to Gazebo boxes | 324 |
| Exported mesh labels without a Gazebo match | 0 |
| Median box IoU | 0.9336 |
| Frames with full box agreement | 37 / 50 |
| Remaining box disagreements | 18 unexplained renderer boxes across 13 frames |
| Capture wall time across scene reports | About 395 seconds |

**The remaining box disagreements are review flags.** They did not stop capture
or produce validation errors. The bounding-box camera is an independent check,
and its boxes are not used as training labels. Differences in clipping,
occlusion, and box definitions can cause extra boxes; inspect the flagged
images before deciding whether a training label needs correction. Painted
ground patches are checked by depth and paint-mask agreement, not mesh boxes.

The dataset also includes intentional difficult views, clipped incidental
objects, and background images. These counts alone are not reasons to discard
frames. All accepted images, including images saved before earlier failures,
are available in the review editor. Rejected attempts appear as diagnostics.

## Build and capture a new pilot

Clone this branch into a separate directory:

```bash
git clone --branch perception/mars-pilot50-v2-review --single-branch https://github.com/agfoti04/AMD-Capstone.git
cd AMD-Capstone
```

On the Linux/Halo host, build the Jazzy image and enter an asset-preparation
shell from the repository root:

```bash
docker build -f docker/Dockerfile.jazzy -t rover-jazzy .
```

```bash
docker run --rm -it --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "${PWD}:/work" -w /work rover-jazzy bash
```

Inside the container:

```bash
source /opt/ros/jazzy/setup.bash
colcon --log-base log_jazzy build --base-paths src \
  --build-base build_jazzy --install-base install_jazzy \
  --packages-select four_wheel_sim rover_training mars_training_environment
source /work/install_jazzy/setup.bash
```

Reuse a prepared `assets_cache`, or fetch and prepare the pinned scans:

```bash
ros2 run mars_training_environment mars_assets all --cache /work/assets_cache
```

Exit this shell. For a new capture, use fresh bundle/output folders and run
from the repository root on the host:

```bash
docker run --rm --network none --shm-size=1g \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e LIBGL_ALWAYS_SOFTWARE=1 \
  -v "${PWD}:/work" -w /work rover-jazzy bash -c '
    set -e
    source /opt/ros/jazzy/setup.bash
    source /work/install_jazzy/setup.bash
    ros2 run mars_training_environment mars_dataset \
      --generate --resume \
      --config /work/src/mars_training_environment/config/pilot50.json \
      --asset-cache /work/assets_cache \
      --bundle /work/dataset/pilot50_scenes_v2 \
      --output /work/dataset/pilot50_v2
  '
```

To resume an existing v2 bundle after a capture failure, **omit `--generate`**
and keep its original bundle/output paths. See the [Halo capture and resume
guide](scripts/HALO_PILOT50_V2.md) for the deployed Halo paths and troubleshooting
history. Source-only transfer archives mentioned there are local deployment
artifacts; cloning this branch provides the current source directly.

## Review captured images

The editor can inspect accepted frames while capture is incomplete. Export
requires successful raw validation and at least one included image per split.

On the Halo host, from this repository in the existing `rover-yolo` environment:

```bash
conda activate rover-yolo
PYTHONPATH="$PWD/src/mars_training_environment:$PWD/src/rover_training" \
  python -m mars_training_environment.review --dataset dataset/pilot50_v2
```

Leave that terminal running. Open `http://127.0.0.1:8765` on the Halo, or forward
port 8765 through VS Code Remote-SSH. A Windows SSH tunnel is another option:

```powershell
ssh -N -L 8765:127.0.0.1:8765 ian@halo-strix
```

Start with **Status → Flagged**, use **Training size → 960**, and inspect
`gazebo_box_disagreement`, `target_reframed`, and
`legacy_ground_region_projection` flags. Correct classes and boxes, then click
**Save corrections / mark reviewed**. Exclusions can be reversed.

After the intended manual review, **Export reviewed training data** creates a
separate corrected dataset and returns its `data.yaml` path. Export does not
require every included frame to be marked reviewed, so finish the intended review
first. Opening `review/index.html` directly provides a read-only snapshot.

Dataset layout:

```text
dataset/pilot50_v2/
  images/<train|val|test>/       clean captures
  labels/<train|val|test>/       YOLO labels
  metadata/<train|val|test>/     pose, geometry, checks, and provenance
  previews/                    overlays and contact sheet
  diagnostics/                 rejected capture attempts
  validation.json              raw dataset validation
  review/                      editor snapshot, edits, history, and exports
```

Captured datasets, downloaded caches, build outputs, and model weights are
gitignored and are not bundled with this source branch.

## Train and evaluate after review

Use the reviewed export's YAML and the pretrained model already selected for
the Halo training setup:

```bash
yolo train cfg=src/mars_training_environment/config/train960.yaml \
  model=/path/to/pretrained-model.pt \
  data=/path/to/reviewed-export/data.yaml device=0
```

The placeholders must be replaced with actual files. The initial v2 training
result is recorded above; a held-out test-set result remains pending. Use an
absolute `project` path for future runs and the checkpoint path printed by
training: the first run saved under `runs/detect/runs/detect/pilot50_v2_960`.
Review can load optional YOLO files
under `predictions/<split>/<frame>.txt` for prediction overlays and matching hints.

## Verification and remaining work

- The class-quota update passed all 113 environment/review tests on Linux and
  24 rover-training tests on Windows. The complete 500-view plan was checked
  against the cached real assets using the same per-scene planner, with exact
  quotas, preserved framing and no audit errors. New balanced rendered capture
  remains pending on the Halo; local Docker was unavailable for this update.
- 102 distinct environment/review tests were verified across the final full run
  and a focused quality/review rerun after correcting a synthetic fixture's
  camera range. Earlier rover-training verification passed 24 tests.
- Local rendered regressions completed all five images for the original scene
  102 and scene 104, with `errors: []`; overlays were visually inspected.
- Browser QA covered class/box edits, drawing/removal, exclusion/restoration,
  resolution changes, and separate corrected exports.
- The final Halo v2 run completed all 50 images with the expected splits.

Next work is manual review/export, 960-pixel training, and test-set prediction
analysis. Live inference while driving the rover, route execution, synchronized
route-level ground truth, physical camera calibration, and hardware-based size
limits remain future work. Renderer-box disagreement analysis remains part of
review; successful capture does not establish that every box is ready for training.

See the [environment reference](src/mars_training_environment/README.md) for
geometry, configuration, asset licensing, and checks. The [legacy simulator
workflows](docs/LEGACY_WORKFLOWS.md) retain the older Humble/Classic and earlier
Harmonic commands.
