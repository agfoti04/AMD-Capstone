# Halo pilot 50: framing, visibility checks, and review

This revision preserves the original three classes, physical clutter, camera
height/tilt, dust, light, burial, orientation, and size variation. It adds
projected-target framing, per-object depth/RGB checks, a review editor, and a
960-pixel training preset. New scenes and capture output need fresh folders.
The existing pilot remains available for comparison and review.

**Latest result, October 12, 2026:** the resumed Halo v2 capture completed all
50 images with `errors: []` and 30 train / 10 val / 10 test splits. All exported
mesh labels have renderer matches; 13 frames retain box-disagreement review
flags. The next action is [review and corrected export](#review-automatically-and-correct-labels).
The transfer/resume sections below document deployment and failure recovery.
Cloning `perception/mars-pilot50-v2-review` from GitHub provides the current
source without the local transfer archives.

## Copy the updated source from Windows

The prepared archive contains source, configuration, and documentation only.
It contains no captured dataset, cached scans, build folders, or model weights.

```powershell
cd C:\Projects\Capstone
scp .\halo-pilot50-v2-source-20261012.tar.gz ian@halo-strix:~/
```

Use the same host/IP as the SSH command that already works.

## Open the project through Remote-SSH

In VS Code, press Ctrl+Shift+P, select **Remote-SSH: Connect to Host**, and
connect to `ian@halo-strix` (or your existing SSH alias). On the Halo:

```bash
mkdir -p ~/capstone-training/mars-pilot50-v2
tar -xzf ~/halo-pilot50-v2-source-20261012.tar.gz \
  -C ~/capstone-training/mars-pilot50-v2
```

Open `/home/ian/capstone-training/mars-pilot50-v2` in that remote VS Code
window. Its file explorer, image previews, and integrated terminal now access
the Halo. You can inspect capture images, labels, diagnostics, and reports
without copying each result back to Windows. Remote-SSH also forwards the
review server's port. See Microsoft's [Remote-SSH guide](https://code.visualstudio.com/docs/remote/ssh).

The Remote Development pack also includes Dev Containers and WSL support.
Remote-SSH is the first tool to use here. Dev Containers can later provide
ROS-aware editing/debugging in a running development container, once Docker
access is configured for the remote editor. GPU training continues in the
existing `rover-yolo` Conda environment on the Halo host.

Installing the extension alone does not move a coding-agent session from
Windows to the Halo. Check the session's workspace and execution environment
before expecting it to edit remote files or run remote commands. The changes
in this revision were implemented in `C:\Projects\Capstone`.

For Codex in the desktop app to operate directly on the Halo, use the app's
own **Settings > Connections > SSH** configuration and choose the remote
project. The remote host needs Codex installed and authenticated, with its
command available in the SSH login shell. This connection is separate from
VS Code Remote-SSH. See [OpenAI's remote-connection guidance](https://learn.chatgpt.com/docs/remote-connections).

## Build on the Halo

The existing `rover-jazzy` image can be reused; the Dockerfile is unchanged.

```bash
cd ~/capstone-training/mars-pilot50-v2

sudo docker run --rm \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD:/work" -w /work \
  rover-jazzy bash -c '
    set -e
    source /opt/ros/jazzy/setup.bash
    colcon --log-base log_jazzy build \
      --base-paths src \
      --build-base build_jazzy --install-base install_jazzy \
      --packages-select four_wheel_sim rover_training mars_training_environment
  '
```

## When ready, capture the next 50 images

This command starts an actual capture. It reuses the prepared scans from the
first pilot through a read-only cache mount. If that cache lives elsewhere,
change the second `-v` source path.

```bash
cd ~/capstone-training/mars-pilot50-v2

sudo docker run -d --name mars-pilot50-v2 --network none --shm-size=1g \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e LIBGL_ALWAYS_SOFTWARE=1 \
  -v "$PWD:/work" \
  -v "$HOME/capstone-training/mars-pilot50/assets_cache:/assets_cache:ro" \
  -w /work rover-jazzy bash -c '
    set -e
    source /opt/ros/jazzy/setup.bash
    source /work/install_jazzy/setup.bash
    ros2 run mars_training_environment mars_dataset \
      --generate --resume \
      --config /work/src/mars_training_environment/config/pilot50.json \
      --asset-cache /assets_cache \
      --bundle /work/dataset/pilot50_scenes_v2 \
      --output /work/dataset/pilot50_v2
  '

sudo docker logs --follow --tail 50 mars-pilot50-v2
```

The plan is 30 train / 10 val / 10 test images. Each populated scene has four
ordinary target-focused views and one difficult view; the empty scene has
five background views. The view planner requires target visibility, margin,
and size at 960 pixels. Capture checks the actual camera pose again, along
with each retained object's sensor depth and rendered sky appearance.

A failed check stops that frame before export and saves diagnostics. The
review page includes these rejected examples. They need inspection to resolve
the cause; passing the offline planner is not a guarantee about the renderer.
The visibility fix removes only a mesh label whose pixels match sky and have
no valid sensor depth. The failed evidence is retained in metadata and flagged
by name in review. Other depth failures and missing primary targets still stop
the frame.

## Review automatically, and correct labels

The dataset receives `review/index.html` automatically as images are captured.
For live updates and editing, run on the Halo host in a separate terminal:

```bash
cd ~/capstone-training/mars-pilot50-v2
conda activate rover-yolo
PYTHONPATH="$PWD/src/mars_training_environment:$PWD/src/rover_training" \
  python -m mars_training_environment.review --dataset dataset/pilot50_v2
```

In VS Code's **Ports** panel, forward port **8765** and open its browser link.
Alternatively, use a separate local PowerShell terminal:

```powershell
ssh -N -L 8765:127.0.0.1:8765 ian@halo-strix
```

Open `http://127.0.0.1:8765` on Windows. The page refreshes every five seconds
and supports scene/split/class/status filtering, size comparisons, target and
failure information, drawing/removing boxes, class and coordinate edits,
notes, reversible exclusions, and restoration of original labels.

Corrections are recorded in sidecars; the original capture is preserved.
**Export reviewed training data** writes the corrected, included frames to
a new revision folder, keeping scene splits. It returns the actual host
`data.yaml` path. Excluded frames reduce the export count. Export is available
after raw capture validation completes, and each split must retain an image.

To review the original pilot, use the same command with:

```bash
--dataset ~/capstone-training/mars-pilot50/dataset/pilot50
```

Legacy frames are flagged as lacking the new visibility audit.

After training, optional YOLO predictions in
`predictions/<split>/<frame>.txt` appear as dashed boxes and provide wrong-class,
missing-detection, and extra-detection review hints. These files are predictions
only; editing ground truth does not overwrite them.

## After reviewing, train in Conda

Use the returned reviewed-export YAML and the same pretrained model selected
for the earlier smoke test. The preset uses 960 pixels, batch 4, 30 epochs,
and a fixed seed; adjust batch size based on measured GPU memory use.

```bash
conda activate rover-yolo
yolo train cfg=src/mars_training_environment/config/train960.yaml \
  model=/path/to/pretrained-model.pt \
  data=/path/returned/by/review/export/data.yaml device=0
```

The review/export process and training remain separate actions. No capture or
training was started automatically when implementing this revision.

## Resume after the scene 102 invisible-label failure

The October 12 capture completed scene 101, then stopped because scene 102/view
001 included a sky-only `rock_006` label. Apply the small source patch, rebuild,
and resume the existing plan. No scene generation is needed for this fix.

In Windows PowerShell, run this single command:

```powershell
& 'C:\Windows\System32\OpenSSH\scp.exe' C:\Projects\Capstone\halo-pilot50-v2-visibility-fix-20261012.tar.gz ian@halo-strix:~/
```

On the Halo, extract the source patch:

```bash
tar -xzf ~/halo-pilot50-v2-visibility-fix-20261012.tar.gz -C ~/capstone-training/mars-pilot50-v2
```

Run the build command above from `~/capstone-training/mars-pilot50-v2`, then:

```bash
sudo docker run -d --name mars-pilot50-v2-resume --network none --shm-size=1g \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e LIBGL_ALWAYS_SOFTWARE=1 \
  -v "$PWD:/work" \
  -v "$HOME/capstone-training/mars-pilot50/assets_cache:/assets_cache:ro" \
  -w /work rover-jazzy bash -c '
    set -e
    source /opt/ros/jazzy/setup.bash
    source /work/install_jazzy/setup.bash
    ros2 run mars_training_environment mars_dataset \
      --resume \
      --bundle /work/dataset/pilot50_scenes_v2 \
      --output /work/dataset/pilot50_v2
  '
```

```bash
sudo docker logs --follow --tail 50 mars-pilot50-v2-resume
```

Resume keeps completed scene 101 and recaptures the incomplete scene 102. The
old failed container can remain for its logs. Restart the review server if it
was already running so that it uses the updated source. If capture fails again,
the outer log now includes the capture node's actual exception; preserve its
diagnostics for investigation.

## Resume after the scene 102 target-size failure

The visibility patch got past view 001. View 004 then exposed a planning
resolution bug: its target was estimated at 12.5 pixels at training size 960,
but measured only 11 pixels at capture resolution, below the difficult-view
minimum of 12. The framing patch adds conservative coarse bounds and a final
full-resolution planning check. Existing bundles can recover a size-only
failure by moving closer within the original distance/collision/slope limits,
up to three times. Every recaptured frame must pass all normal checks. The
original plan and completed scene 101 are preserved; do not regenerate them.
The original scene 102 passed a local rendered regression with all five frames
and `errors: []`; view 004 measured 13 pixels after one correction. It retains
one renderer-box disagreement for manual review. The complete Halo run must
still be verified after deployment.

Copy the new patch from Windows PowerShell:

```powershell
& 'C:\Windows\System32\OpenSSH\scp.exe' C:\Projects\Capstone\halo-pilot50-v2-framing-fix-20261012.tar.gz ian@halo-strix:~/
```

Extract on the Halo:

```bash
tar -xzf ~/halo-pilot50-v2-framing-fix-20261012.tar.gz -C ~/capstone-training/mars-pilot50-v2
```

```bash
cd ~/capstone-training/mars-pilot50-v2
```

Run the build command above, wait for all three packages to finish, then run
this command. It uses a new container name because the previous resume
container still exists.

```bash
sudo docker run -d --name mars-pilot50-v2-framing --network none --shm-size=1g \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e LIBGL_ALWAYS_SOFTWARE=1 \
  -v "$PWD:/work" \
  -v "$HOME/capstone-training/mars-pilot50/assets_cache:/assets_cache:ro" \
  -w /work rover-jazzy bash -c '
    set -e
    source /opt/ros/jazzy/setup.bash
    source /work/install_jazzy/setup.bash
    ros2 run mars_training_environment mars_dataset \
      --resume \
      --bundle /work/dataset/pilot50_scenes_v2 \
      --output /work/dataset/pilot50_v2
  '
```

```bash
sudo docker logs --follow --tail 50 mars-pilot50-v2-framing
```

Capture-node logs include the measured size and any reframing attempt.
Accepted-frame metadata records the original plan, actual pose and correction
history. Failed correction attempts remain in diagnostics subfolders and in
the review page. After the run completes, verify `errors: []`, `frames: 50`,
and split counts of 30 train / 10 val / 10 test:

```bash
cat /home/ian/capstone-training/mars-pilot50-v2/dataset/pilot50_v2/validation.json
```

Restart the review server using the earlier instructions so it loads the new
code. Check `target_reframed` flags along with the invisible-label flags,
review the remaining frames, and export the corrected dataset.

## Resume after the scene 104 ground-patch failure

The Halo framing retry completed scenes 102 and 103, then stopped at scene
104/view 000 on `pale_lookalike_000`. The primary `silica_soil_001` target was
valid. The incidental patch was beyond the RGB-D sensor's radial 8 m limit,
even though its forward depth was below 8 m. Ogre2 clips by total distance at
the far boundary and sets those pixels to background RGB/infinite depth; the
annotation rasterizer previously used only forward depth.

The ground patch now matches the sensor's radial far clipping for all mesh
and terrain pixels. Painted ground labels also come from actual terrain ray
hits and the paint mask, replacing lifted proxy meshes. Missing sensor data,
valid-but-wrong depth, primary-target failures, and RGB sky checks still fail
capture. Existing scene plans and completed scenes 101–103 can be resumed.
The original scene 104 passed a local rendered regression with all five
images and `errors: []`. Its primary ground patch had 100% depth and paint
support; the invisible incidental patch was not labeled. The subsequent Halo
retry completed all 50 images with the expected splits and no validation errors.

Transfer the cumulative patch from Windows PowerShell:

```powershell
& 'C:\Windows\System32\OpenSSH\scp.exe' C:\Projects\Capstone\halo-pilot50-v2-ground-fix-20261012.tar.gz ian@halo-strix:~/
```

On the Halo, extract it:

```bash
tar -xzf ~/halo-pilot50-v2-ground-fix-20261012.tar.gz -C ~/capstone-training/mars-pilot50-v2
```

```bash
cd ~/capstone-training/mars-pilot50-v2
```

Run the three-package build command above and wait for success, then:

```bash
sudo docker run -d --name mars-pilot50-v2-ground --network none --shm-size=1g \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e LIBGL_ALWAYS_SOFTWARE=1 \
  -v "$PWD:/work" \
  -v "$HOME/capstone-training/mars-pilot50/assets_cache:/assets_cache:ro" \
  -w /work rover-jazzy bash -c '
    set -e
    source /opt/ros/jazzy/setup.bash
    source /work/install_jazzy/setup.bash
    ros2 run mars_training_environment mars_dataset \
      --resume \
      --bundle /work/dataset/pilot50_scenes_v2 \
      --output /work/dataset/pilot50_v2
  '
```

```bash
sudo docker logs --follow --tail 50 mars-pilot50-v2-ground
```

Do not regenerate the bundle. Completed scenes 101, 102 and 103 should be
skipped. After capture completes, check the validation report for `errors: []`,
50 frames and 30/10/10 splits. Restart the review server to load the new code.
Earlier accepted frames with ground labels carry
`legacy_ground_region_projection` flags for manual inspection; this flag alone
does not block export or alter their stored labels.

Ogre2's shader source:
https://github.com/gazebosim/gz-rendering/blob/gz-rendering8/ogre2/src/media/materials/programs/GLSL/depth_camera_fs.glsl
