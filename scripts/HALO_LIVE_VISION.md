# Browser driving with live object detection

Use the browser for both driving and camera predictions. The simulator now runs
without a Gazebo window by default. The live camera renders RGB at 960 x 540
and 30 Hz, with the original field of view and mounting pose. Depth and simulator
bounding-box sensors are disabled only for live driving. Saved scene JSON,
training images, capture settings and trained weights are preserved.
The launcher defaults to `ROVER_RENDERER=auto`: it exposes the GPU and its
device groups when a DRM render node is present, otherwise it reports a
software-rendering fallback. GPU use by YOLO alone does not accelerate Gazebo.
Successful GPU rendering depends on the container's Mesa/EGL support.

The dashboard targets up to 30 processed frames per wall second. Actual FPS
depends on Gazebo rendering, simulation speed, GPU inference and the network.
The previous dashboard capped processing at 8 FPS and rendered all capture
sensors at full resolution. The new input uses rectangular inference at
`imgsz=960`; `--square` restores square padding for a comparison. Live imagery
and these prediction settings are distinct from the held-out dataset benchmark.

Driving uses a persistent WebSocket with commands every 50 ms while the page
is active. Key changes and release send immediately, without waiting for an
earlier acknowledgment. A separate video connection pipelines up to three
frames awaiting display, so every frame need not wait for a network round trip.
It stops sending when that window is full and resumes with the latest frame,
skipping intermediate frames on slow links. One browser tab owns driving.

The local camera-to-detector link now sends lossless BGR pixels directly over
localhost, avoiding PNG compression and decompression on every inference.
Only the annotated browser image is JPEG-encoded. Recording still saves raw
PNG images. Status includes camera-fetch and annotation/encoding times.

## Deploy the update

Release controls and press Ctrl+C in the existing vision and simulator
terminals. Keep the Windows SSH tunnel open if it is already running.

In Windows PowerShell:

```powershell
scp C:\Projects\Capstone\halo-browser-fps-20261014.tar.gz ian@halo-strix:~/
```

On the Halo:

```bash
cd /home/ian/capstone-training/mars-pilot50-v2
```

```bash
tar --touch -xzf /home/ian/halo-browser-fps-20261014.tar.gz -C /home/ian/capstone-training/mars-pilot50-v2
```

Use `--touch` even when deploying over existing files. The earlier archive had
epoch timestamps, which allowed setuptools incremental builds to reuse older
modules. This archive has current timestamps; `--touch` also uses the Halo's
clock so source files are newer than its existing build products. The launcher
now rejects an outdated installed live package rather than silently running it.

Copy the complete build block, including its closing quote:

```bash
sudo docker run --rm --init \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD:/work" -w /work rover-jazzy bash -c '
    set -e
    source /opt/ros/jazzy/setup.bash
    colcon --log-base log_jazzy build --base-paths src \
      --build-base build_jazzy --install-base install_jazzy \
      --packages-select four_wheel_sim rover_training rover_perception mars_training_environment
  '
```

Require all four packages to finish successfully. The archive includes the
complete perception package and the updated live/world launch and robot files.

Install the small browser-server dependency in the existing host environment:

```bash
source /home/ian/miniforge3/etc/profile.d/conda.sh
```

```bash
conda activate rover-yolo
```

```bash
python -m pip install -r scripts/requirements-live.txt
```

This adds aiohttp for WebSockets; it does not request a Torch, ROCm or
Ultralytics upgrade. Inference still uses the existing host GPU environment.

## Start the simulator

The three evaluation scenes already generated on the Halo can be reused.
In the first Halo terminal, including an SSH terminal:

```bash
bash /home/ian/capstone-training/mars-pilot50-v2/scripts/halo_live_sim.sh \
  dataset/live_eval_scenes/scenes/scene_050001/scene.json browser
```

Leave this terminal running. No Gazebo window is expected. The simulator runs
in `rover-live-sim`, ROS domain 86, with the existing separate Gazebo partition.

To change scenes, release controls, stop this terminal with Ctrl+C and rerun
the command with `scene_050000` or `scene_050002` in the path. The launcher
accepts only the separate `dataset/live_eval_scenes` bundle.

If that bundle has not yet been generated, run this once from the project root:

```bash
sudo docker run --rm --init --network none \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD:/work" \
  -v "$HOME/capstone-training/mars-pilot50/assets_cache:/assets_cache:ro" \
  -w /work rover-jazzy bash -c '
    set -e
    source /opt/ros/jazzy/setup.bash
    source /work/install_jazzy/setup.bash
    ros2 run mars_training_environment mars_generate \
      --config /work/src/mars_training_environment/config/live_eval.json \
      --asset-cache /assets_cache --output /work/dataset/live_eval_scenes
  '
```

## Start detection and open the browser

In a second Halo terminal:

```bash
source /home/ian/miniforge3/etc/profile.d/conda.sh
```

```bash
conda activate rover-yolo
```

```bash
bash /home/ian/capstone-training/mars-pilot50-v2/scripts/halo_live_vision.sh \
  /home/ian/capstone-training/mars-pilot50-v2/runs/detect/coverage500_balanced_960/weights/best.pt \
  --scene scene_050001
```

Leave it running. The scene option records provenance; the simulator launch
selects the actual scene. The weights must contain rock, pale_soil_patch and
yellow_crystal_rock, with optional mudstone_slab at ID 3. The
[four-class workflow](HALO_FOUR_CLASS.md) reuses reviewed captures to train
that extension. No depth or simulator labels are fed into inference.

On Windows, reuse the existing tunnel or open one PowerShell terminal with:

```powershell
ssh -N -L 8771:127.0.0.1:8771 ian@halo-strix
```

Open **http://127.0.0.1:8771** and refresh with Ctrl+F5 after an update.
The same address works directly in a Halo desktop browser without a tunnel.

## Drive and adjust speed

- Hold W/S or up/down arrows to move forward/backward; A/D or left/right arrows
  turn. Buttons also support hold and release.
- Drive speed starts at **0.30 m/s** and adjusts from **0.10 to 0.60 m/s**.
- Turn speed starts at **0.80 rad/s** and adjusts from **0.20 to 1.50 rad/s**.
- Sliders apply while driving. These upper limits match the configured rover
  controller values and are enforced in the backend as well as the UI.
- Release to stop. Space, Stop, loss of window focus or hiding the page clears
  held inputs. After reconnecting, release and press the keys again.
- Keep a single driving tab open. A second tab cannot override the driver.

The bridge publishes commands at 50 Hz and expires them after 0.35 wall seconds
without fresh browser input. The browser disconnects controls after a 0.5-second
acknowledgment stall. The displayed camera must also remain within its three-second
freshness limit. A short rendering gap does not clear held controls, while a
prolonged gap, failed inference or a lost video connection stops movement.

The dashboard shows measured display FPS, inference time, control round-trip
time, displayed frame age and simulation speed. Control latency measures the
command acknowledgment, not the time until the rover visibly moves. A simulation
speed of `0.50x` means one simulation second takes two wall seconds, so the
rover will also appear to move more slowly.

## Performance options

Automatic rendering prefers the GPU; the terminal should report `renderer=gpu`
on the Halo. This requests hardware rendering; confirm camera output and FPS
because Mesa may still fall back to software. To require exposing the GPU:

```bash
ROVER_RENDERER=gpu bash /home/ian/capstone-training/mars-pilot50-v2/scripts/halo_live_sim.sh \
  dataset/live_eval_scenes/scenes/scene_050001/scene.json browser
```

This exposes `/dev/dri` with the device groups needed by the container user.
If EGL/Ogre fails or no images arrive, inspect the simulator logs. A temporary
software fallback at a smaller camera resolution is:

```bash
ROVER_RENDERER=software ROVER_LIVE_WIDTH=640 bash /home/ian/capstone-training/mars-pilot50-v2/scripts/halo_live_sim.sh \
  dataset/live_eval_scenes/scenes/scene_050001/scene.json browser
```

Software rendering can remain slow. A container whose Mesa driver cannot
support this GPU needs a graphics-driver/image update before hardware rendering
will help; switching inference libraries does not fix that rendering problem.

For less rendering work, restart the simulator with a 640 x 360 camera:

```bash
ROVER_LIVE_WIDTH=640 bash /home/ian/capstone-training/mars-pilot50-v2/scripts/halo_live_sim.sh \
  dataset/live_eval_scenes/scenes/scene_050001/scene.json browser
```

Restart detection with `--imgsz 640` appended as well. This trades small/distant
object detail for throughput. Supported camera widths are 640, 960, 1280 and
1920. Keep the default 960 for the first comparison.

For the optional Gazebo window, set `ROVER_GUI=true` on the simulator command
from an Ubuntu desktop terminal. This requires DISPLAY and xauth. Browser
driving still works; the extra window adds rendering work.

## Record a test and troubleshoot

Append a fresh directory to the vision command to save raw PNGs, annotated
JPEGs, prediction rows and weights/config provenance:

```bash
--record-dir "dataset/live_sessions/scene_050001_$(date +%Y%m%d_%H%M%S)"
```

This is an extra argument to the vision command, not a separate shell command.
Recording can reduce throughput. These observations are not reviewed training
labels or a ground-truth accuracy evaluation.

For status from another Halo terminal:

```bash
curl --max-time 5 -sS http://127.0.0.1:8770/status
```

```bash
curl --max-time 5 -sS http://127.0.0.1:8771/status
```

The bridge and vision status should both report `pipeline_version: 2`, and both
should show `image_size: [960, 540]` with the default profile. A 1920 x 1080 feed
or missing bridge version indicates an old simulator or installed package:
stop it, extract with `--touch`, rebuild and restart. The October 14 report of
approximately 10 ms inference, 1.38 s camera age and 950% container CPU points
to simulation/rendering or camera delivery, rather than slow model inference.
The reported 0.65 processed FPS and 0.32 simulation speed are also consistent
with a low-rate capture camera instead of the intended 30-Hz live profile.

To verify the installed bridge before starting simulation:

```bash
sudo docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD:/work" -w /work rover-jazzy bash -c '
    source /opt/ros/jazzy/setup.bash
    source /work/install_jazzy/setup.bash
    python3 -c "from rover_perception.live_core import LIVE_PIPELINE_VERSION; import rover_perception.live_bridge as b; print(b.__file__); print(LIVE_PIPELINE_VERSION)"
  '
```

It should print the installed bridge path and `2`. If controls keep reconnecting,
close other driving tabs, inspect the displayed command latency, and check that
the SSH tunnel and both Halo terminals remain connected. If no images arrive:

```bash
sudo docker logs --tail 100 rover-live-sim
```

Local checks cover real WebSocket/HTTP command delivery, lossless raw pixel
transport, held/released input, speed limits, disconnection expiry, bounded
pipelined video, browser keyboard
and slider behavior, and real Xacro expansion for live/capture sensor profiles.
They use controlled camera/model data. Actual Gazebo/ROCm FPS must be measured
on the Halo after deployment. Archive verification now checks updates over
stale build modules as well as comparing installed wheel contents with source.

References: [Gazebo headless rendering](https://gazebosim.org/api/sim/8/headless_rendering.html),
[Ultralytics prediction settings](https://docs.ultralytics.com/modes/predict/),
[aiohttp WebSockets](https://docs.aiohttp.org/en/stable/web_quickstart.html#websockets),
[Mesa rendering environment variables](https://docs.mesa3d.org/envvars.html).
