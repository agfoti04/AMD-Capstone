# Browser driving with live object detection

Use the browser for both driving and camera predictions. The simulator now runs
without a Gazebo window by default. The live camera renders RGB at 960 x 540
and 30 Hz, with the original field of view and mounting pose. Depth and simulator
bounding-box sensors are disabled only for live driving. Saved scene JSON,
training images, capture settings and trained weights are preserved.

The dashboard targets up to 30 processed frames per wall second. Actual FPS
depends on Gazebo rendering, simulation speed, GPU inference and the network.
The previous dashboard capped processing at 8 FPS and rendered all capture
sensors at full resolution. The new input uses rectangular inference at
`imgsz=960`; `--square` restores square padding for a comparison. Live imagery
and these prediction settings are distinct from the held-out dataset benchmark.

Driving uses a persistent WebSocket with commands every 50 ms while the page
is active. Key changes and release send immediately, without waiting for an
earlier acknowledgment. A separate video connection sends a new frame only
after the browser displays the previous one, skipping intermediate frames on
slow links. One browser tab owns the driving connection at a time.

## Deploy the update

Release controls and press Ctrl+C in the existing vision and simulator
terminals. Keep the Windows SSH tunnel open if it is already running.

In Windows PowerShell:

```powershell
scp C:\Projects\Capstone\halo-browser-drive-20261013.tar.gz ian@halo-strix:~/
```

On the Halo:

```bash
cd /home/ian/capstone-training/mars-pilot50-v2
```

```bash
tar -xzf /home/ian/halo-browser-drive-20261013.tar.gz -C /home/ian/capstone-training/mars-pilot50-v2
```

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
yellow_crystal_rock. No depth or simulator labels are fed into inference.

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

The default keeps the working software-rendering setup while removing the GUI,
depth sensor and bounding-box sensor and reducing the rendered pixel count.
The strongest further improvement may be GPU rendering if the Docker image's
Mesa driver supports the Halo GPU. Stop the simulator first, then try:

```bash
ROVER_RENDERER=gpu bash /home/ian/capstone-training/mars-pilot50-v2/scripts/halo_live_sim.sh \
  dataset/live_eval_scenes/scenes/scene_050001/scene.json browser
```

This exposes `/dev/dri` with the device groups needed by the container user.
If EGL/Ogre fails or no images arrive, stop and use the ordinary launch command
again. GPU rendering and live performance still require verification on the Halo.

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

The bridge should report `control_mode: browser`. The vision status should show
`image_size: [960, 540]` with the default profile. If controls keep reconnecting,
close other driving tabs, inspect the displayed command latency, and check that
the SSH tunnel and both Halo terminals remain connected. If no images arrive:

```bash
sudo docker logs --tail 100 rover-live-sim
```

Local checks cover real WebSocket/HTTP command delivery, held/released input,
speed limits, disconnection expiry, slow-viewer backpressure, browser keyboard
and slider behavior, and real Xacro expansion for live/capture sensor profiles.
They use controlled camera/model data. Actual Gazebo/ROCm FPS must be measured
on the Halo after deployment.

References: [Gazebo headless rendering](https://gazebosim.org/api/sim/8/headless_rendering.html),
[Ultralytics prediction settings](https://docs.ultralytics.com/modes/predict/),
[aiohttp WebSockets](https://docs.aiohttp.org/en/stable/web_quickstart.html#websockets).
