# Drive a new Mars scene with live model predictions

Use the Halo's existing remote Ubuntu desktop. Gazebo shows the world and rover;
a browser at `http://127.0.0.1:8771` shows the camera with predicted boxes,
three class names, confidence, camera age and inference timing. Hold W/A/S/D,
arrow keys or direction buttons to drive; release or press Space/Stop to stop.
The browser must have focus for keyboard controls.

This is manual driving with live inference, not autonomous navigation. The live
view uses only RGB as model input. Depth and simulator bounding boxes are not
fed to the detector. No ground-truth accuracy score is computed by this view.

## Wait for capture and training to finish

Prepare the files on Windows now. Transfer/extract and rebuild on the Halo after
the current capture finishes, so its mounted source/install files stay stable.
Start the live simulator after training to avoid competing for CPU/GPU memory.
The launcher uses ROS domain 86 and a separate Gazebo partition, and it runs in
its own `rover-live-sim` container. It never stops the capture container.

## Deploy the patch

In Windows PowerShell:

```powershell
& 'C:\Windows\System32\OpenSSH\scp.exe' C:\Projects\Capstone\halo-live-vision-20261013.tar.gz ian@halo-strix:~/
```

On the Halo, after capture completes:

```bash
tar -xzf ~/halo-live-vision-20261013.tar.gz -C ~/capstone-training/mars-pilot50-v2
```

```bash
cd ~/capstone-training/mars-pilot50-v2
```

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

Require all four packages to finish. The Docker bridge uses ROS/OpenCV packages
already needed for capture. Inference uses the existing host `rover-yolo`
environment, so there is no need to reinstall PyTorch or ROCm inside Docker.

## Generate separate evaluation scenes once

The provided profile creates three mixed scenes with seeds 50000–50002, distinct
from the pilot and coverage500 seeds. Keep them out of training exports. These
are new layouts; mesh holdouts still follow the generator's seed-based split.
Scene files are generated in a separate bundle because the launch rewrites its
world paths, robot description and bridge configuration.

```bash
sudo docker run --rm --init --network none \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD:/work" \
  -v "$HOME/capstone-training/mars-pilot50/assets_cache:/assets_cache:ro" \
  -w /work rover-jazzy bash -c '
    set -e
    source /opt/ros/jazzy/setup.bash
    source /work/install_jazzy/setup.bash
    exec ros2 run mars_training_environment mars_generate \
      --config /work/src/mars_training_environment/config/live_eval.json \
      --asset-cache /assets_cache \
      --output /work/dataset/live_eval_scenes
  '
```

This plans viewpoints as a scene feasibility check; the live launch does not
execute them or capture training data. The rover starts in the clear spawn area
and moves only when you send driving commands.

## Start Gazebo on the remote Ubuntu desktop

Open a terminal **inside the Ubuntu desktop you are remotely viewing**. A normal
SSH-only shell generally has no desktop DISPLAY/cookie and cannot open this GUI.
The launcher shares the desktop's current X11/XWayland authentication cookie;
it does not require `xhost +`. If xauth is missing, install it with
`sudo apt-get install xauth`, then rerun the launcher.

```bash
cd ~/capstone-training/mars-pilot50-v2
```

```bash
bash scripts/halo_live_sim.sh
```

Leave this terminal running. Wait for Gazebo, the rover controllers and the
camera bridge. Software rendering is the initial preset to match the working
capture setup; simulation speed can be below real time. Host GPU inference is
separate. This preset does not claim a real-time end-to-end frame rate.

To choose another scene, stop this terminal with Ctrl+C and run:

```bash
bash scripts/halo_live_sim.sh dataset/live_eval_scenes/scenes/scene_050001/scene.json
```

## Start vision with your trained weights

In a second Halo terminal:

```bash
cd ~/capstone-training/mars-pilot50-v2
```

```bash
conda activate rover-yolo
```

Select the exact checkpoint printed by your training run. To try the already
completed 50-image baseline first:

```bash
export ROVER_LIVE_MODEL="$HOME/capstone-training/mars-pilot50-v2/runs/detect/runs/detect/pilot50_v2_960/weights/best.pt"
```

After balanced training, replace that value with its `best.pt` path.

```bash
bash scripts/halo_live_vision.sh "$ROVER_LIVE_MODEL" --scene scene_050000
```

Open `http://127.0.0.1:8771` in the **Halo desktop browser**. Alternatively, tunnel
the dashboard from Windows and use your local browser:

```powershell
ssh -N -L 8771:127.0.0.1:8771 ian@halo-strix
```

The dashboard defaults to `device=0`, `imgsz=960`, `rect=False`, confidence 0.25
and at most 8 processed frames per wall second. It fetches the newest frame
instead of building a camera queue. The model input is a lossless native-size
PNG decoded to BGR; only the annotated browser preview is JPEG. Class names
come from the checkpoint, which must contain the three rover classes.

If necessary, override settings without editing any source:

```bash
bash scripts/halo_live_vision.sh "$ROVER_LIVE_MODEL" --scene scene_050000 --conf 0.35 --fps 5
```

The bridge clamps movement to 0.30 m/s and 0.80 rad/s. Browser release, blur and
hidden-page events request a stop. Commands expire after 0.35 seconds without
a heartbeat; stale camera data also stops motion. A paused simulator therefore
cannot preserve a drive command indefinitely. On a disconnected page, release
the physical keys and press them again after reconnection.

## Optionally record evidence

Restart the vision command with a new recording directory:

```bash
bash scripts/halo_live_vision.sh "$ROVER_LIVE_MODEL" \
  --scene scene_050000 \
  --record-dir "dataset/live_sessions/scene_050000_$(date +%Y%m%d_%H%M%S)"
```

Each processed frame saves a raw PNG and annotated JPEG. `predictions.jsonl`
stores simulation timestamps, classes, confidence and pixel boxes; `session.json`
stores the weights path/hash, scene ID and inference settings. Existing session
directories are rejected to prevent overwriting a prior recording. Recording
can reduce throughput and use substantial disk space; it is off by default.
These files are observational evidence, not reviewed training labels.

Approach soil/crystals from several angles, vary distance, pass behind rocks
and turn through shadows. Watch for missed targets, wrong classes, extra boxes
and flicker. Compare model versions on the same scene/route/settings. Manual
routes are not automatically repeatable; formal accuracy evaluation still uses
the held-out dataset and reviewed ground truth.

## Stop and troubleshoot

Release controls, press Stop, then Ctrl+C in the vision terminal and the Gazebo
terminal. If the GUI terminal was lost, stop only the live simulator:

```bash
sudo docker stop rover-live-sim
```

- Waiting for camera: make sure Gazebo is playing, controllers have loaded and
  `http://127.0.0.1:8770/status` reports a changing sequence and recent camera age.
- Port in use: close the prior live session; ports 8770/8771 are independent of
  the dataset review ports 8765/8766.
- GUI cannot open: rerun inside the remote Ubuntu desktop terminal and check
  DISPLAY/X11 cookie access. The launcher's desktop checks explain missing access.
- GPU inference fails: verify `rover-yolo` is active; `--device cpu --fps 2` is an
  explicit slower fallback for diagnosing the display/bridge separately.

The 12 live camera/control tests passed on Windows and Linux. They cover
color/stride handling, all-class output, HTTP inference and controls, stale
frames, command expiry and recording, using controlled ROS/model doubles.
The shipped browser script also passed a DOM/event harness for keyboard,
pointer, heartbeat, release, blur, hidden-page and reconnection behavior.
Bash syntax checks and a package build passed. All three evaluation seeds
were planned locally with the production generator; the 30-view quota audit
returned `errors: []`. The local Browser connection and Docker were
unavailable; a complete GUI + real-checkpoint + Halo GPU test remains required
on the actual Halo after training.

The inference options follow [Ultralytics' prediction documentation](https://docs.ultralytics.com/modes/predict/).
