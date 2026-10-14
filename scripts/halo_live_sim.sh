#!/usr/bin/env bash
# Run from an Ubuntu desktop terminal on the Halo after capture/training finish.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
scene_rel="${1:-dataset/live_eval_scenes/scenes/scene_050000/scene.json}"
camera_timeout="${ROVER_CAMERA_TIMEOUT:-3.0}"
scene_abs="$(realpath -- "$project_dir/$scene_rel")"
if [[ "$scene_abs" != "$project_dir/dataset/live_eval_scenes/"*/scene.json ]]; then
  echo 'Choose a scene.json inside dataset/live_eval_scenes (a separate evaluation bundle).' >&2
  exit 1
fi
[[ -f "$scene_abs" ]] || { echo 'Generate dataset/live_eval_scenes first; see HALO_LIVE_VISION.md.' >&2; exit 1; }
[[ -n "${DISPLAY:-}" ]] || { echo 'Run this in a terminal on the remotely viewed Ubuntu desktop (DISPLAY is missing).' >&2; exit 1; }
command -v xauth >/dev/null || { echo 'Install xauth on the Halo desktop: sudo apt-get install xauth' >&2; exit 1; }
task_xauth="$(mktemp /tmp/rover-live-xauth.XXXXXX)"
trap 'rm -f -- "$task_xauth"' EXIT
# Copy only the current desktop cookie. Do not globally open the X server with xhost +.
xauth nlist "$DISPLAY" | sed 's/^..../ffff/' | xauth -f "$task_xauth" nmerge -
[[ -n "$(xauth -f "$task_xauth" nlist)" ]] || { echo 'No X11 cookie found; run inside the active Ubuntu desktop session.' >&2; exit 1; }
scene_container="/work/${scene_abs#"$project_dir/"}"
sudo docker run --rm --init -it --name rover-live-sim --network host --shm-size=1g \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e DISPLAY -e XAUTHORITY=/tmp/rover-live.xauth \
  -e QT_X11_NO_MITSHM=1 -e QT_QPA_PLATFORM=xcb -e LIBGL_ALWAYS_SOFTWARE=1 \
  -e ROS_DOMAIN_ID=86 -e ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
  -e GZ_PARTITION="rover_live_$(id -u)" -e LIVE_SCENE="$scene_container" \
  -e LIVE_CAMERA_TIMEOUT="$camera_timeout" \
  -v "$project_dir:/work" -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
  -v "$task_xauth:/tmp/rover-live.xauth:ro" \
  -w /work rover-jazzy bash -c '
    set -e
    source /opt/ros/jazzy/setup.bash
    source /work/install_jazzy/setup.bash
    exec ros2 launch mars_training_environment mars_live.launch.py \
      scene:="$LIVE_SCENE" camera_timeout:="$LIVE_CAMERA_TIMEOUT"
  '
