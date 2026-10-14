#!/usr/bin/env bash
# Browser driving defaults to headless RGB rendering; run from any Halo terminal.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
scene_rel="${1:-dataset/live_eval_scenes/scenes/scene_050000/scene.json}"
control_mode="${2:-browser}"
[[ "$control_mode" == browser || "$control_mode" == gazebo ]] || { echo 'Control mode must be browser or gazebo.' >&2; exit 1; }
gui="${ROVER_GUI:-false}"
[[ "$gui" == true || "$gui" == false ]] || { echo 'ROVER_GUI must be true or false.' >&2; exit 1; }
renderer="${ROVER_RENDERER:-auto}"
[[ "$renderer" == auto || "$renderer" == software || "$renderer" == gpu ]] || { echo 'ROVER_RENDERER must be auto, software or gpu.' >&2; exit 1; }
if [[ "$renderer" == auto ]]; then
  renderer=software
  for device in /dev/dri/renderD*; do
    if [[ -e "$device" ]]; then renderer=gpu; break; fi
  done
fi
scene_abs="$(realpath -- "$project_dir/$scene_rel")"
if [[ "$scene_abs" != "$project_dir/dataset/live_eval_scenes/"*/scene.json ]]; then
  echo 'Choose a scene.json inside dataset/live_eval_scenes.' >&2
  exit 1
fi
[[ -f "$scene_abs" ]] || { echo 'Generate dataset/live_eval_scenes first; see HALO_LIVE_VISION.md.' >&2; exit 1; }
docker_args=(--rm --init -it --name rover-live-sim --network host --shm-size=1g
  --user "$(id -u):$(id -g)" -e HOME=/tmp
  -e ROS_DOMAIN_ID=86 -e ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
  -e GZ_PARTITION="rover_live_$(id -u)" -e LIVE_SCENE="/work/${scene_abs#"$project_dir/"}"
  -e LIVE_CAMERA_TIMEOUT="${ROVER_CAMERA_TIMEOUT:-3.0}" -e LIVE_CONTROL_MODE="$control_mode"
  -e LIVE_GUI="$gui" -e LIVE_WIDTH="${ROVER_LIVE_WIDTH:-960}" -v "$project_dir:/work" -w /work)
if [[ "$renderer" == gpu ]]; then
  [[ -d /dev/dri ]] || { echo 'No /dev/dri device found; use ROVER_RENDERER=software.' >&2; exit 1; }
  docker_args+=(--device /dev/dri -e LIBGL_ALWAYS_SOFTWARE=0)
  for device in /dev/dri/renderD* /dev/dri/card*; do
    [[ -e "$device" ]] && docker_args+=(--group-add "$(stat -c %g "$device")")
  done
else
  docker_args+=(-e LIBGL_ALWAYS_SOFTWARE=1)
  echo 'Software rendering selected: CPU camera rendering can severely limit FPS.' >&2
fi
if [[ "$gui" == true ]]; then
  [[ -n "${DISPLAY:-}" ]] || { echo 'ROVER_GUI=true requires a terminal on the Ubuntu desktop.' >&2; exit 1; }
  command -v xauth >/dev/null || { echo 'Install xauth: sudo apt-get install xauth' >&2; exit 1; }
  task_xauth="$(mktemp /tmp/rover-live-xauth.XXXXXX)"
  trap 'rm -f -- "$task_xauth"' EXIT
  xauth nlist "$DISPLAY" | sed 's/^..../ffff/' | xauth -f "$task_xauth" nmerge -
  [[ -n "$(xauth -f "$task_xauth" nlist)" ]] || { echo 'No X11 cookie found.' >&2; exit 1; }
  docker_args+=(-e DISPLAY -e XAUTHORITY=/tmp/rover-live.xauth -e QT_X11_NO_MITSHM=1 -e QT_QPA_PLATFORM=xcb
    -v /tmp/.X11-unix:/tmp/.X11-unix:ro -v "$task_xauth:/tmp/rover-live.xauth:ro")
fi
echo "Live browser drive: ${ROVER_LIVE_WIDTH:-960}px RGB, renderer=$renderer, GUI=$gui"
sudo docker run "${docker_args[@]}" rover-jazzy bash -c '
  set -e
  source /opt/ros/jazzy/setup.bash
  source /work/install_jazzy/setup.bash
  python3 -c "from rover_perception.live_core import LIVE_PIPELINE_VERSION; from mars_training_environment.robot import live_camera_config; assert LIVE_PIPELINE_VERSION == 2" || {
    echo "Installed live packages are outdated. Extract with tar --touch, rebuild both packages and restart." >&2
    exit 1
  }
  exec ros2 launch mars_training_environment mars_live.launch.py \
    scene:="$LIVE_SCENE" camera_timeout:="$LIVE_CAMERA_TIMEOUT" control_mode:="$LIVE_CONTROL_MODE" \
    gui:="$LIVE_GUI" live_width:="$LIVE_WIDTH"
'
