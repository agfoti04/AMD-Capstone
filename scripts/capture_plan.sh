#!/usr/bin/env bash
# Save every scene in the seeded plan. Run this inside the Humble container,
# after: source /opt/ros/humble/setup.bash && source install/setup.bash
set -euo pipefail

SEED="${1:-7}"
OUT="${2:-$PWD/dataset/rocks}"

export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"

mapfile -t SCENE_IDS < <(ros2 run rover_training list_scenes -- --seed "${SEED}")
if [[ ${#SCENE_IDS[@]} -eq 0 ]]; then
  echo "list_scenes printed no ids" >&2
  exit 1
fi

for scene_id in "${SCENE_IDS[@]}"; do
  echo "=== ${scene_id} ==="
  xvfb-run -a ros2 launch rover_training capture_scene.launch.py \
    gui:=false \
    seed:="${SEED}" \
    scene_id:="${scene_id}" \
    output_dir:="${OUT}"
done

ros2 run rover_training preview_labels -- --dataset "${OUT}"
echo "Images and previews are under ${OUT}"
