#!/usr/bin/env bash
# Use the existing rover-yolo Conda environment, keeping its ROCm packages intact.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 1 ]]; then
  echo 'Usage: bash scripts/halo_live_vision.sh /absolute/path/to/best.pt [extra dashboard arguments]' >&2
  exit 1
fi
weights="$1"
shift
[[ -f "$weights" ]] || { echo "Checkpoint does not exist: $weights" >&2; exit 1; }
cd -- "$project_dir"
export PYTHONPATH="$project_dir/src/rover_perception${PYTHONPATH:+:$PYTHONPATH}"
exec python -m rover_perception.live_dashboard --model "$weights" --device 0 --imgsz 960 \
  --camera-timeout "${ROVER_CAMERA_TIMEOUT:-3.0}" "$@"
