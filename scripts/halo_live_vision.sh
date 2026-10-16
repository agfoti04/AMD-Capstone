#!/usr/bin/env bash
# Use rover-yolo for Torch, or source halo_npu_env.sh for the NPU backend.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
model_args=()
if [[ $# -gt 0 && "$1" != --* ]]; then
  weights="$1"
  shift
  [[ -f "$weights" ]] || { echo "Checkpoint does not exist: $weights" >&2; exit 1; }
  model_args=(--model "$weights")
fi
cd -- "$project_dir"
python -c 'import aiohttp' 2>/dev/null || {
  echo 'Install the live dashboard dependency in the selected environment: python -m pip install -r scripts/requirements-live.txt' >&2
  exit 1
}
export PYTHONPATH="$project_dir/src/rover_perception${PYTHONPATH:+:$PYTHONPATH}"
exec python -m rover_perception.npu_acceptance launch "${model_args[@]}" --device 0 --imgsz 960 \
  --camera-timeout "${ROVER_CAMERA_TIMEOUT:-3.0}" "$@"
