#!/usr/bin/env bash
# Run in rover-yolo. Uses a separate, reviewed four-class dataset and run.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
dataset="${1:-$project_dir/dataset/coverage500_fourclass}"
base_model="${2:-/home/ian/capstone-training/yolo26n.pt}"
run_name="${3:-coverage500_fourclass_960}"
[[ "$run_name" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Use letters, digits, underscores or hyphens for the run name.' >&2; exit 1; }
[[ ! -e "$project_dir/runs/detect/$run_name" && ! -e "$project_dir/runs/detect/${run_name}_dataset.json" ]] || { echo 'Training run or its dataset manifest already exists; provide a fresh name as the third argument.' >&2; exit 1; }
[[ -f "$base_model" ]] || { echo "Base checkpoint does not exist: $base_model" >&2; exit 1; }
export PYTHONPATH="$project_dir/src/mars_training_environment:$project_dir/src/rover_training${PYTHONPATH:+:$PYTHONPATH}"
data_yaml="$(python -m mars_training_environment.reclassify export --dataset "$dataset" --require-coverage --print-data-path \
  --run-manifest "$project_dir/runs/detect/${run_name}_dataset.json")"
exec yolo train cfg="$project_dir/src/mars_training_environment/config/train960.yaml" \
  model="$base_model" data="$data_yaml" device=0 \
  project="$project_dir/runs/detect" name="$run_name"
