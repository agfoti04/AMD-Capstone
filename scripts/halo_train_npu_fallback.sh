#!/usr/bin/env bash
# Run in rover-yolo only after the YOLO26 NPU candidates fail the documented gates.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
dataset_manifest="${1:-$project_dir/runs/detect/coverage500_fourclass_960_dataset.json}"
base_model="${2:-$project_dir/yolov8m.pt}"
run_name="${3:-coverage500_fourclass_yolov8m_npu_960}"
[[ "$run_name" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Use letters, digits, underscores or hyphens for the run name.' >&2; exit 1; }
[[ ! -e "$project_dir/runs/detect/$run_name" && ! -e "$project_dir/runs/detect/${run_name}_dataset.json" ]] || { echo 'Run already exists; choose a fresh name.' >&2; exit 1; }
[[ -f "$base_model" ]] || { echo "Download the pretrained YOLOv8m checkpoint to $base_model (or supply it as argument 2)." >&2; exit 1; }
export PYTHONPATH="$project_dir/src/rover_perception${PYTHONPATH:+:$PYTHONPATH}"
data_yaml="$(python -c 'import sys; from rover_perception.npu_prepare import resolve_dataset_manifest,create_calibration_manifest; _,p=resolve_dataset_manifest(sys.argv[1]); create_calibration_manifest(p); print(p)' "$dataset_manifest")"
# Keep the exact reviewed dataset revision used by the current four-class model.
python -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True); p.open("x",encoding="utf-8").write(json.dumps({"data_yaml":sys.argv[2],"source_training_manifest":str(Path(sys.argv[3]).resolve()),"base_architecture":"yolov8m"},indent=2)+"\n")' \
  "$project_dir/runs/detect/${run_name}_dataset.json" "$data_yaml" "$dataset_manifest"
exec yolo train cfg="$project_dir/src/mars_training_environment/config/train960.yaml" \
  model="$base_model" data="$data_yaml" device=0 epochs=30 imgsz=960 batch=4 seed=0 \
  project="$project_dir/runs/detect" name="$run_name"
