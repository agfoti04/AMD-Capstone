# Four-class mudstone training on the Halo

This workflow copies the reviewed 500-image export into a separate dataset.
It reuses exact images, edited boxes, frame exclusions and train/val/test
assignments. It identifies slabs from capture metadata, never model predictions.

| ID | Class | Appearance |
| --- | --- | --- |
| 0 | rock | Ordinary rocks and intact pale stones |
| 1 | pale_soil_patch | Pale soil and ground lookalikes |
| 2 | yellow_crystal_rock | Yellow crystal/fracture appearance |
| 3 | mudstone_slab | Both veined_lookalike and cheyava_rock slabs |

Capture keeps its established three-class schema; reclassification creates an
explicit new version. Capture checks refer to the reused pixels/geometry, while
class provenance and human review are tracked separately.

## Transfer and prepare

In Windows PowerShell:

```powershell
scp C:\Projects\Capstone\halo-fourclass-20261014.tar.gz ian@halo-strix:~/
```

In a Halo terminal:

```bash
cd /home/ian/capstone-training/mars-pilot50-v2
tar --touch -xzf /home/ian/halo-fourclass-20261014.tar.gz -C /home/ian/capstone-training/mars-pilot50-v2
source /home/ian/miniforge3/etc/profile.d/conda.sh
conda activate rover-yolo
export PYTHONPATH="$PWD/src/mars_training_environment:$PWD/src/rover_training${PYTHONPATH:+:$PYTHONPATH}"
```

These tools and live inference run from host source files. The working
pipeline-2 simulator can keep running; no Docker/ROS rebuild is required.

Use the approved export reported by the successful 500-image training run:

```bash
python -m mars_training_environment.reclassify prepare \
  --source dataset/coverage500_balanced \
  --review-export dataset/coverage500_balanced/review/exports/70a3ffb2825a2df0 \
  --output dataset/coverage500_fourclass
```

If you subsequently approved another export, substitute its exact directory.
Preparation verifies image/label hashes and capture revisions. It rejects
changed exports, an existing output, destinations inside the source dataset,
and scene groups spanning different splits.

The new root is `/home/ian/capstone-training/mars-pilot50-v2/dataset/coverage500_fourclass`.
Its `relabel_report.json` records source evidence, changes and uncertain frames.
Unmatched manual rock boxes and conflicting reviews are preserved and flagged
for a human decision. No deleted box is recreated.

Run preparation once. Reuse this root for subsequent reviews. To start another
version, choose a fresh output directory.

## Review the new classes

Stop an existing port-8765 review server with Ctrl+C in its terminal. In the
activated Halo terminal above:

```bash
python -m mars_training_environment.review --dataset dataset/coverage500_fourclass --port 8765
```

In a separate Windows PowerShell terminal:

```powershell
ssh -N -L 8765:127.0.0.1:8765 ian@halo-strix
```

Open **http://127.0.0.1:8765** and refresh with Ctrl+F5. The class menu contains
`mudstone_slab` and its boxes are purple. Choose **Status: Unreviewed**. Open
each changed/uncertain frame, check object class and box extent, correct manual
boxes with the class dropdown, and click **Save corrections / mark reviewed**.
Exclude a frame if it cannot be labeled reliably. Ordinary rock, soil and
crystal boxes should retain their meanings.

Click **Export reviewed training data** after completing those checks. Export
is blocked while included changed/uncertain frames remain unreviewed. Curated
exports are versioned under `coverage500_fourclass/review/exports`.

## Audit coverage before training

In another activated Halo terminal, cd to the project and set the same
PYTHONPATH as above:

```bash
python -m mars_training_environment.reclassify audit --dataset dataset/coverage500_fourclass
```

Require `pending_class_reviews: []`. `class_coverage` counts instances, positive
images and independent scene groups for each class/split. `coverage_warnings: []`
means this pilot guidance is met:

| Split | Instances per class | Positive images | Independent scene groups |
| --- | ---: | ---: | ---: |
| Train | 30 | 10 | 3 |
| Val | 10 | 5 | 2 |
| Test | 10 | 5 | 2 |

These are practical pilot thresholds, not a statistical accuracy guarantee.
If warnings remain, send the audit output before training so additional
captures can target the underrepresented class/split. More captures are needed
only if measured coverage is insufficient. Do not duplicate frames to reach
these counts or move views of a scene between splits.

## Train a separate model

Stop live inference/simulation to free the GPU. In the activated Halo terminal:

```bash
bash scripts/halo_train_fourclass.sh
```

The helper rechecks coverage/reviews, exports the latest corrected labels,
and starts a separate 30-epoch YOLO26n run at 960 pixels/batch 4 on device 0.
It uses `/home/ian/capstone-training/yolo26n.pt` as its base and an absolute
project directory, adapting the model to four classes. It does not resume the
old three-class run. Keep the current ROCm/Torch/Ultralytics environment.

The new weights are:

```text
/home/ian/capstone-training/mars-pilot50-v2/runs/detect/coverage500_fourclass_960/weights/best.pt
```

The helper refuses an existing run name. For a second run, supply a fresh
third argument and use that name's weights in subsequent commands:

```bash
bash scripts/halo_train_fourclass.sh \
  /home/ian/capstone-training/mars-pilot50-v2/dataset/coverage500_fourclass \
  /home/ian/capstone-training/yolo26n.pt coverage500_fourclass_960_v2
```

## Evaluate on the unchanged test split

Use the exact curated export recorded by the training helper:

```bash
FOUR_DATA=$(python -c "import json; print(json.load(open('runs/detect/coverage500_fourclass_960_dataset.json'))['data_yaml'])")
```

```bash
yolo val \
  model=/home/ian/capstone-training/mars-pilot50-v2/runs/detect/coverage500_fourclass_960/weights/best.pt \
  data="$FOUR_DATA" split=test imgsz=960 batch=4 rect=False device=0 plots=True \
  project=/home/ian/capstone-training/mars-pilot50-v2/runs/detect \
  name=coverage500_fourclass_test
```

Inspect per-class precision/recall/mAP and the confusion matrix, especially
rock versus mudstone_slab. Three- and four-class overall mAP values are not
directly equivalent because class definitions changed. Keep the test split
for final evaluation; tune with training/validation examples.

## Browser test with the new weights

In the first Halo terminal, from the project root:

```bash
bash scripts/halo_live_sim.sh dataset/live_eval_scenes/scenes/scene_050001/scene.json browser
```

In a second activated rover-yolo terminal:

```bash
bash /home/ian/capstone-training/mars-pilot50-v2/scripts/halo_live_vision.sh \
  /home/ian/capstone-training/mars-pilot50-v2/runs/detect/coverage500_fourclass_960/weights/best.pt \
  --scene scene_050001
```

Reuse the port-8771 tunnel, open **http://127.0.0.1:8771**, and refresh with
Ctrl+F5. The dashboard accepts both old three-class weights and the new
four-class model. The checkpoint you start determines the labels displayed.
The slabs remain scene objects and should be predicted as `mudstone_slab`
after training. Driving and GPU rendering remain compatible with pipeline 2.

## Validation and remaining Halo work

Local checks cover source/pixel/box/split preservation, exclusions, both slab
appearances, uncertain/conflicting edits, stale source detection, four-class
review/export, coverage guidance and live predictions. The real Halo dataset
is not in this workspace. Its counts, human review, GPU training and held-out
metrics must be obtained by running the commands above.

References: [Ultralytics detection datasets](https://docs.ultralytics.com/datasets/detect/),
[training](https://docs.ultralytics.com/modes/train/),
[validation](https://docs.ultralytics.com/modes/val/).
