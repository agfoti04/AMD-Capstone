# Halo NPU object detection

Work on `npu-testing`. The current four-class dataset and GPU checkpoint are
the baseline. The NPU backend is explicit and never silently switches to GPU
or accepts a CPU-only model. GPU segmentation and depth are a later milestone.
See [measured status](../docs/NPU_TEST_REPORT.md) before assuming any target is met.

The October 14 Halo comparison selected **retain GPU**. Real NPU inference and
the browser stream work, but tested candidates miss accuracy/throughput gates.
The commands below reproduce experiments; they are not a recommendation to
replace the working GPU detector with a rejected candidate.

## Environments and deployment

The Halo project is `/home/ian/capstone-training/mars-pilot50-v2`.
Use separate terminals for these environments:

| Environment | Purpose |
|---|---|
| `rover-yolo` Conda | Existing GPU training, baseline evaluation, GPU browser |
| `rover-export` Conda | CPU ONNX export, parity and AMD Quark quantization |
| `~/capstone-training/venvs/rover-npu` | Ryzen AI 1.8, compilation and NPU inference |

Current Quark requires NumPy 2; Ryzen AI 1.8's flexml pins NumPy <=1.26.4.
Do not install Quark or stock onnxruntime into the NPU environment. The SDK
supplies its own `onnxruntime-vitisai` provider. The working GPU environment
does not require dependency changes.

Deploy the source archive from Windows PowerShell:

```powershell
scp -i "$env:USERPROFILE\.ssh\halo_ed25519" C:\Projects\Capstone\halo-npu-testing-20261014.tar.gz ian@halo-strix:~/
```

On the Halo, stop a running dashboard before replacing its source:

```bash
cd /home/ian/capstone-training/mars-pilot50-v2
tar --touch -xzf ~/halo-npu-testing-20261014.tar.gz -C "$PWD"
```

The host dashboard runs directly from source. NPU integration does not require
a ROS rebuild when the version-2 camera bridge is already installed. For an
older simulator installation, follow the build block in `HALO_LIVE_VISION.md`.

## Initial runtime setup

This session already created the export and NPU environments on the Halo.
For a fresh machine, follow AMD's official [Linux installation guide](https://ryzenai.docs.amd.com/en/latest/linux.html)
for Ubuntu 24.04, Python 3.12, XRT and the SDK installer. Install the SDK into
`/home/ian/capstone-training/venvs/rover-npu` rather than the GPU environment.

The current Halo uses its existing kernel-7 `amdxdna` module. XRT 2.25.37 and
its Boost dependencies were extracted without a system install into
`~/capstone-training/npu-install/xrt-root`. `halo_npu_env.sh` uses this directory
when system XRT is absent; `ROVER_XRT_ROOT` and `ROVER_NPU_ENV` override locations.
It preserves the SDK Python library search paths and puts the matched XRT ahead
of the older coreutil library bundled by VOE.

The old firmware 1.0.0.166 enumerated but aborted device execution. The user ran
the checksum-checked `halo_update_npu_firmware.sh` helper; firmware 1.1.2.65 now
loads through `npu_7.sbin`. The original fallback firmware remains installed.
Do not rerun firmware updates on another device without checking its identity.

In an NPU terminal:

```bash
cd /home/ian/capstone-training/mars-pilot50-v2
source scripts/halo_npu_env.sh
python -m rover_perception.npu_tools preflight --output experiments/npu/preflight.json
python -m rover_perception.npu_tools verify-reference \
  --quicktest /home/ian/capstone-training/venvs/rover-npu/quicktest/quicktest.py \
  --output experiments/npu/reference_protocol7.json
```

Require `verified: true`. A provider listing or `Test Finished` alone does not
prove NPU use: the SDK can return a successful CPU fallback. This verification
also requires an enumerated device and a positive NPU operator assignment.

## Export and prepare the current detector

If recreating the export environment, clone the existing GPU environment:

```bash
source /home/ian/miniforge3/etc/profile.d/conda.sh
conda create --name rover-export --clone rover-yolo -y
conda activate rover-export
conda install --channel conda-forge gxx_linux-64 -y
python -m pip install -r scripts/requirements-npu-export.txt
```

In the export terminal:

```bash
source /home/ian/miniforge3/etc/profile.d/conda.sh
conda activate rover-export
cd /home/ian/capstone-training/mars-pilot50-v2
export PYTHONPATH="$PWD/src/rover_perception"
python -m rover_perception.npu_prepare export \
  --weights runs/detect/coverage500_fourclass_960/weights/best.pt \
  --dataset-manifest runs/detect/coverage500_fourclass_960_dataset.json \
  --output experiments/npu/yolo26n_960_fp32 --imgsz 960
```

Use a fresh output directory. The successful export from this session is
`experiments/npu/yolo26n_960_fp32_v3/manifest.json`; earlier attempts remain as
failure evidence. Export preserves the checkpoint's trained head: forcing
`nms=False` on this YOLO26 checkpoint selects its other head. Opset 17, static
batch one and both raw-output and decoded-box parity are checked. Parity images
are spread across the training split rather than only the first background scene.

For BF16, using the successful export from this session:

```bash
HIP_VISIBLE_DEVICES="" CUDA_VISIBLE_DEVICES="" MAX_JOBS=4 \
python -m rover_perception.npu_prepare quantize \
  --manifest experiments/npu/yolo26n_960_fp32_v3/manifest.json \
  --precision bf16 --output experiments/npu/yolo26n_960_bf16
```

Quark's custom ops are built on CPU. Empty visibility values hide the GPU for
this process; this ROCm version rejects `-1`. Compilation uses the Conda compiler.
The manifest freezes all 300 reviewed training images with hashes; validation
and test images cannot be used for calibration. XINT8 uses the same command
with `--precision xint8` and a new output directory. Detection decoding nodes
are excluded from both precisions using graph traversal, not hard-coded layer
names. This preserves FP32 coordinates and probabilities for small targets.

Back in the NPU terminal, require the reference hardware gate before compilation:

```bash
python -m rover_perception.npu_prepare compile \
  --manifest experiments/npu/yolo26n_960_bf16/manifest.json \
  --hardware-evidence experiments/npu/reference_protocol7.json
```

Inspect `compile_report.json`. It must prove actual VitisAI execution and place
learned convolution/matrix computation on the NPU, allowing explicitly identified
CPU decoding. Model hashes, driver/runtime fingerprints, assignment reports and
runtime profiles identify the cache. Driver/runtime changes require fresh proof.

## Accuracy and paired replay

In a GPU terminal:

```bash
source /home/ian/miniforge3/etc/profile.d/conda.sh
conda activate rover-yolo
cd /home/ian/capstone-training/mars-pilot50-v2
export PYTHONPATH="$PWD/src/rover_perception"
python -m rover_perception.npu_tools evaluate --backend torch \
  --model runs/detect/coverage500_fourclass_960/weights/best.pt \
  --dataset-manifest runs/detect/coverage500_fourclass_960_dataset.json \
  --split val --output experiments/npu/gpu_val_960.json
```

The shared evaluator uses square padding for both backends, low-confidence
predictions for mAP, and recall at confidence 0.25 / IoU 0.5. These settings and
the shared matching implementation can differ from the original Ultralytics
training summary; compare both backends using this same evaluator.

Start the existing GPU-rendered browser simulator in another Halo terminal:

```bash
cd /home/ian/capstone-training/mars-pilot50-v2
bash scripts/halo_live_sim.sh dataset/live_eval_scenes/scenes/scene_050001/scene.json browser
```

Record its lossless camera frames once, then replay that same set for every model:

```bash
python -m rover_perception.npu_tools record \
  --output experiments/npu/camera_replay --seconds 60
python -m rover_perception.npu_tools benchmark --backend torch \
  --model runs/detect/coverage500_fourclass_960/weights/best.pt \
  --frames experiments/npu/camera_replay --seconds 60 --repeats 3 \
  --output experiments/npu/gpu_replay.json
```

In the NPU terminal:

```bash
python -m rover_perception.npu_tools evaluate --backend npu \
  --manifest experiments/npu/yolo26n_960_bf16/manifest.json \
  --dataset-manifest runs/detect/coverage500_fourclass_960_dataset.json \
  --split val --output experiments/npu/npu_bf16_val.json
python -m rover_perception.npu_tools benchmark --backend npu \
  --manifest experiments/npu/yolo26n_960_bf16/manifest.json \
  --frames experiments/npu/camera_replay --seconds 60 --repeats 3 \
  --output experiments/npu/npu_bf16_replay.json
python -m rover_perception.npu_tools compare \
  --baseline-evaluation experiments/npu/gpu_val_960.json \
  --candidate-evaluations experiments/npu/npu_bf16_val.json \
  --benchmarks experiments/npu/gpu_replay.json experiments/npu/npu_bf16_replay.json \
  --output experiments/npu/comparison.json
```

The replay report separates preprocessing, execution, postprocessing, annotation
and JPEG costs with median/p95 measurements. Disk I/O and network are excluded;
live tests measure those limits. For isolating the backend, also run Torch at
each candidate's input size. The adoption comparison keeps the original
960-pixel baseline, so a smaller candidate must account for its accuracy tradeoff.

Qualifying candidates need three complete runs at >=20 FPS, mAP50-95 drop <=0.02,
each class's recall drop <=0.05, and verified NPU computation. Among qualifiers,
highest validation mAP wins, then lower median latency. Reports must match their
model/cache, host, class IDs, reviewed labels and recorded input hashes. If none
qualifies, keep the GPU detector. Reserve `--split test` for the selected model.

Repeat preparation at `--imgsz 640` in fresh directories if 960 is too slow.
If all YOLO26 configurations fail, use the reviewed dataset to train YOLOv8m:

```bash
# In rover-yolo; supply a downloaded official pretrained YOLOv8m checkpoint.
bash scripts/halo_train_npu_fallback.sh \
  runs/detect/coverage500_fourclass_960_dataset.json /path/to/yolov8m.pt
```

The fallback uses 30 epochs, 960 pixels, batch 4, seed 0, and a new run name.
YOLO26 weights cannot be loaded as YOLOv8 weights. Prepare and evaluate that
checkpoint through the same gates; do not regenerate or relabel the dataset.
This session's completed fallback is
`runs/detect/coverage500_fourclass_yolov8m_npu_960/weights/best.pt`; its dataset
manifest has the same run name plus `_dataset.json`. The measured INT8 artifacts
are `experiments/npu/yolov8m_960_xint8` and `yolov8m_640_xint8`. The BF16 candidate
with FP32 decoding is `yolov8m_960_bf16_fp32decode`. Preserve these directories;
use new names when changing quantization/compiler options.

## Launch, drive, and rollback

Only promote a candidate after its validation/replay comparison qualifies it.
Also require `compile_report.json` to verify the current model/runtime. The
paths above illustrate the workflow; consult the measured report for accepted
artifacts. The early YOLO26 candidates failed accuracy and should not replace
the working GPU detector. To test a qualifying artifact in the NPU terminal:

```bash
bash scripts/halo_live_vision.sh --backend npu \
  --model-manifest experiments/npu/yolo26n_960_bf16/manifest.json \
  --scene scene_050001
```

Use the same browser at http://127.0.0.1:8771, the same SSH tunnel, held WASD,
speed sliders and connection-stop behavior. The dashboard compiles/warms up
before it is ready; stale camera data or inference failure stops movement.
It refuses unverified deployment evidence rather than switching providers.

From another NPU terminal, observe a five-minute drive:

```bash
python -m rover_perception.npu_tools live-check \
  --seconds 300 --output experiments/npu/live_acceptance.json
```

The observer does not drive. Manually test hold, release, Space, loss of focus
and disconnection. Inspect displayed FPS, camera age and control acknowledgment
latency. A replay pass does not imply a live pass when Gazebo or the network is slow.

To return to the GPU, release controls, stop the NPU dashboard with Ctrl+C,
and in a fresh `rover-yolo` terminal run:

```bash
cd /home/ian/capstone-training/mars-pilot50-v2
bash scripts/halo_live_vision.sh \
  /home/ian/capstone-training/mars-pilot50-v2/runs/detect/coverage500_fourclass_960/weights/best.pt \
  --scene scene_050001
```

Experiment artifacts stay under `experiments/npu/`, excluded from Git. The source
archive contains tooling, tests and documentation, never datasets, weights or SDKs.

References: [AMD Linux setup](https://ryzenai.docs.amd.com/en/latest/linux.html),
[AMD detector example](https://github.com/amd/RyzenAI-SW/tree/main/CNN-examples/object_detection/yolov8m),
[AMD assignment and deployment](https://ryzenai.docs.amd.com/en/latest/modelrun.html),
[upstream firmware](https://gitlab.com/kernel-firmware/linux-firmware/-/tree/main/amdnpu/17f0_11).
