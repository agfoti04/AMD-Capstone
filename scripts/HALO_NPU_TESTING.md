# Halo NPU object detection

Work on `npu-testing`. The current four-class dataset and GPU checkpoint are
the baseline. The NPU backend is explicit and never silently switches to GPU
or accepts a CPU-only model. GPU segmentation and depth are a later milestone.
See [measured status](../docs/NPU_TEST_REPORT.md) before assuming any target is met.

The October 14 comparison selected **retain GPU**. The optimized October 15
YOLOv8s candidate subsequently passed validation/replay; October 16 retesting
uses fresh proof for kernel 7.0.0-38 and fixes its sparse-decoder edge case.
Consult the measured report for the remaining held-out/live/manual gates.
Historical commands below reproduce experiments and include rejected models.

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

## Reproduce scheduling and padding experiments

Best measured candidate from October 15:
`experiments/npu/optimization_20261015/s_rect_a16w8_sparse/manifest.json`.
It is YOLOv8s, A16W8 with per-channel weights, static input `[576,960]`,
one ORT thread, spinning disabled and sparse FP32 DFL decoding. Validation
mAP50-95 is 0.84203; three 60-second replays achieved 38.31, 38.62 and
38.71 FPS including drawing/JPEG. See the report for live/test status.

With the existing simulator/camera bridge running, launch it explicitly:

```bash
cd /home/ian/capstone-training/mars-pilot50-v2
source scripts/halo_npu_env.sh
bash scripts/halo_live_vision.sh --backend npu \
  --model-manifest experiments/npu/optimization_20261015/s_rect_a16w8_sparse/manifest.json \
  --scene scene_050001
```

Use the existing browser/tunnel on port 8771. Stop an earlier dashboard with
Ctrl+C first if it owns that port. The usual GPU checkpoint command is retained.

Create a fresh scheduling/compiler variant without touching the source model:

```bash
python -m rover_perception.npu_tune \
  --manifest experiments/npu/yolov8m_960_xint8/manifest.json \
  --output experiments/npu/my_tuned_candidate \
  --threads 1 --spinning off --opt-level 0
python -m rover_perception.npu_prepare compile \
  --manifest experiments/npu/my_tuned_candidate/manifest.json \
  --hardware-evidence experiments/npu/reference_protocol7.json
```

Thread settings are part of the model manifest, browser/session metadata, and
benchmark identity. They do not change the compiled graph cache key. A new
manifest requires fresh execution/offload evidence before live use. Benchmark
reports record CPU governor and NPU power mode before and after each run.
AMD documents integer compiler levels 0, 1, 2, 3 and experimental 65536. Measure
them individually; a higher level does not guarantee faster inference.

The optional `--external-dfl` switch cuts the verified fixed YOLOv8 decoder from
the ONNX graph. NumPy filters class logits first, then computes FP32 distance
expectations only for surviving anchors. It preserves the learned NPU heads.
The tool checks five training images at confidence 0.001 and 0.25 against the
original CPU graph, with 0.01-pixel / 0.000001-confidence tolerances. It refuses
unknown layouts or changed/missing parity evidence. Full NPU validation remains
necessary: changing graph boundaries can change the AMD compiler's numerics.

To reduce padding work while retaining the camera's 960-pixel width, export a
separate static rectangular candidate in `rover-export`, then quantize it using
the same frozen 300 training images:

```bash
python -m rover_perception.npu_prepare export \
  --weights runs/detect/coverage500_fourclass_yolov8m_npu_960/weights/best.pt \
  --dataset-manifest runs/detect/coverage500_fourclass_yolov8m_npu_960_dataset.json \
  --imgsz 960 --input-height 576 \
  --output experiments/npu/my_rect_fp32
python -m rover_perception.npu_prepare quantize \
  --manifest experiments/npu/my_rect_fp32/manifest.json \
  --precision xint8 --output experiments/npu/my_rect_xint8
```

A 960x576 tensor has 40% fewer pixels than 960x960. A 960x540 camera frame
gets 18 padding rows on either side instead of 210; the difference is divisible
by the largest stride, 32. This preserves image detail and grid alignment, but
boundary context and quantization can still affect predictions. Evaluate all
100 validation images before choosing it. Runtime size comes from the manifest;
the camera itself stays 960x540. The existing fallback training helper also
accepts an official YOLOv8s checkpoint as argument 2 and a fresh run name as
argument 3, recording the actual base architecture.

When XINT8 loses too much accuracy, test AMD's floating-scale integer formats
in a fresh directory. These use the same training-only calibration checks:

```bash
python -m rover_perception.npu_prepare quantize \
  --manifest experiments/npu/my_rect_fp32/manifest.json \
  --precision a8w8 --per-channel --output experiments/npu/my_rect_a8w8
```

`a8w8` uses 8-bit activations/weights and floating scales; `a16w8` uses 16-bit
activations and 8-bit weights. The optional `--per-channel` flag applies weight
scales per channel for these experiments. The manifest records the exact
precision and setting. Both use AMD's X2 integer compiler and require fresh
runtime proof and accuracy evaluation; neither implies CPU fallback is allowed.
See [AMD's integer quantization guidance](https://quark.docs.amd.com/latest/supported_accelerators/ryzenai/int8_int16_quantization.html).
For the observed YOLO26 shared-output assertion in Quark's concat alignment,
`--no-align-concat` exposes the supported setting in a new experiment; the
manifest records it. Recheck NPU execution and accuracy after this override.

Run formal speed comparisons after training/calibration/compiler jobs finish.
Stop Gazebo for isolated replay comparisons; keep it running for the live
workload. Use the same replay and do three
60-second batch-one runs. A short screen or a run competing with calibration
does not establish sustained performance. Changing NPU power mode is optional
and is never done by the launch scripts. On this staged XRT installation, a
privileged invocation must explicitly retain both `XILINX_XRT` and
`LD_LIBRARY_PATH`; the default mode should be restored after an experiment.
See the measured report for actual results and the recommended candidate.

## Launch, drive, and rollback

Only promote a candidate after validation, replay, held-out accuracy, five-minute
live telemetry and the manual drive/stop review all pass.
Also require `compile_report.json` to verify the current model/runtime. The
paths above illustrate the workflow; consult the measured report for accepted
artifacts. The early YOLO26 candidates failed accuracy and should not replace
the working GPU detector. To test a qualifying artifact in the NPU terminal:

```bash
bash scripts/halo_live_vision.sh --backend npu \
  --model-manifest experiments/npu/resume_20261016T221548Z/candidate/manifest.json \
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

For the resumed run, use a fresh NPU terminal and the exact experiment directory:

```bash
cd /home/ian/capstone-training/mars-pilot50-v2
source scripts/halo_npu_env.sh
NPU_RUN=experiments/npu/resume_20261016T221548Z
python -m rover_perception.npu_tools live-check \
  --seconds 300 --interval 0.25 --output "$NPU_RUN/live_acceptance.json"
```

Warm up the dashboard before starting the observer, then drive in
`scene_050001` for the complete five minutes. Avoid raw-frame recording during
the throughput acceptance run. The observer uses `processed_frames` and
`processing_session_id`; it checks the total rate and every complete minute
at >=20 FPS, readiness, errors, stalls, restarts and camera-age growth <=0.25 s.
Four status requests per second can measure a 30-FPS detector accurately.
Run deliberate stop/failure checks after the uninterrupted drive.

The manual checklist is held WASD, simultaneous steering, speed sliders,
release, Space, focus loss, browser disconnection, command expiry <=0.5 s,
stale-camera stopping and backend-error stopping. The controlled backend,
stale-camera and command-expiry regressions must also pass. Note the browser's
display FPS and control acknowledgment latency separately from server FPS.
Only after actually completing/reviewing these checks, record the human review:

```bash
python -m rover_perception.npu_acceptance confirm-controls \
  --live "$NPU_RUN/live_acceptance.json" --reviewer ian --confirmed \
  --notes 'Describe observed display FPS, control latency and stop checks here.' \
  --output "$NPU_RUN/manual_controls.json"
```

Promote only the frozen model that produced all the matching reports:

```bash
python -m rover_perception.npu_acceptance promote \
  --manifest "$NPU_RUN/candidate/manifest.json" \
  --comparison "$NPU_RUN/comparison.json" \
  --gpu-test "$NPU_RUN/gpu_test.json" --npu-test "$NPU_RUN/npu_test.json" \
  --live "$NPU_RUN/live_acceptance.json" --controls "$NPU_RUN/manual_controls.json"
```

This writes the Git-ignored `experiments/npu/active_npu.json` only after all
gates pass. Its model, reports, source files and runtime fingerprint are bound
by hashes. After promotion, the normal NPU terminal command is:

```bash
bash scripts/halo_live_vision.sh --scene scene_050001
```

An explicit checkpoint or `--backend torch --model /path/to/best.pt` still
selects GPU inference. Explicit `--backend npu --model-manifest ...` launches
an experimental candidate without claiming full acceptance. Kernel/runtime,
source or accepted-artifact changes invalidate automatic NPU launch and require
fresh evidence; there is no silent CPU/GPU fallback.

The resumed detached jobs, stage logs and reports are under `$NPU_RUN`.
`formal_job.json` records completed stages or a failure; inspect this before
restarting work after a network interruption. Source archives exclude these
artifacts, datasets, checkpoints and SDKs. Existing experiments are preserved.

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
