# Halo NPU integration: measured status

Branch: `npu-testing`. Measurements on October 14, 2026, on the actual Halo.

| Item | Observed result |
|---|---|
| Host | Ubuntu 24.04.5, kernel 7.0.0-31, Ryzen AI Max+ 395 / Radeon 8060S |
| NPU | RyzenAI-npu5, aie2p 6x8, PCI 1022:17f0 rev 11 |
| Runtime | Separate Ryzen AI 1.8 environment, VitisAI ORT 1.27.0, XRT 2.25.37 |
| Firmware | Old 1.0.0.166 failed execution; user installed upstream 1.1.2.65 and reloaded NPU driver |
| AMD reference CNN | Completed actual inference, 398 operators assigned to NPU, 2 to VITIS_EP_CPU |
| Four-class YOLO26 export | Static batch-one, 960 square, opset 17; FP32 raw-output and decoded parity passed |
| Calibration | Exact reviewed export f008f002e67aff76; 300 training images frozen with hashes |
| GPU validation baseline | Shared evaluator: mAP50 0.95734, mAP50-95 0.79334 |
| GPU fixed-threshold recall | Rock 0.95499, soil 0.94444, crystal 0.90476, slab 0.94737 |
| YOLO26n XINT8 960 on NPU | 27.97 mean replay FPS, but mAP50-95 0.11919 and zero recall at confidence 0.25: rejected |
| YOLO26n BF16 640 on NPU | Actual VAIML execution verified; mAP50-95 0.53225: rejected |
| YOLOv8m XINT8 960 on NPU | mAP50-95 0.79705; 12.54 mean replay FPS; soil recall drop 5.56 points: rejected |
| YOLOv8m BF16 960, FP32 decoding | CPU validation mAP50-95 0.85989; AMD compiler tiling error, all-CPU fallback rejected |
| NPU dashboard stream | Actual NPU frames, four-class metadata and zero-control WebSocket acknowledgment verified; clean shutdown stopped commands |
| Adoption | Retain GPU; no NPU candidate meets all gates. Five-minute NPU drive and held-out test were not run |

The GPU baseline uses the shared evaluator at confidence 0.001 for AP and
confidence 0.25 / IoU 0.5 for recall. This is a fresh baseline, not a replacement
for the original Ultralytics validation/test summary. Median GPU model execution
was 5.64 ms across the 100 validation images; this excludes the full browser path.

The successful FP32 candidate is `experiments/npu/yolo26n_960_fp32_v3` on the Halo.
The first attempts exposed an Ultralytics 8.4.175 export-head change: `nms=False`
forces one-to-one inference even when the checkpoint uses one-to-many. The tool
now preserves the source head and checks outputs before quantization.

The unquantized ONNX CPU evaluation reached 0.79334 mAP50-95, matching the
GPU baseline. The XINT8 ONNX model reached only 0.53971 on CPU and 0.11919 on
the NPU. The additional NPU confidence suppression is repeatable on the same
training image. Its compiler report assigns 1,537 operators to NPU and 13
residual operations to CPU, with no learned Conv/MatMul on CPU; runtime profiling
confirms execution. Offload and speed pass, but accuracy fails. The X1 legacy
compiler path aborted during execution and is not an accepted alternative.

Three 60-second batch-one replays used the same 1,436 lossless 960 x 540 camera
frames. Source delivery during recording was 23.93 FPS. Replay includes drawing
and JPEG encoding, excludes disk/network, and can exceed the source rate:

| Stage | GPU YOLO26n median / p95 ms | NPU YOLO26n XINT8 median / p95 ms |
|---|---:|---:|
| Preprocessing | 0.57 / 1.15 | 1.09 / 1.73 |
| Model execution | 6.47 / 10.14 | 32.34 / 38.67 |
| Postprocessing | 1.59 / 2.23 | 0.53 / 0.70 |
| Drawing | 0.42 / 0.53 | 0.09 / 0.12 |
| JPEG | 0.88 / 1.00 | 0.71 / 0.87 |
| Total | 10.53 / 15.31 | 34.85 / 41.82 |
| Processed FPS by run | 92.43, 89.68, 90.07 | 28.18, 28.10, 27.62 |

The rejected model produces fewer detections, so its drawing cost is not an
equal-work speed benefit. GPU was faster in this comparison. These experiments
overlapped CPU compilation/calibration work and are not isolated power tests.

Reducing input size alone also loses accuracy: GPU YOLO26n at 640 reached
0.65978; GPU YOLOv8m at 640 reached 0.76937, below the 0.77334 adoption floor.
BF16 YOLO26n at 640 reached 0.53423 on CPU, close to its 0.53225 NPU result;
this loss is mainly quantization rather than an NPU-only execution mismatch.
Preparation now keeps coordinate/probability decoding in FP32 for both BF16
and XINT8. Earlier whole-graph BF16 artifacts remain as experiment evidence.
The corrected BF16 YOLOv8m candidate reaches 0.85989 mAP50-95 on CPU, preserving
the 0.85882 FP32 GPU result. Its 960-pixel NPU compilation failed after roughly
18 minutes with `transpose4d failed to find tiling solution for perm13`.
AMD then assigned all 867 operators to CPU. The adapter rejected this fallback
before declaring the dashboard ready. This is a measured failure of this
candidate/compiler configuration, not proof that every BF16 configuration is
unsupported.

The original 960 BF16 compile was stopped after its initial 20-minute budget.
The 640 BF16 compile completed after roughly 18 minutes; its assignment report
uses the device name VAIML, which verification now recognizes alongside NPU.
No successful compile is treated as an accuracy result.

YOLOv8m fallback training completed 30 epochs at 960/batch 4/seed 0 in a new
run, preserving all original weights and labels. Its FP32 GPU validation
mAP50-95 was 0.85882. Its 640 XINT8 NPU candidate reached 0.70897 and was
rejected. The 960 XINT8 candidate passes the overall mAP gate against the
original YOLO26n baseline but fails both speed and soil recall:

| Fixed-confidence recall | GPU baseline | NPU YOLOv8m XINT8 960 |
|---|---:|---:|
| Rock | 95.50% | 94.32% |
| Pale soil | 94.44% | 88.89% |
| Yellow crystal | 90.48% | 95.24% |
| Mudstone slab | 94.74% | 100.00% |

NPU replay runs reached 12.76, 12.51 and 12.35 processed FPS; median/p95 total
latency was 79.80/90.26 ms. Median model execution was 72.99 ms, preprocessing
1.69 ms, postprocessing 1.05 ms, drawing 0.35 ms and JPEG 0.85 ms. Learned
convolutions are on NPU (1,234 assigned operators); 25 CPU operations include
fixed DFL decoding. Its CPU convolution is verified against the original
decoding graph outputs, including ORT's optimized NCHWc name, and is not a
learned backbone/head convolution.

The architecture-matched YOLOv8m GPU comparison used the same 960-pixel input,
checkpoint, recorded frames and three 60-second runs: 35.15 mean processed FPS,
with 28.44/29.29 ms median/p95 total latency. The NPU was slower in this
comparison. ORT execution timing includes its residual CPU graph decoding;
Python NMS/box scaling is measured separately as postprocessing.

Artifacts and full reports are in `experiments/npu/` on the Halo. The original
checkpoint SHA256 is `bc10fb9bbde39d1ef6b8896dd156e15c2a3bb882304125139af6942a604c6b70`;
the fallback checkpoint is `c6fa6fdd48b14d70af19ecae5962ee041e30d29d2609271a996fca7b65179065`.

The AMD reference initially crashed with a device-command error. XRT's own
latency diagnostic also failed, establishing a device/runtime issue independent
of YOLO. After the firmware update the reference actually ran on the NPU.
The verifier also caught a successful CPU fallback caused by an incorrect
library search order; successful process exit alone is not accepted as proof.

Local tests cover the adapters, preprocessing/decoding, manifests, calibration,
cache invalidation, evidence validation, metrics, candidate selection, and the
existing browser/drive behavior. The deployment archive is built from source
and its extracted package is checked against stale installed modules.
Windows and the actual Linux NPU environment both passed 132 Python tests;
the browser control JavaScript checks and extracted-archive wheel build passed.

The actual NPU dashboard smoke test received five distinct JPEG frames through
the video WebSocket, verified NPU model/proof metadata, acknowledged a zero
command through the independent control WebSocket, recorded session provenance,
and exited cleanly with bridge command `[0, 0]`. It used temporary localhost port
8772 and left no dashboard listener running. It does not establish a five-minute
drive, client network latency or held-control acceptance.

`experiments/npu/comparison_all.json` records `status: retain_gpu` and no selected
detector. Other reports include `gpu_val_960.json`, `gpu_replay.json`,
`npu_yolov8m_960_xint8_val.json`, `npu_yolov8m_960_xint8_replay.json`,
`gpu_yolov8m_960_replay.json`, `dashboard_npu_smoke.json` and
`linux_regressions.log`. Original labels/weights and the reviewed test split
remain unchanged. No held-out test was used to choose a candidate.

The next experiment should target a smaller documented YOLOv8 architecture and
accuracy-preserving quantization, focusing on soil recall. That is follow-up
work; the current evidence does not support promoting the tested NPU models or
starting GPU depth/segmentation alongside an accepted NPU detector yet.

Adoption requires >=20 processed FPS in each of three 60-second replays,
<=0.02 overall validation mAP50-95 loss, <=0.05 recall loss for every class,
verified learned computation on the NPU, and a separate five-minute live test.
No NPU performance or accuracy target is claimed until its saved report passes.
