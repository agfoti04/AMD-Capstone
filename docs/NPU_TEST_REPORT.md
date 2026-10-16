# Halo NPU integration: measured status

Branch: `npu-testing`. Initial measurements on October 14, 2026, on the actual Halo.
The October 15 optimization results appear below; the initial failures are
retained as experiment history.

**Current October 16 status:** the optimized NPU detector passes fresh
validation, three-run replay and paired held-out accuracy gates on kernel
7.0.0-38. The five-minute manual browser drive/stop review is still pending,
so no automatic NPU-default record is installed. See
[resumed acceptance](#october-16-resumed-acceptance) for the latest results;
the first table below describes the initial October 14 experiments.

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

The October 15 experiments below follow up with a smaller YOLOv8 architecture,
different quantization formats, scheduling changes and reduced padding.

Adoption requires >=20 processed FPS in each of three 60-second replays,
<=0.02 overall validation mAP50-95 loss, <=0.05 recall loss for every class,
verified learned computation on the NPU, and a separate five-minute live test.
No NPU performance or accuracy target is claimed until its saved report passes.

## October 15 optimization experiments

All new artifacts live under `experiments/npu/optimization_20261015/` on the Halo,
outside Git. Original checkpoints, reviewed labels, runtime environments and
the simulator were preserved. YOLOv8s was trained in a fresh run for 30 epochs
at 960 pixels, batch 4, seed 0, using the exact same reviewed dataset export.
No held-out test images were used for calibration or candidate selection.

| Candidate | Input H x W | Validation mAP50-95 | Soil recall at 0.25 | Finding |
|---|---|---:|---:|---|
| Original GPU YOLO26n | 960 x 960 | 0.79334 | 94.44% | Fixed comparison baseline |
| GPU YOLOv8s FP32 | 960 x 960 | 0.84409 | 92.59% | New architecture's unquantized baseline |
| NPU YOLOv8s XINT8 | 960 x 960 | 0.76133 | 87.04% | Fails accuracy; CPU quantized result is identical |
| NPU YOLOv8s A8W8, per-channel | 960 x 960 | 0.67634 | 85.19% | Fails accuracy; CPU reaches only 0.66827 |
| NPU YOLOv8m XINT8 | 576 x 960 | 0.79559 | 88.89% | Soil recall remains outside original gate |
| NPU YOLOv8m A8W8, per-channel | 576 x 960 | 0.80662 | 79.63% | Overall AP hides unacceptable soil-recall loss |
| NPU YOLO26n A16W8, per-channel | 960 x 960 | 0.79349 | 94.44% | Preserves original accuracy |
| NPU YOLO26n A16W8, per-channel | 576 x 960 | 0.79027 | 94.44% | Passes validation accuracy gates |
| NPU YOLOv8s A16W8, per-channel | 960 x 960 | 0.84157 | 92.59% | Passes validation accuracy gates |
| NPU YOLOv8s A16W8, per-channel | 576 x 960 | 0.84203 | 92.59% | Passes validation accuracy gates; dense and sparse decoding agree on metrics |

A16W8 keeps activations at 16 bits and weights at 8 bits with floating scales.
This recovered the accuracy lost with XINT8 and A8W8 on the smaller model.
It is actual NPU execution with learned-compute assignment audited, rather than
CPU fallback. Every quantized candidate uses the frozen 300 training images.
AMD documents these formats in its
[integer quantization guide](https://quark.docs.amd.com/latest/supported_accelerators/ryzenai/int8_int16_quantization.html).

Threading was a repeatable initial bottleneck. Alternating one-thread/default
trials on the original YOLOv8m XINT8 detector gave approximately 17.3 FPS versus
13.2-14.3 FPS, before drawing/encoding. Steady profiling attributed roughly
4-7 ms to CPU graph decoding and 51-61 ms to the Vitis provider. Provider timing
includes dispatch, copies and synchronization; it is not pure device arithmetic.
One thread improved both CPU decoding and the provider duration, so CPU decoding
alone did not explain the original GPU/NPU gap. Scheduling/resource contention
is implicated, but these timings do not identify its exact mechanism.

Manifest-controlled thread counts and spinning now make these settings
reproducible. Integer compiler levels 0, 1, 2, 3 and experimental 65536 screened
at approximately 17.1-17.5 FPS on the original model with one thread. The small
spread does not justify claiming that a higher level is faster. Standard level
0 is retained for the final candidates. NPU performance power mode gave about
17.2 FPS versus about 17.1 FPS in default mode, with no convincing gain. The
user restored default mode, which was verified with XRT. XRT reports power and
temperature as unavailable; no energy-efficiency improvement is claimed.

The 576-row tensor retains the complete 960x540 camera image with 18 rows of
padding above and below, versus 210 at 960 square. This reduces tensor pixels
by 40% while preserving camera detail and stride alignment. It was exported,
checked against PyTorch and recalibrated separately. It is not a downsampled
640-pixel model. Predictions can still change at boundaries, hence the full
validation comparison.

Optional sparse DFL decoding exports raw distance/class logits, filters classes
first and computes FP32 distances only for surviving anchors. Learned feature
extraction and prediction stay on NPU. Five training images at both AP and live
confidence thresholds must match CPU reference decoding before compilation;
the report is hash-bound to the transformed model. The new model/decoder also
undergoes full NPU validation. On the original larger detector, a clean short
screen improved from 17.2 FPS to 18.3 FPS with sparse decoding.

YOLO26 floating-scale quantization initially hit Quark's shared-output concat
alignment assertion. A fresh A16W8 experiment disabled the exposed `AlignConcat`
setting and passed subsequent NPU execution/accuracy checks. No SDK files were
patched. Other failed experiments remain in their original directories.

The best validation/replay candidate is
`optimization_20261015/s_rect_a16w8_sparse/manifest.json`, using A16W8,
per-channel weights, static `[576,960]`, one ORT thread, spinning disabled,
standard X2 optimization level 0 and sparse FP32 DFL decoding.

| Clean replay stage | Median ms |
|---|---:|
| Preprocessing | 0.589 |
| ORT execution, including residual graph operations | 23.324 |
| Sparse decoding/NMS/scaling | 0.929 |
| Drawing | 0.306 |
| JPEG | 0.671 |
| Total | **25.899** |

Three complete 60-second runs achieved **38.31, 38.62 and 38.71 processed FPS**
(6,940 frames total). Validation mAP50-95 is **0.84203**, 4.87 percentage points
above the original GPU YOLO26n baseline. Fixed-confidence recalls are rock
97.46%, soil 92.59%, crystal 95.24%, slab 100%; the largest recall decrease
versus that baseline is soil's 1.85 points. All original validation/replay gates
pass, with actual learned computation on the NPU.
Total p95 latency was **26.686 ms**; individual stage distributions are saved
in the replay JSON.

This is approximately three times the historical larger XINT8 candidate's
12.54 FPS, with a different architecture, input shape and precision. It is an
optimization of the deployment as a whole, not a claim that scheduling alone
tripled throughput. Replay preloads the same 1,436 lossless frames in RAM and
includes drawing/JPEG; it excludes camera delivery, network and browser work.
There were no training, calibration or compilation jobs competing during these
clean timing runs. Short screens in the preceding discussion are distinct
from these sustained measurements.

## October 16 recovery and outage log review

The saved application reports and a sudo-collected network/kernel journal
archive were retrieved after connectivity returned. They establish a Wi-Fi
disconnect followed by automatic fallback to the campus visitor network.
Tailscale's HTTPS connections then received the visitor portal's certificate,
preventing reconnection. The host continued running and logging until the
following afternoon. The earlier whole-host/GPU-hang suspicion does not fit
this outage's evidence.

Times below are October 15, 2026, in America/Chicago (CDT, UTC-05:00):

| Time | Evidence |
|---|---|
| 10:09:51 | Optimized NPU replay saved successfully: 38.31, 38.62, 38.71 FPS |
| 10:10:52 | Kernel records a roam away from the current `TAMU_WiFi` access point; authentication attempts to several campus access points are rejected with status 28 |
| 10:11:08 | NetworkManager reports `link timed out`, changes `TAMU_WiFi` from activated to failed with reason `ssid-not-found`, and removes its IP lease/routes |
| 10:11:08 | User journal records an orderly stop of GNOME Remote Desktop Handover and removal of its virtual monitor; the RDP protocol reports a server-initiated administrative disconnect |
| 10:11:09 onward | Tailscale reports the underlying link change and `network is unreachable`; Wi-Fi reconnection attempts start EAP authentication but time out |
| About 10:11:46 | Tailscale's recorded last-seen time from the outage investigation |
| 10:12:24 | NetworkManager exhausts the Wi-Fi authentication attempt and reports `no-secrets`; this follows association timeouts and does not establish that a password was wrong |
| 10:12:29-30 | NetworkManager explicitly **auto-activates `TAMU_VISITOR`**, obtains a different IP address and reports `CONNECTED_SITE` |
| 10:12:31 onward | Tailscale rejects certificates valid for `visitor.net.tamu.edu` instead of `controlplane.tailscale.com` and its DERP relay hosts |
| 10:13:24 | GPU YOLO26n baseline finished all three runs: 101.64, 95.78, 94.44 FPS |
| 10:13:25 | Comparison saved `qualified_pending_live_and_test` for the optimized NPU candidate |
| 10:16:35 | Architecture/input-matched GPU YOLOv8s comparison finished: 87.04, 83.77, 83.72 FPS |
| 10:16:39 onward | NPU session initialized for the held-out evaluation; a Python decoder exception subsequently stopped that workflow |

The same boot's journals contain desktop/update-notifier activity on October
16 at 00:20 and 11:41 CDT and repeated visitor-portal certificate failures
through 14:48:49 CDT. There is no contemporaneous kernel panic, OOM kill or GPU
reset in the collected outage window. Earlier NPU mailbox/AIE errors and an
AMDGPU workqueue warning exist on October 14; they are not evidence that a
hardware hang caused the next day's connection loss. The optional coredump
metadata query was unavailable because `coredumpctl` is not installed.

The Wi-Fi disconnection and unusable visitor fallback are established. The
underlying reason that campus access-point authentication/roaming failed
still needs campus wireless/controller logs; host logs alone cannot settle
whether that was infrastructure or client behavior. The certificate mismatch
is strong evidence of captive-portal interception. This matches
[Tailscale's documented captive-portal limitation](https://tailscale.com/docs/integrations/captive-portals).

Read-only checks on October 16 at 16:57 CDT confirm that `TAMU_WiFi` is active
and the currently saved `TAMU_VISITOR` and `TAMU_IoT` profiles already have
autoconnect disabled. No network settings were changed during this review.
Keeping visitor autoconnect disabled prevents this automatic fallback for
that profile; it cannot prevent the initial campus Wi-Fi failure. For
unattended operation, prefer an approved wired connection where available
and ask campus IT to investigate the October 15 10:10-10:12 CDT roam/EAP
failures. Guest-portal sign-in is unsuitable as the only remote recovery path.

The fresh GPU baseline averages 97.28 replay FPS, with median/p95 latency
10.48/10.89 ms. Matched GPU YOLOv8s at 576x960 averages 84.84 FPS, with
11.96/12.30 ms median/p95 latency. The NPU deployment passes its target and
frees the GPU from detection, but the GPU remains faster for detection alone.
No new hardware tests were run during this log review.

The decoder failure is separate from the connection loss:
`ValueError: Raw output must contain exactly 8 xywh/class channels`.
Sparse DFL decoding can emit an 8x8 matrix when exactly eight anchors survive
filtering; the generic raw-output decoder rejects that as ambiguous. This is
an application bug requiring a regression fix, not evidence of an OS/GPU hang.
The held-out evaluation did not produce `selected_test.json`, and the
five-minute live drive did not start. Neither is claimed complete.

Retrieved evidence is stored locally under the Git-ignored directory
`experiments/npu/outage_20261015_review/`, including `formal.log`,
`live_acceptance.log`, `gpu_replay.json`, `gpu_matched_replay.json`,
`comparison.json`, `formal_progress.json`, `user-journal-window.log`, and
`user-journal-continuation.log`. The root-collected archive
`halo-outage-20261015.EFTdJI.tgz` and its extracted `system-archive/` directory
contain the NetworkManager, wpa_supplicant, Tailscale and kernel evidence.
Its SHA-256 is
`a7d00fa9bf0ac82f3104efda53e71322ff5111a9356e3a6debe174158ead1142`.
`current-network-settings.txt` records the recovery settings. Raw logs remain
Git-ignored. The remote experiment files remain unchanged. The RDP disconnect
message does not identify a person or establish an independent cause.

The recovered host now runs kernel 7.0.0-38 rather than the benchmark's
7.0.0-31. Existing NPU runtime/cache evidence must be rechecked against this
host state before resuming deployment. The first recovery check reported
`NTPSynchronized=no`; the later collector and 16:57 check both report
`NTPSynchronized=yes`. Precise reboot/recovery wall times should still be
interpreted with that earlier synchronization gap in mind.

## October 16 resumed acceptance

Run directory on the Halo:
`experiments/npu/resume_20261016T221548Z`. The original October 15 candidate,
dataset and checkpoints are preserved. A source backup was saved before
deploying the verified 57-file resumption archive.

The sparse decoder now handles exactly eight surviving anchors using its
known channels-first layout; ambiguous external raw tensors remain rejected.
Tests compare zero, one, seven, eight, nine and many detections against dense
decoding at confidence 0.001 and 0.25, including rectangular image scaling and
four-class mapping. Five spread-out training images also passed actual CPU
dense/sparse parity after deployment. No calibration/test labels changed.

Live status now includes a completed-frame counter and processing-session ID.
The five-minute observer derives FPS from counter deltas instead of counting
sampled sequence changes. It requires >=20 FPS overall and in each complete
60-second window, no restart/reset/stall/readiness error, and camera-age growth
<=0.25 seconds. A synthetic 30-FPS stream sampled at 4 Hz passes the measurement
regression. Browser display FPS and acknowledgment latency remain separate
client observations; passive server sampling does not establish manual driving.

All **151 Python tests** passed both locally and in the actual Halo NPU
environment. Browser-control JavaScript checks passed locally, and the
extracted source archive built successfully. This includes backend failure,
stale camera, held/released controls, expiry, disconnect, bounded WebSocket
video delivery, cache/runtime changes and acceptance-promotion rejection tests.

Readiness on kernel **7.0.0-38-generic** passed: AMD's reference CNN performed
actual NPU inference with **398 NPU operators**, and XRT's latency diagnostic
passed at **50 microseconds** in default power mode. This small XRT diagnostic
is not YOLO latency. Fresh candidate compilation/execution proof reports learned
Conv computation on NPU and no learned CPU compute nodes. Firmware, drivers,
SDK packages and power mode were not changed.

The fresh shared-evaluator validation results match October 15:

| Metric | GPU YOLO26n baseline | NPU YOLOv8s A16W8 |
|---|---:|---:|
| mAP50-95 | 0.79334 | **0.84203** |
| Rock recall at 0.25 | 95.50% | 97.46% |
| Pale soil recall at 0.25 | 94.44% | 92.59% |
| Yellow crystal recall at 0.25 | 90.48% | 95.24% |
| Mudstone slab recall at 0.25 | 94.74% | 100.00% |

All automated stages completed successfully, including one full paired
100-image held-out test after the earlier decoder-aborted attempt. The model
configuration was frozen before test; no test-based tuning was performed.

| Fresh isolated replay | GPU YOLO26n 960 square | GPU YOLOv8s 576x960 | NPU YOLOv8s 576x960 |
|---|---:|---:|---:|
| FPS, run 1 | 107.10 | 86.97 | 38.30 |
| FPS, run 2 | 101.09 | 83.60 | 38.46 |
| FPS, run 3 | 99.08 | 83.58 | 38.60 |
| Mean run FPS | 102.43 | 84.72 | **38.46** |
| Median total latency, ms | 9.87 | 11.94 | 25.90 |
| p95 total latency, ms | 10.49 | 12.34 | 26.61 |

The NPU processed 6,924 frames across the three complete 60-second runs. Median
stages were preprocessing 0.596 ms, ORT execution 23.324 ms, decoding/scaling
0.920 ms, drawing 0.311 ms and JPEG encoding 0.688 ms. Stage medians need not
sum exactly to total median. Replay includes drawing/encoding and excludes
camera delivery, disk reads and browser/network work. Gazebo was not running.

During these uncapped replay runs, sampled whole-GPU busy readings averaged
about 71-72% for GPU YOLO26n, 62% for matched GPU YOLOv8s, and **0% in every
NPU-run sample**. NPU process CPU time was about 32% of one core, versus about
101-102% of one core for the GPU processes. Sampled process RSS was about
2,696 MiB for NPU versus 4,210-4,243 MiB for GPU, including the preloaded camera
recording and runtime context. These are different-throughput runs, not an
equal-30-FPS resource or power comparison. The offload proof and idle GPU
samples demonstrate that NPU detection leaves GPU compute available; they do
not establish an energy-efficiency percentage.

| Held-out test metric | GPU YOLO26n baseline | NPU YOLOv8s A16W8 |
|---|---:|---:|
| mAP50-95 | 0.72177 | **0.76665** |
| Rock recall at 0.25 | 89.60% | 92.10% |
| Pale soil recall at 0.25 | 86.84% | 85.53% |
| Yellow crystal recall at 0.25 | 88.64% | 93.18% |
| Mudstone slab recall at 0.25 | 71.43% | 80.95% |

The NPU improves held-out mAP50-95 by 4.49 percentage points. The only class
recall decrease is pale soil at 1.32 points, below the five-point limit. Both
validation and held-out gates pass with the same evaluator/thresholds. Slab's
21 test instances still limit how precisely its recall can be estimated.

Reports include `gpu_val.json`, `npu_val.json`, `gpu_replay.json`,
`matched_replay.json`, `npu_replay.json`, `comparison.json`, `gpu_test.json`,
`npu_test.json` and `test_gate.json`. `formal_job.json` ended at
`await_live_drive`; the completed reports were also downloaded locally into
the matching Git-ignored experiment directory.

Five-minute live telemetry and the human drive/stop review remain required
before an NPU default can be installed. The launcher supports a hash-bound
accepted-model record, but an explicit experimental NPU launch is not adoption.
The explicit NPU browser service is staged on localhost:8771 awaiting the
user's sudo-launched simulator. No accepted-model record has been installed.
A real Halo no-argument launcher smoke check refused that missing acceptance
record with an actionable GPU/explicit-NPU launch message. This confirms that
the staged experimental service does not silently become the default.
