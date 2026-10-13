# Halo: regenerate the 500-image plan with class quotas

The first coverage500 plan had only 14 crystal-targeted training views because
the planner could switch to a soil target after a crystal pose failed. The
balanced profile requires 4 rock, 3 pale-soil and 3 yellow-crystal views in
every mixed scene. Explicit rock-only scenes keep 10 rock views; background
scenes keep 10 empty views. These are primary-target quotas, not quotas for
every incidental label in the final images.

Framing, camera dimensions, distance/size/burial ranges, depth checks and
visibility thresholds are preserved. Each retry keeps the requested class.
If a sampled mixed layout cannot meet the quotas, generation retries a
deterministically sampled layout up to 12 times, recording the layout seed
and attempt. The original scene seed still determines its train/val/test split
and held-out mesh set. An exhausted quota fails generation before capture.

Expected plan:

| Split | Images | Rock targets | Soil targets | Crystal targets | Background |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 300 | 126 | 72 | 72 | 30 |
| Val | 100 | 42 | 24 | 24 | 10 |
| Test | 100 | 42 | 24 | 24 | 10 |

Local verification completed for all 500 planned views using the cached real
assets and the production per-scene planner. All mixed scenes met their quotas
on the first sampled layout, and the quota audit returned `errors: []`. The
113-test environment/review suite passed on Linux, and 24 rover tests passed
on Windows. Docker was unavailable locally, so a new rendered capture must
still be verified on the Halo after deployment.

## Transfer the source patch

In Windows PowerShell:

```powershell
& 'C:\Windows\System32\OpenSSH\scp.exe' C:\Projects\Capstone\halo-coverage500-quota-fix-20261013.tar.gz ian@halo-strix:~/
```

On the Halo:

```bash
tar -xzf ~/halo-coverage500-quota-fix-20261013.tar.gz -C ~/capstone-training/mars-pilot50-v2
```

```bash
cd ~/capstone-training/mars-pilot50-v2
```

## Rebuild the updated Python packages

```bash
sudo docker run --rm --init \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD:/work" -w /work rover-jazzy bash -c '
    set -e
    source /opt/ros/jazzy/setup.bash
    colcon --log-base log_jazzy build --base-paths src \
      --build-base build_jazzy --install-base install_jazzy \
      --packages-select four_wheel_sim rover_training mars_training_environment
  '
```

Wait for all three packages to finish. The existing Docker image can be reused.

## Generate into a fresh balanced bundle

Keep `coverage500_scenes`, the 50-image pilot, reviewed exports and model
checkpoints for comparison. The new profile is included in the patch; the
previous `config/coverage500.json` is not overwritten.

```bash
sudo docker run -d --init --name mars-coverage500-balanced-plan --network none \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD:/work" \
  -v "$HOME/capstone-training/mars-pilot50/assets_cache:/assets_cache:ro" \
  -w /work rover-jazzy bash -c '
    set -e
    source /opt/ros/jazzy/setup.bash
    source /work/install_jazzy/setup.bash
    exec ros2 run mars_training_environment mars_generate \
      --config /work/src/mars_training_environment/config/coverage500_balanced.json \
      --asset-cache /assets_cache \
      --output /work/dataset/coverage500_balanced_scenes
  '
```

```bash
sudo docker logs --follow --tail 30 mars-coverage500-balanced-plan
```

Progress now appears at the start and completion of each scene, including
class counts and layout retries. Ctrl+C while following these logs ends the
log viewer. To stop the actual planner, use
`sudo docker stop mars-coverage500-balanced-plan`. Do not start two planners
against the same output directory.

Generation must finish with 50 scenes / 500 planned images before capture.

## Audit the quotas

In the Halo host's `rover-yolo` environment:

```bash
conda activate rover-yolo
```

```bash
PYTHONPATH="$PWD/src/mars_training_environment:$PWD/src/rover_training" \
  python -m mars_training_environment.planning \
    --bundle dataset/coverage500_balanced_scenes \
    --require-quotas --expected-frames 500
```

Require `errors: []`, 300/100/100 splits, and the exact target counts in the
table above. The audit checks target identities and each scene's quotas,
including explicit rock-only/background exceptions. It rejects an older
unbalanced plan when `--require-quotas` is supplied.

## Capture after the audit passes

```bash
sudo docker run -d --init --name mars-coverage500-balanced --network none --shm-size=1g \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e LIBGL_ALWAYS_SOFTWARE=1 \
  -v "$PWD:/work" \
  -v "$HOME/capstone-training/mars-pilot50/assets_cache:/assets_cache:ro" \
  -w /work rover-jazzy bash -c '
    set -e
    source /opt/ros/jazzy/setup.bash
    source /work/install_jazzy/setup.bash
    exec ros2 run mars_training_environment mars_dataset \
      --resume \
      --bundle /work/dataset/coverage500_balanced_scenes \
      --output /work/dataset/coverage500_balanced
  '
```

```bash
sudo docker logs --follow --tail 50 mars-coverage500-balanced
```

```bash
cat /home/ian/capstone-training/mars-pilot50-v2/dataset/coverage500_balanced/validation.json
```

Require 500 accepted frames, 300/100/100 splits and `errors: []`. This is the
rendered capture check; an offline plan alone does not establish sensor support.

## Review and export

Restart any older review server on port 8766 so it loads the new source/dataset.

```bash
PYTHONPATH="$PWD/src/mars_training_environment:$PWD/src/rover_training" \
  python -m mars_training_environment.review \
    --dataset dataset/coverage500_balanced --port 8766
```

Forward 8766 in VS Code Remote-SSH, or use this Windows tunnel:

```powershell
ssh -N -L 8766:127.0.0.1:8766 ian@halo-strix
```

Open `http://127.0.0.1:8766`. Inspect the new soil/crystal examples at 960 pixels,
including whether the yellow face is actually visible and whether partially
occluded targets are recognizable. Save corrections/exclusions and export.
Use the new export's YAML for the next training run, retaining the old model as
a baseline. Class quotas guarantee framed target selection; they do not
guarantee equal final label counts or model accuracy.

The profile retains existing seeded lighting, texture and ground-dust variation.
It does not add new crystal-dust severity controls or guaranteed occlusion bins.
