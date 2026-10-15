#!/usr/bin/env bash
# Source this file from the project checkout. Runtime only; Quark stays in rover-export.
rover_project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
rover_npu_env="${ROVER_NPU_ENV:-$HOME/capstone-training/venvs/rover-npu}"
rover_xrt_root="${ROVER_XRT_ROOT:-/opt/xilinx/xrt}"
if [[ ! -f "$rover_xrt_root/setup.sh" ]]; then
  rover_xrt_root="$HOME/capstone-training/npu-install/xrt-root/opt/xilinx/xrt"
fi
if [[ ! -f "$rover_npu_env/bin/activate" || ! -f "$rover_xrt_root/setup.sh" ]]; then
  echo 'Missing Ryzen AI environment or XRT. Follow scripts/HALO_NPU_TESTING.md.' >&2
  return 1 2>/dev/null || exit 1
fi
source "$rover_npu_env/bin/activate"
source "$rover_xrt_root/setup.sh" >/dev/null
# Prefer matched XRT 2.25 libraries over the older coreutil bundled by VOE.
rover_boost_lib="$HOME/capstone-training/npu-install/xrt-root/usr/lib/x86_64-linux-gnu"
export LD_LIBRARY_PATH="$rover_xrt_root/lib:$rover_boost_lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$rover_project_dir/src/rover_perception:${PYTHONPATH:-}"
export CPATH="/usr/include/x86_64-linux-gnu:${CPATH:-}"
unset rover_project_dir rover_npu_env rover_xrt_root rover_boost_lib
