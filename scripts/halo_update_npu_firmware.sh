#!/usr/bin/env bash
# Install the staged upstream protocol-7 firmware for this Strix Halo NPU.
# Source: linux-firmware/amdnpu/17f0_11/npu.sbin.1.1.2.65
set -euo pipefail
firmware="${1:-/home/ian/capstone-training/npu-install/firmware/npu.sbin.1.1.2.65}"
target=/lib/firmware/amdnpu/17f0_11/npu_7.sbin
expected=3e3c996ef1e562e96ee4c4d90faa9faf1132c72da3af1bcf35d592cc34903fed
[[ "$EUID" -eq 0 ]] || { echo 'Run this reviewed helper with sudo bash.' >&2; exit 1; }
[[ "$(uname -r | cut -d. -f1)" -eq 7 ]] || { echo 'This protocol-7 helper is verified only for kernel 7.' >&2; exit 1; }
[[ -d /lib/firmware/amdnpu/17f0_11 && -c /dev/accel/accel0 ]] || { echo 'Expected the Strix Halo NPU and existing firmware directory.' >&2; exit 1; }
[[ "$(cat /sys/class/accel/accel0/device/vendor)" == 0x1022 &&
   "$(cat /sys/class/accel/accel0/device/device)" == 0x17f0 &&
   "$(cat /sys/class/accel/accel0/device/revision)" == 0x11 ]] || { echo 'This helper is only for PCI 1022:17f0 revision 11.' >&2; exit 1; }
[[ -f "$firmware" && "$(sha256sum "$firmware" | cut -d' ' -f1)" == "$expected" ]] || { echo 'Staged upstream firmware is missing or its SHA256 does not match.' >&2; exit 1; }
if [[ -e "$target" ]]; then
  [[ "$(sha256sum "$target" | cut -d' ' -f1)" == "$expected" ]] || { echo "Refusing to replace existing firmware: $target" >&2; exit 1; }
else
  install -o root -g root -m 0644 -- "$firmware" "$target"
fi
# Only reload the NPU driver. Leave the existing fallback firmware in place.
modprobe -r amdxdna
modprobe amdxdna
echo 'NPU driver reloaded. The agent can now rerun XRT and AMD reference tests.'
