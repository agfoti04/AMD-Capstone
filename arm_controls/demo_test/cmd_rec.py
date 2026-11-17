#!/usr/bin/env python3
"""Raspberry Pi arm receiver (proof of concept: prints, no I2C yet).

Receives ArmCommands over UDP from the Strix, converts each joint angle
into a servo angle, pulse width and PCA9685 tick count, and prints them.

    python3 pi_arm_receiver.py
    python3 pi_arm_receiver.py --min-pulse-us 600 --max-pulse-us 2400

Standard library only. Needs arm_protocol.py next to it.
Edit the JOINTS table below for your wiring and servo calibration.
"""
import argparse
from dataclasses import dataclass
import math
import socket
import time

from arm_protocol import ARM_JOINTS, DEFAULT_PORT, ArmCommands, unpack


# ----------------------------------------------------------------------
# Per-joint configuration: EDIT THESE
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class JointConfig:
    channel: int               # PCA9685 channel 0-15
    center_deg: float = 90.0   # servo angle that corresponds to joint = 0
    sign: int = 1              # -1 if the servo turns the opposite way
    min_deg: float = -90.0     # joint-space limits (software safety stops)
    max_deg: float = 90.0


JOINTS = {
    'joint_1': JointConfig(channel=0),
    'joint_2': JointConfig(channel=1),
    'joint_3': JointConfig(channel=2),
    'joint_4': JointConfig(channel=3),
    'joint_6': JointConfig(channel=4),
}


# ----------------------------------------------------------------------
# Conversion: joint radians -> servo angle -> pulse -> PCA9685 ticks
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ServoOutput:
    channel: int
    cmd_rad: float
    cmd_deg: float       # as received
    servo_deg: float     # after limits, in servo coordinates
    pulse_us: float
    ticks: int           # PCA9685 "off" count out of 4096 (on count = 0)
    limited: bool        # True if a limit changed the command


def to_servo_output(cmd_rad, cfg: JointConfig, args):
    cmd_deg = math.degrees(cmd_rad)
    joint_deg = max(cfg.min_deg, min(cfg.max_deg, cmd_deg))
    servo_deg = cfg.center_deg + cfg.sign * joint_deg
    servo_clamped = max(0.0, min(args.servo_range_deg, servo_deg))
    limited = (joint_deg != cmd_deg) or (servo_clamped != servo_deg)

    span = args.max_pulse_us - args.min_pulse_us
    pulse_us = args.min_pulse_us + servo_clamped / args.servo_range_deg * span
    ticks = round(pulse_us * args.freq * 4096 / 1e6)
    ticks = max(0, min(4095, ticks))
    return ServoOutput(cfg.channel, cmd_rad, cmd_deg, servo_clamped,
                       pulse_us, ticks, limited)


def compute_outputs(cmds: ArmCommands, args):
    return {j: to_servo_output(getattr(cmds, j), JOINTS[j], args)
            for j in ARM_JOINTS}


# ----------------------------------------------------------------------
# Backends: where the servo outputs go
# ----------------------------------------------------------------------
class PrintBackend:
    def apply(self, outputs):
        print(f'  {"joint":<10} {"ch":>2} {"servo [deg]":>12} '
              f'{"pulse [us]":>11} {"ticks":>6}', flush=True)
        for j in ARM_JOINTS:
            o = outputs[j]
            flag = '  <- LIMIT' if o.limited else ''
            print(f'  {j.capitalize():<10} {o.channel:>2} '
                  f'{o.servo_deg:>12.2f} {o.pulse_us:>11.1f} '
                  f'{o.ticks:>6}{flag}', flush=True)


# class Pca9685Backend:   # future
#     def __init__(self, freq): ...   # set PCA9685 prescale for `freq`
#     def apply(self, outputs):
#         for o in outputs.values():
#             set channel o.channel: on=0, off=o.ticks


# ----------------------------------------------------------------------
# Receiver loop
# ----------------------------------------------------------------------
def print_received(cmds: ArmCommands, count, label):
    print(f'--- ArmCommands #{count} ({label}) ---', flush=True)
    print(f'  {"joint":<10} {"cmd [rad]":>10} {"cmd [deg]":>10}',
          flush=True)
    for j in ARM_JOINTS:
        rad = getattr(cmds, j)
        print(f'  {j.capitalize():<10} {rad:>+10.4f} '
              f'{math.degrees(rad):>+10.2f}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=DEFAULT_PORT)
    ap.add_argument('--timeout', type=float, default=1.0,
                    help='seconds without packets before warning')
    ap.add_argument('--freq', type=float, default=50.0,
                    help='PCA9685 PWM frequency, Hz')
    ap.add_argument('--servo-range-deg', type=float, default=180.0,
                    help='servo travel between min and max pulse')
    ap.add_argument('--min-pulse-us', type=float, default=500.0)
    ap.add_argument('--max-pulse-us', type=float, default=2500.0)
    args = ap.parse_args()

    backend = PrintBackend()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', args.port))
    sock.settimeout(0.05)
    print(f'Listening on UDP :{args.port}  (link warning after '
          f'{args.timeout}s)', flush=True)

    last_seq = None
    last_rx = time.monotonic()
    link_lost = False
    last_applied = None
    count = 0

    try:
        while True:
            try:
                data, addr = sock.recvfrom(64)
            except socket.timeout:
                data = None
            now = time.monotonic()

            if data is not None:
                try:
                    seq, cmds = unpack(data)
                except ValueError as e:
                    print(f'Dropped bad packet from {addr[0]}: {e}',
                          flush=True)
                    continue

                if last_seq is None:
                    print(f'First packet from {addr[0]}:{addr[1]}',
                          flush=True)
                else:
                    diff = (seq - last_seq) & 0xFFFFFFFF
                    if diff == 0 or diff >= 0x80000000:
                        continue                  # duplicate / out of order
                    if diff > 1:
                        print(f'(missed {diff - 1} packet(s))', flush=True)

                last_seq, last_rx = seq, now
                if link_lost:
                    link_lost = False
                    print('Link restored', flush=True)

                if cmds != last_applied:          # heartbeats stay quiet
                    last_applied = cmds
                    count += 1
                    print_received(cmds, count, f'seq {seq}')
                    print('  -> servo outputs:', flush=True)
                    backend.apply(compute_outputs(cmds, args))

            elif (last_seq is not None and not link_lost
                  and now - last_rx > args.timeout):
                # Servos have no brake: relaxing them would let the arm
                # drop, so the failsafe is to HOLD the last position.
                link_lost = True
                print('LINK TIMEOUT: holding last servo positions',
                      flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()


if __name__ == '__main__':
    main()