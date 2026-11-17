#!/usr/bin/env python3
"""Raspberry Pi wheel receiver (proof of concept: prints, no GPIO yet).

Receives WheelCommands over UDP from the Strix, converts each wheel's
rad/s into VNH5019 inputs (INA, INB, PWM duty), and prints them.

    python3 pi_wheel_receiver.py
    python3 pi_wheel_receiver.py --invert-right --max-wheel-rad-s 12

Pure standard library. Needs wheel_protocol.py next to it.
"""
import argparse
from dataclasses import dataclass
import socket
import time

from wheel_protocol import DEFAULT_PORT, WheelCommands, unpack

MOTOR_NAMES = ('front_left', 'rear_left', 'front_right', 'rear_right')


# ----------------------------------------------------------------------
# Conversion: rad/s -> INA / INB / PWM
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MotorOutput:
    ina: int          # 0 / 1
    inb: int          # 0 / 1
    duty: float       # 0.0 .. 1.0
    direction: str    # robot-frame intent: 'FWD', 'REV' or 'BRAKE'


def to_motor_output(rad_s, max_rad_s, min_duty, invert,
                    zero_threshold=0.02):
    """VNH5019 truth table used here:
        INA=1 INB=0  -> spin one way         (PWM sets speed)
        INA=0 INB=1  -> spin the other way   (PWM sets speed)
        INA=0 INB=0  -> brake to GND         (zero command)

    Open-loop: duty is linear in speed with a deadband offset (min_duty).
    `invert` flips INA/INB for motors mounted mirrored; `direction` still
    reports what the robot is being asked to do.
    """
    frac = min(abs(rad_s) / max_rad_s, 1.0)
    if frac < zero_threshold:
        return MotorOutput(0, 0, 0.0, 'BRAKE')
    duty = min_duty + (1.0 - min_duty) * frac
    robot_forward = rad_s > 0.0
    spin_a = robot_forward != invert
    return MotorOutput(1 if spin_a else 0, 0 if spin_a else 1, duty,
                       'FWD' if robot_forward else 'REV')


def compute_outputs(cmds: WheelCommands, args):
    outputs = {}
    for name in MOTOR_NAMES:
        invert = args.invert_left if 'left' in name else args.invert_right
        outputs[name] = to_motor_output(
            getattr(cmds, name), args.max_wheel_rad_s, args.min_duty, invert)
    return outputs


# ----------------------------------------------------------------------
# Backends: where the motor outputs go
# ----------------------------------------------------------------------
class PrintBackend:
    def apply(self, outputs):
        for name in MOTOR_NAMES:
            o = outputs[name]
            print(f'  {name:<12} {o.direction:<5} INA={o.ina} INB={o.inb} '
                  f'PWM={o.duty * 100:5.1f}%  (duty {o.duty:.3f})',
                  flush=True)

    def brake_all(self):
        pass


# class GpioBackend:   # future: pigpio / lgpio writes of INA, INB, PWM
#     PINS = {'front_left': (ina, inb, pwm), ...}
#     def apply(self, outputs): ...
#     def brake_all(self): ...


# ----------------------------------------------------------------------
# Receiver loop
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=DEFAULT_PORT)
    ap.add_argument('--timeout', type=float, default=0.5,
                    help='seconds without packets before braking')
    ap.add_argument('--max-wheel-rad-s', type=float, default=12.0,
                    help='wheel speed at 100%% duty (calibrate; 0.6 m/s / '
                         '0.05 m radius = 12)')
    ap.add_argument('--min-duty', type=float, default=0.12,
                    help='duty needed to just overcome static friction')
    ap.add_argument('--invert-left', action='store_true')
    ap.add_argument('--invert-right', action='store_true')
    args = ap.parse_args()

    backend = PrintBackend()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', args.port))
    sock.settimeout(0.05)
    print(f'Listening on UDP :{args.port}  (timeout {args.timeout}s)',
          flush=True)

    last_seq = None
    last_rx = time.monotonic()
    failsafe = True              # braked until the first valid packet
    last_applied = None
    count = 0

    def apply(cmds, label):
        nonlocal last_applied, count
        if cmds == last_applied:          # heartbeats don't spam the log
            return
        last_applied = cmds
        count += 1
        print(f'--- WheelCommands #{count} ({label}) ---', flush=True)
        for name in MOTOR_NAMES:
            print(f'  {name:<12} {getattr(cmds, name):+8.3f} rad/s',
                  flush=True)
        print('  -> motor outputs:', flush=True)
        backend.apply(compute_outputs(cmds, args))

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

                if last_seq is not None:
                    diff = (seq - last_seq) & 0xFFFFFFFF
                    if diff == 0 or diff >= 0x80000000:
                        continue                  # duplicate / out of order
                    if diff > 1:
                        print(f'(missed {diff - 1} packet(s))', flush=True)
                if last_seq is None:
                    print(f'First packet from {addr[0]}:{addr[1]}',
                          flush=True)

                last_seq, last_rx = seq, now
                if failsafe:
                    failsafe = False
                    last_applied = None           # force a print on resume
                apply(cmds, f'seq {seq}')

            elif not failsafe and now - last_rx > args.timeout:
                failsafe = True
                print('LINK TIMEOUT: braking all motors', flush=True)
                apply(WheelCommands(), 'failsafe')
    except KeyboardInterrupt:
        pass
    finally:
        backend.brake_all()
        sock.close()


if __name__ == '__main__':
    main()