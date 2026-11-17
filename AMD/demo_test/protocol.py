"""Wire format shared by the Strix (sender) and the Raspberry Pi (receiver).

Packet, little-endian, 23 bytes:
    2s  magic  b'WC'
    B   version
    I   sequence number (increments every packet, wraps at 2**32)
    4f  front_left, rear_left, front_right, rear_right   (rad/s)

Needs only the Python standard library, so no ROS install is required on
the Pi.
"""
from dataclasses import dataclass
import struct

MAGIC = b'WC'
VERSION = 1
DEFAULT_PORT = 5005
_FMT = struct.Struct('<2sBI4f')


@dataclass(frozen=True)
class WheelCommands:
    """Angular velocity command per wheel, rad/s (+ = robot forward)."""
    front_left: float = 0.0
    rear_left: float = 0.0
    front_right: float = 0.0
    rear_right: float = 0.0


def pack(seq: int, c) -> bytes:
    return _FMT.pack(MAGIC, VERSION, seq & 0xFFFFFFFF,
                     c.front_left, c.rear_left, c.front_right, c.rear_right)


def unpack(data: bytes):
    """Return (seq, WheelCommands). Raises ValueError on a bad packet."""
    if len(data) != _FMT.size:
        raise ValueError(f'bad length {len(data)} (expected {_FMT.size})')
    magic, version, seq, fl, rl, fr, rr = _FMT.unpack(data)
    if magic != MAGIC:
        raise ValueError('bad magic')
    if version != VERSION:
        raise ValueError(f'unsupported version {version}')
    # float32 round trip: clean it up so 6.0 stays 6.0 in comparisons
    r = lambda x: round(x, 3)
    return seq, WheelCommands(r(fl), r(rl), r(fr), r(rr))