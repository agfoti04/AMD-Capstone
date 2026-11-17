"""Arm wire format shared by the Strix (sender) and the Raspberry Pi.

Packet, little-endian, 27 bytes:
    2s  magic  b'AC'      (wheel packets use b'WC', so they can't be mixed up)
    B   version
    I   sequence number (increments every packet, wraps at 2**32)
    5f  joint_1, joint_2, joint_3, joint_4, joint_6   (radians)

Standard library only.
"""
from dataclasses import dataclass
import struct

MAGIC = b'AC'
VERSION = 1
DEFAULT_PORT = 5006          # wheels use 5005
ARM_JOINTS = ('joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_6')
_FMT = struct.Struct('<2sBI5f')


@dataclass(frozen=True)
class ArmCommands:
    """Commanded joint positions, radians."""
    joint_1: float = 0.0
    joint_2: float = 0.0
    joint_3: float = 0.0
    joint_4: float = 0.0
    joint_6: float = 0.0

    @staticmethod
    def from_joint_dict(d):
        """Build from e.g. {'Joint_1': 0.7275, ...}. Case-insensitive.
        Raises KeyError if a joint is missing, so a partial message can
        never silently command a joint to 0."""
        lower = {k.lower(): float(v) for k, v in d.items()}
        return ArmCommands(*(lower[j] for j in ARM_JOINTS))


def pack(seq: int, c: ArmCommands) -> bytes:
    return _FMT.pack(MAGIC, VERSION, seq & 0xFFFFFFFF,
                     *(getattr(c, j) for j in ARM_JOINTS))


def unpack(data: bytes):
    """Return (seq, ArmCommands). Raises ValueError on a bad packet."""
    if len(data) != _FMT.size:
        raise ValueError(f'bad length {len(data)} (expected {_FMT.size})')
    magic, version, seq, *vals = _FMT.unpack(data)
    if magic != MAGIC:
        raise ValueError('bad magic')
    if version != VERSION:
        raise ValueError(f'unsupported version {version}')
    # Round away float32 noise; "+ 0.0" turns -0.0 into 0.0
    return seq, ArmCommands(*(round(v, 4) + 0.0 for v in vals))