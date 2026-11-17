"""Strix-side sink: sends ArmCommands to the Raspberry Pi over UDP."""
import socket

from protocol import DEFAULT_PORT, ArmCommands, pack


class ArmUdpSink:
    def __init__(self, host, port=DEFAULT_PORT):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.seq = 0

    def send(self, cmds: ArmCommands):
        self.seq = (self.seq + 1) & 0xFFFFFFFF
        self.sock.sendto(pack(self.seq, cmds), self.addr)

    def send_joints(self, joint_dict):
        """Convenience: {'Joint_1': rad, ...} -> ArmCommands -> UDP."""
        self.send(ArmCommands.from_joint_dict(joint_dict))