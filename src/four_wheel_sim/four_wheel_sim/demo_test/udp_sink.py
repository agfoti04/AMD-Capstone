"""Strix-side sink: sends WheelCommands to the Raspberry Pi over UDP."""
import socket

from wheel_protocol import DEFAULT_PORT, pack


class UdpSink:
    def __init__(self, host, port=DEFAULT_PORT):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.seq = 0

    def send(self, cmds, reason=''):
        self.seq = (self.seq + 1) & 0xFFFFFFFF
        self.sock.sendto(pack(self.seq, cmds), self.addr)