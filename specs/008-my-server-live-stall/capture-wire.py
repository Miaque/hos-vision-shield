"""Container-local passive TCP/RTP metadata; never writes RTSP credentials or payloads."""
import hashlib
import json
from pathlib import Path
import socket
import struct
import sys
import time

from vision_shield.media.rtsp_observer import nal_parts, picture_digest


class Interleaved:
    def __init__(self, record):
        self.record, self.buffer, self.units = record, b'', []
        self.stamp = None

    def feed(self, data):
        self.buffer += data
        assert len(self.buffer) <= 256 * 1024, 'TCP parser budget exceeded'
        while len(self.buffer) >= 4:
            if self.buffer[0] != 36:
                # Attach mid-connection; discard only the incomplete initial RTP picture.
                start = self.buffer.find(b'$', 1)
                if start < 0:
                    self.buffer = self.buffer[-3:]
                    return
                self.buffer = self.buffer[start:]
                continue
            channel, length = self.buffer[1], int.from_bytes(self.buffer[2:4], 'big')
            if channel > 3 or length < 8:
                self.buffer = self.buffer[1:]
                continue
            if len(self.buffer) < 4 + length:
                return
            packet, self.buffer = self.buffer[4:4 + length], self.buffer[4 + length:]
            if packet[0] >> 6 != 2:
                self.record('invalid_rtp')
                continue
            if channel % 2:
                while len(packet) >= 4:
                    size = (int.from_bytes(packet[2:4], 'big') + 1) * 4
                    if packet[1] == 200 and len(packet) >= 28:
                        self.record('sr', channel=channel, ssrc=int.from_bytes(packet[4:8], 'big'),
                                    ntp=int.from_bytes(packet[8:16], 'big') / 2**32,
                                    rtp=int.from_bytes(packet[16:20], 'big'))
                    packet = packet[size:]
                continue
            assert len(packet) >= 12
            seq, stamp, ssrc = struct.unpack('!HII', packet[2:12])
            offset = 12 + 4 * (packet[0] & 15)
            if packet[0] & 16:
                offset += 4 + 4 * int.from_bytes(packet[offset + 2:offset + 4], 'big')
            payload = packet[offset:len(packet) - packet[-1] if packet[0] & 32 else len(packet)]
            if channel == 2:
                self.record('audio', timestamp=stamp, ssrc=ssrc, seq=seq,
                            payload_sha256=hashlib.sha256(payload).hexdigest())
                continue
            if stamp != self.stamp:
                self.stamp, self.units = stamp, []
            for start, unit in nal_parts(payload):
                if start:
                    self.units.append(bytearray(unit))
                elif self.units:
                    self.units[-1].extend(unit)
            if packet[1] & 128:
                self.record('frame', timestamp=stamp, ssrc=ssrc, seq=seq,
                            vcl_sha256=picture_digest(self.units))
                self.units, self.stamp = [], None


def capture(path, seconds, peer_ip, peer_port):
    end, written, flows = time.monotonic() + seconds, 0, {}
    clock_offset = time.time() - time.monotonic()
    packet_unix = time.time()
    with path.open('w', encoding='utf-8') as target:
        def record(kind, **values):
            nonlocal written
            line = json.dumps(dict(kind=kind, monotonic=packet_unix - clock_offset,
                                   unix=packet_unix, clock='kernel_SO_TIMESTAMPNS', **values)) + '\n'
            written += len(line.encode())
            assert written < 32 * 1024 * 1024, 'metadata budget exceeded'
            target.write(line)

        with socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(3)) as sock:
            sock.bind(('eth0', 0))
            sock.setsockopt(socket.SOL_SOCKET, getattr(socket, 'SO_TIMESTAMPNS', 35), 1)
            sock.settimeout(.5)
            while time.monotonic() < end:
                try:
                    packet, controls, flags, address = sock.recvmsg(262144, 128)
                except socket.timeout:
                    target.flush()
                    continue
                timestamps = [struct.unpack('qq', data[:16]) for level, kind, data in controls
                              if level == socket.SOL_SOCKET and kind == getattr(socket, 'SO_TIMESTAMPNS', 35)]
                assert len(timestamps) == 1 and not flags, 'Missing or truncated kernel timestamp'
                seconds_part, nanos = timestamps[0]
                packet_unix = seconds_part + nanos / 1e9
                ip = packet[14:]
                if len(ip) < 40 or ip[0] >> 4 != 4 or ip[9] != 6:
                    continue
                src, dst = socket.inet_ntoa(ip[12:16]), socket.inet_ntoa(ip[16:20])
                tcp = ip[(ip[0] & 15) * 4:]
                sport, dport, seq = struct.unpack('!HHI', tcp[:8])
                if (dst, dport) != (peer_ip, peer_port):
                    continue
                data = tcp[(tcp[12] >> 4) * 4:]
                if not data:
                    continue
                key = f'{src}:{sport}'
                if key not in flows:
                    assert len(flows) < 64, 'connection budget exceeded'
                    flows[key] = [seq, Interleaved(lambda kind, flow=key, **v: record(kind, flow=flow, **v))]
                expected, parser = flows[key]
                delta = (seq - expected + 2**31) % 2**32 - 2**31
                if delta > 0:
                    record('tcp_gap', flow=key, bytes=delta)
                    parser.buffer, parser.units, parser.stamp = b'', [], None
                elif delta < 0:
                    data = data[-delta:]
                if data:
                    record('tcp_payload', flow=key, size=len(data))
                    parser.feed(data)
                    flows[key][0] = (seq + len(tcp[(tcp[12] >> 4) * 4:])) % 2**32


def self_check():
    rows = []
    parser = Interleaved(lambda kind, **v: rows.append(dict(kind=kind, **v)))
    payload = b'\x65abc'
    packet = struct.pack('!BBHII', 128, 224, 1, 90000, 7) + payload
    wire = b'$\x00' + len(packet).to_bytes(2, 'big') + packet
    for byte in wire:
        parser.feed(bytes([byte]))
    assert rows == [dict(kind='frame', timestamp=90000, ssrc=7, seq=1,
                         vcl_sha256=picture_digest([payload]))]


if __name__ == '__main__':
    self_check()
    if len(sys.argv) > 1:
        capture(Path(sys.argv[1]), float(sys.argv[2]), sys.argv[3], int(sys.argv[4]))
