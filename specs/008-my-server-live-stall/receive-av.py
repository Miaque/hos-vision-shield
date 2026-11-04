"""Read both RTP tracks and RTCP SR directly; persist only clocks and content hashes."""
import hashlib
import json
from pathlib import Path
import socket
import struct
import sys
from threading import Thread
import time
from urllib.parse import urljoin, urlsplit

from vision_shield.media.rtsp_observer import nal_parts, picture_digest


def receive(path, url, seconds):
    assert 0 < seconds <= 600
    deadline, connection, written = time.monotonic() + seconds, 0, 0
    with path.open('w', encoding='utf-8') as output:
        def record(kind, **data):
            nonlocal written
            line = json.dumps(dict(kind=kind, connection=connection, monotonic=time.monotonic(),
                                   unix=time.time(), **data)) + '\n'
            written += len(line.encode())
            assert written <= 32 * 1024 * 1024, 'metadata budget exceeded'
            output.write(line)

        while time.monotonic() < deadline:
            connection += 1
            try:
                parsed = urlsplit(url)
                with socket.create_connection((parsed.hostname, parsed.port or 554), timeout=5) as sock:
                    sock.settimeout(5)
                    buffer, session, cseq = b'', None, 0

                    def fill(length):
                        nonlocal buffer
                        while len(buffer) < length:
                            data = sock.recv(65536)
                            if not data:
                                raise EOFError
                            buffer += data
                            assert len(buffer) <= 256 * 1024, 'RTSP receive budget exceeded'

                    def response():
                        nonlocal buffer
                        while b'\r\n\r\n' not in buffer:
                            fill(len(buffer) + 1)
                        head, buffer = buffer.split(b'\r\n\r\n', 1)
                        lines = head.decode().split('\r\n')
                        assert lines[0].split()[1] == '200', 'RTSP request rejected'
                        headers = {k.lower(): v.strip() for k, v in (x.split(':', 1) for x in lines[1:])}
                        length = int(headers.get('content-length', 0))
                        assert length < 65536
                        fill(length)
                        body, buffer = buffer[:length], buffer[length:]
                        return headers, body

                    def send(method, target, extra=''):
                        nonlocal cseq
                        cseq += 1
                        headers = f'Session: {session}\r\n' if session else ''
                        sock.sendall((f'{method} {target} RTSP/1.0\r\nCSeq: {cseq}\r\n{headers}{extra}\r\n').encode())

                    send('DESCRIBE', url, 'Accept: application/sdp\r\n')
                    headers, sdp = response()
                    tracks, track = [], None
                    for line in sdp.decode().splitlines():
                        if line.startswith('m='):
                            track = dict(kind=line[2:].split()[0])
                            tracks.append(track)
                        elif track is not None and line.startswith('a=control:'):
                            track['control'] = urljoin(headers.get('content-base', url.rstrip('/') + '/'), line[10:])
                        elif track is not None and line.startswith('a=rtpmap:'):
                            track['rate'] = int(line.split('/')[1])
                            track['latm'] = 'mp4a-latm' in line.lower()
                    assert len(tracks) in (1, 2) and tracks[0]['kind'] == 'video'
                    for index, track in enumerate(tracks):
                        send('SETUP', track['control'],
                             f'Transport: RTP/AVP/TCP;unicast;interleaved={2*index}-{2*index+1}\r\n')
                        reply, _ = response()
                        session = reply['session'].split(';')[0]
                    send('PLAY', url)
                    response()
                    record('open', sdp=sdp.decode())
                    keepalive, stamp, units, sequences = time.monotonic() + 20, None, [], {}
                    while time.monotonic() < deadline:
                        if time.monotonic() >= keepalive:
                            send('OPTIONS', url)
                            keepalive = time.monotonic() + 20
                        fill(4)
                        if buffer[0] != 36:
                            response()
                            continue
                        channel, size = buffer[1], int.from_bytes(buffer[2:4], 'big')
                        fill(4 + size)
                        packet, buffer = buffer[4:4 + size], buffer[4 + size:]
                        if channel % 2:
                            while len(packet) >= 4:
                                length = 4 * (int.from_bytes(packet[2:4], 'big') + 1)
                                if packet[1] == 200 and len(packet) >= 28:
                                    record('sr', channel=channel, ssrc=int.from_bytes(packet[4:8], 'big'),
                                           ntp=int.from_bytes(packet[8:16], 'big') / 2**32,
                                           timestamp=int.from_bytes(packet[16:20], 'big'))
                                packet = packet[length:]
                            continue
                        assert len(packet) >= 12 and packet[0] >> 6 == 2
                        seq, timestamp, ssrc = struct.unpack('!HII', packet[2:12])
                        if channel in sequences and (seq - sequences[channel]) % 65536 != 1:
                            record('sequence_gap', channel=channel, previous=sequences[channel], seq=seq)
                        sequences[channel] = seq
                        offset = 12 + 4 * (packet[0] & 15)
                        if packet[0] & 16:
                            offset += 4 + 4 * int.from_bytes(packet[offset + 2:offset + 4], 'big')
                        payload = packet[offset:len(packet) - packet[-1] if packet[0] & 32 else len(packet)]
                        if tracks[channel // 2]['kind'] == 'audio':
                            if tracks[channel // 2].get('latm'):
                                position, length = 0, 0
                                while True:
                                    part = payload[position]
                                    position, length = position + 1, length + part
                                    if part != 255:
                                        break
                                assert packet[1] & 128 and len(payload) - position == length, 'Fragmented LATM sample'
                                record('audio', ssrc=ssrc, timestamp=timestamp,
                                       sha256=hashlib.sha256(payload[position:]).hexdigest())
                                output.flush()
                                continue
                            bits = int.from_bytes(payload[:2], 'big')
                            assert bits and bits % 16 == 0, 'Unsupported AAC packetization'
                            position = 2 + bits // 8
                            for index in range(bits // 16):
                                length = int.from_bytes(payload[2 + index*2:4 + index*2], 'big') >> 3
                                unit, position = payload[position:position + length], position + length
                                assert len(unit) == length
                                record('audio', ssrc=ssrc, timestamp=(timestamp + 1024*index) % 2**32,
                                       sha256=hashlib.sha256(unit).hexdigest())
                            assert position == len(payload)
                        else:
                            if timestamp != stamp:
                                stamp, units = timestamp, []
                            for start, unit in nal_parts(payload):
                                if start:
                                    units.append(bytearray(unit))
                                elif units:
                                    units[-1].extend(unit)
                            if packet[1] & 128:
                                record('frame', timestamp=stamp, ssrc=ssrc, vcl_sha256=picture_digest(units))
                                units, stamp = [], None
                        output.flush()
            except (OSError, EOFError, AssertionError, ValueError) as error:
                record('error', error_type=type(error).__name__)
                output.flush()
                time.sleep(min(1, max(0, deadline - time.monotonic())))


if __name__ == '__main__':
    root = Path(sys.argv[1])
    root.mkdir(parents=True, exist_ok=True)
    threads = [Thread(target=receive, args=(root / f'{role}.jsonl', url, float(sys.argv[2])))
               for role, url in zip(('raw', 'output'), sys.argv[3:], strict=True)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
