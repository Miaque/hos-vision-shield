"""Local RTSP/TCP RECORD peer: retain actual RTP, SDP and RTCP sender clocks."""
import socket
from threading import Event, Thread
import time

from vision_shield.media.rtsp_observer import nal_parts


class RTSPCapture:
    def __init__(self):
        self.listener = socket.socket()
        self.listener.bind(('127.0.0.1', 0))
        self.listener.listen(1)
        self.listener.settimeout(.2)
        self.url = f'rtsp://127.0.0.1:{self.listener.getsockname()[1]}/capture'
        self.stop, self.closed = Event(), Event()
        self.sdp, self.packets, self.reports, self.methods, self.errors = '', [], [], [], []
        self.peer = None
        self.thread = Thread(target=self.receive, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.closed.wait(2)
        self.stop.set()
        if self.peer:
            try:
                self.peer.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.thread.join(3)
        self.listener.close()
        assert not self.thread.is_alive(), 'Local RTSP capture did not stop'
        assert not self.errors, self.errors

    def receive(self):
        try:
            while not self.stop.is_set():
                try:
                    self.peer, _ = self.listener.accept()
                    break
                except socket.timeout:
                    continue
            if not self.peer:
                return
            with self.peer as peer:
                peer.settimeout(.2)
                buf = b''
                while not self.stop.is_set():
                    try:
                        chunk = peer.recv(65536)
                    except socket.timeout:
                        continue
                    if not chunk:
                        return
                    buf += chunk
                    assert len(buf) <= 256*1024
                    while len(buf) >= 4:
                        if buf[0] == 36:
                            length = int.from_bytes(buf[2:4], 'big')
                            if len(buf) < 4+length:
                                break
                            channel, packet, buf = buf[1], buf[4:4+length], buf[4+length:]
                            at = time.monotonic()
                            assert len(self.packets) < 4096
                            if channel % 2 == 0:
                                self.packets.append(dict(channel=channel, packet=packet, at=at))
                            else:
                                while len(packet) >= 4:
                                    size = (int.from_bytes(packet[2:4], 'big')+1)*4
                                    if packet[1] == 200:
                                        self.reports.append(dict(channel=channel-1,
                                            ntp=int.from_bytes(packet[8:16], 'big')/2**32,
                                            rtp=int.from_bytes(packet[16:20], 'big')))
                                    packet = packet[size:]
                            continue
                        if b'\r\n\r\n' not in buf:
                            break
                        head, body = buf.split(b'\r\n\r\n', 1)
                        lines = head.decode().split('\r\n')
                        method = lines[0].split()[0]
                        fields = {k.lower():v.strip() for k,v in (x.split(':',1) for x in lines[1:])}
                        length = int(fields.get('content-length', 0))
                        if len(body) < length:
                            break
                        body, buf = body[:length], body[length:]
                        self.methods.append(method)
                        if method == 'ANNOUNCE':
                            self.sdp = body.decode()
                        extra = 'Public: OPTIONS, ANNOUNCE, SETUP, RECORD, TEARDOWN\r\n'
                        if 'transport' in fields:
                            extra += f'Transport: {fields["transport"]}\r\n'
                        peer.sendall((f'RTSP/1.0 200 OK\r\nCSeq: {fields["cseq"]}\r\n'
                                      f'Session: local-test\r\n{extra}Content-Length: 0\r\n\r\n').encode())
                        if method == 'TEARDOWN':
                            return
        except Exception as error:
            if not self.stop.is_set():
                self.errors.append(repr(error))
        finally:
            self.closed.set()

    def pictures(self):
        units, first, result = [], None, []
        for row in self.packets:
            if row['channel'] != 0:
                continue
            packet = row['packet']
            assert len(packet) >= 12 and packet[0] == 128
            if first is None:
                first = row['at']
            for start, payload in nal_parts(packet[12:]):
                if start:
                    units.append(bytearray(payload))
                else:
                    units[-1].extend(payload)
            if packet[1] & 128:
                result.append(dict(at=row['at'], first=first, timestamp=int.from_bytes(packet[4:8], 'big'),
                                   payload=b''.join(b'\0\0\0\1'+unit for unit in units)))
                units, first = [], None
        assert not units, 'Truncated RTP picture'
        return result

    def audio(self):
        # RFC3640 AAC-hbr: 13-bit AU size + 3-bit index, no fragmented AUs in this small fixture.
        assert 'sizelength=13' in self.sdp.lower() and 'indexlength=3' in self.sdp.lower()
        result = []
        for row in self.packets:
            if row['channel'] != 2:
                continue
            packet = row['packet']
            assert len(packet) >= 16 and packet[0] == 128
            payload = packet[12:]
            bits = int.from_bytes(payload[:2], 'big')
            assert bits and bits % 16 == 0
            header_end = 2+bits//8
            position = header_end
            for index in range(bits//16):
                size = int.from_bytes(payload[2+index*2:4+index*2], 'big') >> 3
                result.append(dict(at=row['at'], timestamp=(int.from_bytes(packet[4:8], 'big')+1024*index)%2**32,
                                   payload=payload[position:position+size]))
                position += size
            assert position == len(payload), 'Unsupported fragmented AAC fixture'
        return result

    def clock_time(self, channel, timestamp, rate):
        report = next(x for x in self.reports if x['channel'] == channel)
        delta = (timestamp-report['rtp']+2**31)%2**32-2**31
        return report['ntp']+delta/rate
