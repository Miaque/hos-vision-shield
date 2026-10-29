"""Bounded, video-only observer of our own RTSP publisher; no decoding or publishing."""
from collections import deque
import hashlib
from pathlib import Path
import socket
from threading import Event, Lock, Thread
import time
from urllib.parse import urljoin, urlsplit
from uuid import uuid4


def clock_identity():
    result = {}
    for name,path in [('boot_id','/proc/sys/kernel/random/boot_id'),('namespace_offsets','/proc/self/timens_offsets')]:
        try:
            result[name] = Path(path).read_text().strip()
        except OSError:
            result[name] = None
    return result


def nal_parts(payload):
    kind = payload[0] & 31
    if kind < 24:
        return [(True, payload)]
    if kind == 24:
        result, pos = [], 1
        while pos + 2 <= len(payload):
            length = int.from_bytes(payload[pos:pos+2], 'big')
            pos += 2
            if pos + length > len(payload):
                raise ValueError('Truncated STAP-A')
            result.append((True, payload[pos:pos+length]))
            pos += length
        return result
    if kind == 28 and len(payload) >= 2:
        head = bytes([(payload[0] & 224) | (payload[1] & 31)]) if payload[1] & 128 else b''
        return [(bool(head), head + payload[2:])]
    raise ValueError('Unsupported H264 packetization')


def picture_digest(units):
    digest, found = hashlib.sha256(), False
    for unit in units:
        unit = unit.rstrip(b'\x00')
        if unit and unit[0] & 31 in (1, 5):
            digest.update(len(unit).to_bytes(4, 'big'))
            digest.update(unit)
            found = True
    return digest.hexdigest() if found else None


class RTSPObserver:
    def __init__(self, url):
        self.url, self.stop = url, Event()
        self.observer_id, self.clock = uuid4().hex, clock_identity()
        self.last_polled = time.monotonic()
        self.lock, self.frames = Lock(), deque(maxlen=256)
        self.count, self.errors, self.connections = 0, 0, 0
        self.last_error, self.sock = None, None
        self.thread = Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while not self.stop.is_set():
            try:
                self.receive()
            except (OSError, EOFError, ValueError, KeyError, IndexError) as error:
                if not self.stop.is_set():
                    with self.lock:
                        self.errors += 1
                        self.last_error = type(error).__name__
            if time.monotonic()-self.last_polled > 20:
                self.stop.set()
            if self.stop.wait(1):
                break

    def receive(self):
        parsed = urlsplit(self.url)
        with socket.create_connection((parsed.hostname, parsed.port or 554), timeout=2) as sock:
            self.sock = sock
            sock.settimeout(10)
            buf, cseq, session = b'', 0, None
            def fill(length):
                nonlocal buf
                while len(buf) < length:
                    if self.stop.is_set() or time.monotonic()-self.last_polled > 20:
                        self.stop.set()
                        raise EOFError('Observer stopped')
                    chunk = sock.recv(65536)
                    if not chunk:
                        raise EOFError('RTSP EOF')
                    buf += chunk
                    if len(buf) > 256 * 1024:
                        raise ValueError('RTSP buffer limit')

            def response():
                nonlocal buf
                while b'\r\n\r\n' not in buf:
                    fill(len(buf)+1)
                head, buf = buf.split(b'\r\n\r\n', 1)
                fields = head.decode().split('\r\n')
                if fields[0].split()[1] != '200':
                    raise ValueError('RTSP response rejected')
                headers = {k.lower(): v.strip() for k,v in (line.split(':',1) for line in fields[1:])}
                length = int(headers.get('content-length',0))
                if length > 64 * 1024:
                    raise ValueError('RTSP description limit')
                fill(length)
                body, buf = buf[:length], buf[length:]
                return headers, body

            def send(method, url, extra=''):
                nonlocal cseq
                cseq += 1
                headers = f'Session: {session}\r\n' if session else ''
                sock.sendall((f'{method} {url} RTSP/1.0\r\nCSeq: {cseq}\r\n'
                              f'{headers}{extra}\r\n').encode())

            send('DESCRIBE', self.url, 'Accept: application/sdp\r\n')
            headers, sdp = response()
            video, control = False, None
            for line in sdp.decode().splitlines():
                if line.startswith('m='):
                    video = line.startswith('m=video ')
                if video and line.startswith('a=control:'):
                    control = line.split(':',1)[1]
                    break
            if control is None:
                raise ValueError('Missing video track')
            track = urljoin(headers.get('content-base',self.url.rstrip('/')+'/'),control)
            send('SETUP',track,'Transport: RTP/AVP/TCP;unicast;interleaved=0-1\r\n')
            headers,_ = response()
            session = headers['session'].split(';')[0]
            send('PLAY',self.url)
            response()
            with self.lock:
                self.connections += 1
                connection = self.connections
            keepalive = time.monotonic()+20
            stamp, units, byte_count, first = None, [], 0, None
            while not self.stop.is_set():
                if time.monotonic() >= keepalive:
                    send('OPTIONS',self.url)
                    keepalive = time.monotonic()+20
                fill(4)
                if buf[0] != 36:
                    response()
                    continue
                channel,length = buf[1],int.from_bytes(buf[2:4],'big')
                fill(4+length)
                arrived = time.monotonic()
                packet,buf = buf[4:4+length],buf[4+length:]
                if channel != 0 or len(packet) < 12:
                    continue
                moment = int.from_bytes(packet[4:8],'big')
                if moment != stamp:
                    stamp,units,byte_count,first = moment,[],0,arrived
                offset = 12+4*(packet[0]&15)
                if packet[0]&16:
                    offset += 4+int.from_bytes(packet[offset+2:offset+4],'big')*4
                payload = packet[offset:len(packet)-packet[-1] if packet[0]&32 else len(packet)]
                if not payload:
                    continue
                for start,part in nal_parts(payload):
                    if start:
                        units.append(bytearray(part))
                    elif units:
                        units[-1].extend(part)
                byte_count += len(payload)
                if byte_count > 8 * 1024 * 1024:
                    raise ValueError('Observer picture limit')
                if packet[1]&128:
                    digest = picture_digest(units)
                    with self.lock:
                        self.count += 1
                        self.frames.append(dict(sequence=self.count,connection=connection,rtp_timestamp=stamp,
                            first_packet_monotonic=first,received_monotonic=arrived,encoded_vcl_sha256=digest))
                    stamp,units = None,[]

    def snapshot(self):
        self.last_polled = time.monotonic()
        with self.lock:
            return dict(observer_id=self.observer_id,clock=self.clock,
                        frames=list(self.frames),discarded=self.count-len(self.frames),
                        errors=self.errors,last_error=self.last_error,connections=self.connections)

    def close(self):
        self.stop.set()
        if self.sock:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.thread.join(timeout=3)
