"""Passive RTSP/TCP probe, frame-boundary RTP arrivals without libavformat buffering."""
import hashlib
import json
import re
from pathlib import Path
import socket
import struct
import sys
import threading
import time
from urllib.parse import urlsplit, urljoin

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
ROOT.mkdir(parents=True, exist_ok=True)
URLS = {'raw': 'rtsp://127.0.0.1:18555/replay/rfm-01',
        'output': 'rtsp://127.0.0.1:18554/redacted/27c39007-0aee-4185-b8fa-9d1ece4de058'}


def nal_parts(payload):
    kind = payload[0] & 31
    if kind < 24:
        return [(True, payload)]
    if kind == 24:
        result, pos = [], 1
        while pos + 2 <= len(payload):
            length = int.from_bytes(payload[pos:pos+2], 'big')
            pos += 2
            result.append((True, payload[pos:pos+length]))
            pos += length
        return result
    if kind == 28:
        head = bytes([(payload[0] & 224) | (payload[1] & 31)]) if payload[1] & 128 else b''
        return [(bool(head), head + payload[2:])]
    raise ValueError('unsupported H264 RTP packetization')


def picture_digest(units):
    digest = hashlib.sha256()
    found = False
    for unit in units:
        unit = unit.rstrip(b'\x00')
        if unit and unit[0] & 31 in (1, 5):
            digest.update(len(unit).to_bytes(4, 'big'))
            digest.update(unit)
            found = True
    return digest.hexdigest() if found else None


def probe(role, url, deadline):
    written, connection = 0, 0
    with (ROOT / f'rtp-av-{role}.jsonl').open('w', encoding='utf-8') as target:
        def record(kind, **data):
            nonlocal written
            line = json.dumps(dict(kind=kind, role=role, connection=connection,
                                   local_perf=time.perf_counter(), local_unix=time.time(), **data),
                              separators=(',', ':')) + '\n'
            written += len(line.encode())
            if written > 64 * 1024 * 1024:
                raise RuntimeError('probe byte limit reached')
            target.write(line)
            target.flush()

        while time.perf_counter() < deadline:
            connection += 1
            try:
                parsed = urlsplit(url)
                with socket.create_connection((parsed.hostname, parsed.port or 554), timeout=5) as sock:
                    sock.settimeout(10)
                    buf = b''
                    cseq, session = 0, None

                    def fill(length):
                        nonlocal buf
                        while len(buf) < length:
                            chunk = sock.recv(65536)
                            if not chunk:
                                raise EOFError('RTSP EOF')
                            buf += chunk
                            if len(buf)>256*1024:
                                raise ValueError('probe receive buffer limit')

                    def request(method, uri, headers=None):
                        nonlocal buf, cseq
                        cseq += 1
                        lines = [f'{method} {uri} RTSP/1.0', f'CSeq: {cseq}', 'User-Agent: shield-readonly-probe']
                        if session:
                            lines.append(f'Session: {session}')
                        lines.extend(f'{key}: {value}' for key, value in (headers or {}).items())
                        sock.sendall(('\r\n'.join(lines) + '\r\n\r\n').encode())
                        while b'\r\n\r\n' not in buf:
                            fill(len(buf) + 1)
                        head, rest = buf.split(b'\r\n\r\n', 1)
                        fields = head.decode().split('\r\n')
                        if fields[0].split()[1] != '200':
                            raise ValueError('RTSP response ' + fields[0])
                        response = {k.lower(): v.strip() for k, v in
                                    (line.split(':', 1) for line in fields[1:])}
                        length = int(response.get('content-length', 0))
                        buf = rest
                        fill(length)
                        body, buf = buf[:length], buf[length:]
                        return response, body

                    headers, sdp = request('DESCRIBE', url, {'Accept': 'application/sdp'})
                    tracks, current = [], None
                    for line in sdp.decode().splitlines():
                        if line.startswith('m='):
                            current={'type':line[2:].split()[0]}
                            tracks.append(current)
                        if current is not None and line.startswith('a=control:'):
                            current['control']=line.split(':',1)[1]
                        if current is not None and line.startswith('a=rtpmap:'):
                            current['rate']=int(line.split('/')[1])
                        if current is not None and line.startswith('a=fmtp:'):
                            current['fmtp']=line
                    tracks=[x for x in tracks if x['type'] in ('video','audio')]
                    base = headers.get('content-base', url.rstrip('/') + '/')
                    for index,track in enumerate(tracks):
                        control=track['control']
                        address=control if control.startswith('rtsp://') else urljoin(base,control)
                        channel=index*2
                        headers,_=request('SETUP',address,{'Transport':f'RTP/AVP/TCP;unicast;interleaved={channel}-{channel+1}'})
                        session=headers['session'].split(';')[0]
                        track['channel']=channel
                        if track['type']=='audio':
                            fields=track['fmtp'].lower()
                            if not ('sizelength=13' in fields and 'indexlength=3' in fields):
                                raise ValueError('unsupported AAC AU header')
                    by_channel={t['channel']:t for t in tracks}
                    headers, _ = request('PLAY', url)
                    record('open', clock='local_perf_counter', sdp=sdp.decode(),tracks=tracks,
                           rtp_info=headers.get('rtp-info', '').replace(url, '<stream>'))
                    timestamp, units, first, first_seq, previous_seq = None, [], None, None, None
                    packets, byte_count, lost = 0, 0, 0
                    keepalive = time.perf_counter() + 20
                    previous_audio_seq=None
                    audio_lost=0
                    while time.perf_counter() < deadline:
                        if time.perf_counter() >= keepalive:
                            cseq += 1
                            sock.sendall((f'OPTIONS {url} RTSP/1.0\r\nCSeq: {cseq}\r\n'
                                          f'Session: {session}\r\n\r\n').encode())
                            keepalive = time.perf_counter() + 20
                        fill(4)
                        if buf[0] != 36:
                            while b'\r\n\r\n' not in buf:
                                fill(len(buf) + 1)
                            head, rest = buf.split(b'\r\n\r\n', 1)
                            if not head.startswith(b'RTSP/1.0 200'):
                                raise ValueError('RTSP keepalive failed')
                            fields = dict(line.split(b':', 1) for line in head.split(b'\r\n')[1:])
                            length = int(fields.get(b'Content-Length', b'0').strip())
                            buf = rest
                            fill(length)
                            buf = buf[length:]
                            continue
                        channel, length = buf[1], int.from_bytes(buf[2:4], 'big')
                        fill(4 + length)
                        arrived = time.perf_counter()
                        packet, buf = buf[4:4+length], buf[4+length:]
                        if channel%2:
                            while len(packet)>=4:
                                size=(int.from_bytes(packet[2:4],'big')+1)*4
                                if size>len(packet) or size<4:
                                    raise ValueError('invalid RTCP size')
                                if packet[1]==200 and size>=28:
                                    record('sender_report',channel=channel-1,
                                        ntp=int.from_bytes(packet[8:16],'big')/2**32,
                                        rtp_timestamp=int.from_bytes(packet[16:20],'big'),
                                        ssrc=int.from_bytes(packet[4:8],'big'))
                                packet=packet[size:]
                            continue
                        if channel not in by_channel or len(packet) < 12:
                            continue
                        seq, stamp = struct.unpack('!HI', packet[2:8])
                        offset = 12+4*(packet[0]&15)
                        if packet[0]&16:
                            offset+=4+int.from_bytes(packet[offset+2:offset+4],'big')*4
                        payload=packet[offset:len(packet)-packet[-1] if packet[0]&32 else len(packet)]
                        if by_channel[channel]['type']=='audio':
                            if previous_audio_seq is not None:
                                audio_lost+=(seq-previous_audio_seq-1)%65536
                            previous_audio_seq=seq
                            bits=int.from_bytes(payload[:2],'big')
                            if not bits or bits%16 or 2+bits//8>len(payload):
                                raise ValueError('invalid AAC AU headers')
                            position=2+bits//8
                            for index in range(bits//16):
                                size=int.from_bytes(payload[2+index*2:4+index*2],'big')>>3
                                unit=payload[position:position+size]
                                if len(unit)!=size:
                                    raise ValueError('fragmented AAC unsupported in this probe')
                                record('audio_au',channel=channel,seq=seq,rtp_timestamp=(stamp+1024*index)%2**32,
                                       rate=by_channel[channel]['rate'],au_index=index,bytes=size,
                                       sha256=hashlib.sha256(unit).hexdigest(),sequence_gaps_total=audio_lost)
                                position+=size
                            if position!=len(payload):
                                raise ValueError('AAC payload not completely parsed')
                            continue
                        if previous_seq is not None:
                            lost += (seq - previous_seq - 1) % 65536
                        previous_seq = seq
                        if stamp != timestamp:
                            if timestamp is not None and units:
                                record('incomplete_frame', rtp_timestamp=timestamp)
                            timestamp, units, first, first_seq = stamp, [], arrived, seq
                            packets, byte_count = 0, 0
                        offset = 12 + 4 * (packet[0] & 15)
                        if packet[0] & 16:
                            offset += 4 + int.from_bytes(packet[offset+2:offset+4], 'big') * 4
                        payload = packet[offset:len(packet) - packet[-1] if packet[0] & 32 else len(packet)]
                        for start, part in nal_parts(payload):
                            if start:
                                units.append(part)
                            elif units:
                                units[-1] += part
                        packets += 1
                        byte_count += len(packet)
                        if byte_count>8*1024*1024:
                            raise ValueError('probe picture limit')
                        if packet[1] & 128:
                            record('frame', rtp_timestamp=stamp, first_packet_perf=first,
                                   last_packet_perf=arrived, first_seq=first_seq, last_seq=seq,
                                   packets=packets, bytes=byte_count, sequence_gaps_total=lost,
                                   vcl_sha256=picture_digest(units))
                            timestamp, units = None, []
            except (OSError, EOFError, ValueError) as error:
                record('error', error_type=type(error).__name__, detail=str(error)[:120])
                print(role, type(error).__name__, str(error)[:120], flush=True)
                time.sleep(1)
    print(role, 'RTP capture complete', flush=True)


if __name__ == '__main__':
    # Check single NAL, STAP-A, and FU-A against the repository's actual output digest.
    from vision_shield.media.stream_metrics import vcl_digest
    unit = b'\x65abc'
    assert nal_parts(unit) == [(True, unit)]
    assert nal_parts(b'\x78\x00\x04' + unit) == [(True, unit)]
    a, b = nal_parts(b'\x7c\x85ab')[0], nal_parts(b'\x7c\x45c')[0]
    assert a[1] + b[1] == unit
    assert picture_digest([unit]) == vcl_digest(b'\x00\x00\x00\x01' + unit)
    end = time.perf_counter() + (int(sys.argv[2]) if len(sys.argv) > 2 else 180)
    threads = [threading.Thread(target=probe, args=(role, url, end)) for role, url in URLS.items()]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
