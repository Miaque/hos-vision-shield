"""Actual RTSP output, independently received video/AAC and encoded-clock checks."""
from fractions import Fraction
from itertools import pairwise
import base64
import json
import os
from pathlib import Path
import re
import threading
import time

import av
import numpy as np
import pytest

from tests.rtsp_capture import RTSPCapture
from vision_shield.config import Settings
from vision_shield.media import streams


@pytest.fixture(scope='module')
def interleave_source(tmp_path_factory):
    path = tmp_path_factory.mktemp('interleave-source')/'input.nut'
    with av.open(str(path), 'w') as writer:
        video = writer.add_stream('libx264', rate=25)
        video.width, video.height, video.pix_fmt = 64, 48, 'yuv420p'
        video.options = {'preset':'ultrafast','tune':'zerolatency','bf':'0'}
        audio = writer.add_stream('aac', rate=8000)
        audio.layout = 'mono'
        for index in range(100):
            frame = av.VideoFrame.from_ndarray(np.full((48,64,3),80+index,np.uint8), format='rgb24')
            frame.pts, frame.time_base = index, Fraction(1,25)
            for packet in video.encode(frame):
                writer.mux(packet)
            samples = np.arange(320)+index*320
            wave = (.3*np.sin(samples*2*np.pi*440/8000)).astype(np.float32).reshape(1,-1)
            frame = av.AudioFrame.from_ndarray(wave, format='fltp', layout='mono')
            frame.pts, frame.time_base, frame.sample_rate = index*320, Fraction(1,8000), 8000
            for packet in audio.encode(frame):
                writer.mux(packet)
        for track in (video,audio):
            for packet in track.encode():
                writer.mux(packet)
    return path


def run_output(source, scenario, monkeypatch, *, default_interleave=False):
    original_open = av.open
    stop = threading.Event()
    audio_input, wire_clocks, events = [], [], []
    stop_at = None

    class Reader:
        def __init__(self):
            self.container = original_open(str(source))
            self.streams = self.container.streams

        def demux(self, **kwargs):
            video_count = audio_count = 0
            held = []
            started = time.monotonic()
            for packet in self.container.demux(**kwargs):
                if packet.stream.type == 'video' and packet.size:
                    remaining = started+video_count/25-time.monotonic()
                    if remaining > 0:
                        stop.wait(remaining)
                    if video_count == 50:
                        yield from held
                        held.clear()
                    video_count += 1
                if packet.stream.type == 'audio' and packet.size:
                    audio_count += 1
                    if scenario == 'sparse' and audio_count % 4 != 1:
                        continue
                if packet.size and scenario in ('jump','reset') and packet.pts*packet.time_base >= 2:
                    shift = round(Fraction(1 if scenario=='jump' else -1)/packet.time_base)
                    packet.pts += shift
                    packet.dts += shift
                if packet.stream.type == 'audio' and packet.size:
                    audio_input.append((bytes(packet),packet.pts*packet.time_base))
                    if scenario == 'missing' and 25 <= video_count < 50:
                        held.append(packet)
                        continue
                yield packet
            yield from held

        def close(self):
            self.container.close()

    class Writer:
        def __init__(self, inner):
            self.inner = inner

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def __enter__(self):
            self.inner.__enter__()
            return self

        def __exit__(self, *args):
            return self.inner.__exit__(*args)

        def mux(self, packet):
            wire_clocks.append(dict(track=packet.stream.type,pts=packet.pts*packet.time_base,
                                    dts=packet.dts*packet.time_base,payload=bytes(packet)))
            self.inner.mux(packet)

    def open_endpoint(url, *args, **kwargs):
        if url == 'rtsp://fixture/input':
            return Reader()
        if default_interleave:
            kwargs['options'] = {k:v for k,v in kwargs['options'].items() if k!='max_interleave_delta'}
        return Writer(original_open(url, *args, **kwargs))

    class Detector:
        def detect(self, pixels):
            return [(0,0,32,48)]

    def emit(event):
        nonlocal stop_at
        events.append(event)
        if scenario == 'stop' and event.get('stats',{}).get('processed_frames',0) >= 50 and not stop.is_set():
            stop_at = time.monotonic()
            stop.set()

    monkeypatch.setattr(streams.av,'open',open_endpoint)
    config = Settings(_env_file=None,debug=True,stream_diagnostics=False).model_dump()
    config.update(inference_device='cpu',detection_interval=1,tracking='template')
    with RTSPCapture() as capture:
        streams.process_stream('rtsp://fixture/input',capture.url,Detector(),config,stop,emit)
    output = os.getenv('INTERLEAVE_RESULT_DIR')
    if output:
        root = Path(output);root.mkdir(parents=True,exist_ok=True)
        pictures, audio = capture.pictures(),capture.audio()
        result = dict(scenario=scenario,default_interleave=default_interleave,video_frames=len(pictures),
                      mux_audio_packets=sum(x['track']=='audio' for x in wire_clocks),rtp_audio_aus=len(audio),
                      max_rtp_gap_seconds=max(b['at']-a['at'] for a,b in pairwise(pictures)),
                      audio_prefix_matches=[x['payload'] for x in audio]==[x[0] for x in audio_input[:len(audio)]],
                      rtcp_reports=capture.reports)
        (root/f'{scenario}-{"default" if default_interleave else "configured"}-wire.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return capture,audio_input,wire_clocks,events,None if stop_at is None else time.monotonic()-stop_at


@pytest.mark.parametrize('default_interleave',[False,True],ids=['100ms','default'])
@pytest.mark.parametrize('scenario',['continuous','sparse','missing','jump','reset','stop'])
def test_rtsp_interleave_keeps_video_audio_and_sync(interleave_source,monkeypatch,scenario,default_interleave):
    capture,source_audio,clocks,events,stop_seconds = run_output(interleave_source,scenario,monkeypatch,
                                                             default_interleave=default_interleave)
    pictures,audio = capture.pictures(),capture.audio()
    final = events[-1]
    count = final['stats']['processed_frames']
    assert len(pictures) == count
    assert count == 100 if scenario != 'stop' else 50 <= count < 100
    assert final['stats']['dropped_frames'] == final['stats']['stream']['failed_frames'] == 0
    assert final['stream_end']['reason'] == ('stop_requested' if scenario=='stop' else 'source_eof')
    assert 'TEARDOWN' in capture.methods
    if stop_seconds is not None:
        assert stop_seconds < 2
    video_clocks = [x for x in clocks if x['track']=='video']
    audio_clocks = [x for x in clocks if x['track']=='audio']
    for track in (video_clocks,audio_clocks):
        assert all(b['pts']>a['pts'] and b['dts']>a['dts'] for a,b in pairwise(track))
    # Default and 100ms both retain one final AAC AU at RTSP teardown in this fixture.
    # Lock the exact tail deficit and byte-identical contiguous prefix; do not call mux acceptance delivery.
    assert len(audio) > 0 and len(audio_clocks)-len(audio) == 1
    assert [x['payload'] for x in audio] == [x['payload'] for x in audio_clocks[:len(audio)]]
    assert [x['payload'] for x in audio] == [x[0] for x in source_audio[:len(audio)]]
    for track,received,rate in ((video_clocks,pictures,90000),(audio_clocks,audio,8000)):
        offset = (received[0]['timestamp']-round(track[0]['pts']*rate))%2**32
        assert all((r['timestamp']-round(p['pts']*rate))%2**32==offset for r,p in zip(received,track))
    decoder = av.CodecContext.create('h264','r')
    parameter_sets = re.search(r'sprop-parameter-sets=([^;\r\n]+)',capture.sdp)[1]
    decoder.extradata = b''.join(b'\0\0\0\1'+base64.b64decode(value) for value in parameter_sets.split(','))
    frames = []
    for picture in pictures:
        frames.extend(decoder.decode(av.Packet(picture['payload'])))
    frames.extend(decoder.decode(None))
    assert len(frames) == count
    for index,frame in enumerate(frames):
        pixels = frame.to_ndarray(format='rgb24')
        assert pixels[:,:24].mean() < 70
        assert abs(pixels[:,40:].mean()-(80+index)) < 5
    decoder = av.CodecContext.create('aac','r')
    decoder.extradata = bytes.fromhex(re.search(r'config=([0-9a-f]+)',capture.sdp,re.I)[1])
    decoded_audio = [frame for item in audio for frame in decoder.decode(av.Packet(item['payload']))]
    assert len(decoded_audio) == len(audio)
    wave = np.concatenate([x.to_ndarray().reshape(-1) for x in decoded_audio])
    assert np.sqrt(np.mean(wave**2)) > .05
    # RTCP sender reports map the two unrelated RTP clocks onto their common sender time.
    video_zero = capture.clock_time(0,pictures[0]['timestamp'],90000)-float(video_clocks[0]['pts'])
    audio_zero = capture.clock_time(2,audio[0]['timestamp'],8000)-float(audio_clocks[0]['pts'])
    assert abs(video_zero-audio_zero) < .002
    assert abs(float(audio_clocks[-1]['pts']-video_clocks[-1]['pts'])) < .6
    gaps = [b['at']-a['at'] for a,b in pairwise(pictures)]
    if scenario in ('continuous','sparse','missing') and not default_interleave:
        assert max(gaps) < .5, f'Video withheld after mux accepted pictures: {max(gaps):.3f}s'
    if scenario == 'missing' and default_interleave:
        assert max(gaps) > .5, 'Control did not reproduce the original interleave stall'
    output = os.getenv('INTERLEAVE_RESULT_DIR')
    if output:
        root = Path(output);root.mkdir(parents=True,exist_ok=True)
        result = dict(scenario=scenario,default_interleave=default_interleave,frames=count,aac_packets=len(audio),
                      aac_tail_aus_not_received=len(audio_clocks)-len(audio),max_rtp_gap_seconds=max(gaps),
                      av_sender_clock_offset_seconds=video_zero-audio_zero,stop_seconds=stop_seconds,
                      input_aac_packets=len(source_audio),timestamp_repairs=final['stats']['stream']['timestamp_repairs'])
        (root/f'{scenario}-{"default" if default_interleave else "100ms"}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


def test_rtsp_output_reconnects_with_fresh_clocks(interleave_source,monkeypatch):
    for _ in range(2):
        with monkeypatch.context() as patch:
            capture,_,_,events,_ = run_output(interleave_source,'continuous',patch)
        assert len(capture.pictures()) == 100
        assert events[-1]['stream_end']['reason'] == 'source_eof'
