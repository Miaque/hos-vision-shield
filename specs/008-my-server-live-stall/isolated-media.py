"""No-model RTSP media control. Unmasked RTP stays on container loopback."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from contextlib import nullcontext
from threading import Event, Timer

from vision_shield.config import Settings
from vision_shield.media.streams import process_stream
from vision_shield.media.stream_metrics import vcl_digest


class NoModel:
    def detect(self, pixels):
        return []


def run(root, source, seconds, mode='no-model', aligned_frames=0):
    # Reuse the real-RTP test peer, with a bounded larger capture for this live window.
    spec = importlib.util.spec_from_file_location('peer', root / 'rtsp-peer.py')
    peer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(peer)
    stop, events = Event(), []
    timer = Timer(seconds, stop.set)
    config = Settings().model_dump(mode='json')
    config.update(debug=False, audio_enabled=True, redaction_mode='mosaic', detection_interval=3,
                  tracking='flow', max_bitrate=5000000)
    detector = NoModel()
    if mode != 'no-model':
        from vision_shield.detection import load_detector
        detector = load_detector(config)
        if mode == 'original':
            detector.primary.models[1]._graphed = None
    alignment = dict(preroll_video_packets=0, selected_video_packets=0, input_vcl=[])
    if aligned_frames:
        import av
        from vision_shield.media import streams
        from vision_shield.media.rtsp_observer import picture_digest
        original_open = av.open
        with original_open('/data/replay/rfm-01.mp4') as reference:
            length_size = (reference.streams.video[0].codec_context.extradata[4] & 3) + 1
            payload = next(bytes(p) for p in reference.demux(video=0) if p.size)
            units, position = [], 0
            while position < len(payload):
                length = int.from_bytes(payload[position:position + length_size], 'big')
                position += length_size
                units.append(payload[position:position + length])
                position += length
            assert position == len(payload)
            marker = picture_digest(units)
            assert marker, 'Reference must start with a complete VCL picture'

        class Reader:
            def __init__(self, inner):
                self.inner, self.streams = inner, inner.streams

            def demux(self, **kwargs):
                selected = False
                for packet in self.inner.demux(**kwargs):
                    if stop.is_set():
                        return
                    if packet.stream.type == 'video' and packet.size:
                        digest = vcl_digest(bytes(packet))
                        if not selected:
                            selected = digest == marker
                            if not selected:
                                alignment['preroll_video_packets'] += 1
                        if selected:
                            if alignment['selected_video_packets'] == aligned_frames:
                                drain = av.Packet()
                                drain.stream = self.streams.video[0]
                                yield drain
                                return
                            alignment['selected_video_packets'] += 1
                            alignment['input_vcl'].append(digest)
                    if selected:
                        yield packet

            def close(self):
                self.inner.close()

        streams.av.open = lambda url, *a, **kw: Reader(original_open(url, *a, **kw)) if url == source else original_open(url, *a, **kw)
    timer.start()
    try:
        with peer.RTSPCapture(packet_limit=16384) as capture:
            process_stream(source, capture.url, detector, config, stop, events.append)
    finally:
        stop.set()
        timer.cancel()
    pictures = capture.pictures()
    result = dict(redaction_mode=config['redaction_mode'], sdp=capture.sdp,
                  reports=capture.reports, events=events, alignment=alignment,
                  pictures=[{k: v for k, v in p.items() if k != 'payload'} for p in pictures],
                  audio=[dict(at=p['at'], timestamp=p['timestamp'], sha256=hashlib.sha256(p['payload']).hexdigest())
                         for p in capture.audio()])
    (root / f'isolated-{mode}{"-aligned" if aligned_frames else ""}.json').write_text(json.dumps(result), encoding='utf-8')
    assert pictures, 'No real RTP pictures received'
    if aligned_frames:
        assert alignment['selected_video_packets'] == aligned_frames, 'Incomplete aligned sample'
        assert len(pictures) == aligned_frames, 'Aligned sample lost decoded or transmitted pictures'


if __name__ == '__main__':
    run(Path(sys.argv[1]), sys.argv[2], float(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else 'no-model',
        int(sys.argv[5]) if len(sys.argv) > 5 else 0)
