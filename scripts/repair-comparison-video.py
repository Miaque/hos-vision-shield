"""User-authorized, lossless duplicate-clock repair; retains original evaluation failure."""
import importlib.util
import itertools
from pathlib import Path
import shutil
import sys

import av

spec=importlib.util.spec_from_file_location('comparison',Path(__file__).with_name('compare-yolo.py'))
comparison=importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def repair_clock(packet, last, changes):
    index = packet.stream.index
    moment = packet.dts * packet.time_base
    if index in last and moment <= last[index]:
        if moment != last[index] or packet.pts != packet.dts or packet.stream.type not in ('video', 'audio'):
            raise ValueError('Not the authorized duplicate PTS/DTS case')
        changes.append(dict(stream=packet.stream.type, old=packet.pts, new=packet.pts+1,
                            time_base=str(packet.time_base)))
        packet.pts += 1
        packet.dts += 1
    last[index] = packet.dts * packet.time_base


def verify_packets(source, target):
    """Only the three authorized clocks may differ; compressed payloads stay exact."""
    changes = []
    # Muxing can reorder packets between streams and derive adjacent durations from DTS.
    for kind in ('video', 'audio'):
        with av.open(str(source)) as a, av.open(str(target)) as b:
            before = (p for p in a.demux(getattr(a.streams, kind)[0]) if p.dts is not None)
            after = (p for p in b.demux(getattr(b.streams, kind)[0]) if p.dts is not None)
            last = {}
            for p, q in itertools.zip_longest(before, after):
                if p is None or q is None or bytes(p) != bytes(q):
                    raise ValueError('Compressed packet payload or count changed')
                repair_clock(p, last, changes)
                if any(getattr(p, k)*p.time_base != getattr(q, k)*q.time_base for k in ('pts', 'dts')):
                    raise ValueError('Unexpected packet timestamp change')
    return changes


def repair(directory):
    if (directory/'baseline').exists() or (directory/'candidate').exists():
        raise ValueError('Repair must precede either benchmark')
    records=comparison.read(directory/'manifest.json')
    samples=comparison.read(directory/'annotations.json')
    if (directory/'repair.json').exists() or any(m.get('original_id') for m in records):
        raise ValueError('Repair already recorded')
    candidates=[m for m in records if m.get('clock_errors')]
    if len(candidates)!=1:
        raise ValueError('Expected exactly the confirmed duplicate-clock video')
    media=candidates[0]
    if media['sha256']!='db44dd993fe35ae9bbd68c4af26867b0597d994a2f21629c28d4da22020d67ca':
        raise ValueError('Source is not the user-authorized repair input')
    comparison.verify_source(media)
    target=directory/'repaired-input.mp4'
    temporary=target.with_suffix('.mp4.part')
    changes=[]
    with av.open(media['path']) as reader, av.open(str(temporary),'w',format='mp4') as writer:
        streams={s.index:writer.add_stream_from_template(s) for s in reader.streams}
        for source in reader.streams:
            streams[source.index].time_base=source.time_base
        last={}
        for packet in reader.demux():
            if packet.dts is None:continue
            index=packet.stream.index
            repair_clock(packet,last,changes)
            packet.stream=streams[index]
            writer.mux(packet)
    if [c['stream'] for c in changes]!=['video','audio','video']:
        raise ValueError('Expected exactly two video and one audio duplicate-clock repairs')
    if sorted(verify_packets(media['path'],temporary),key=lambda c:(c['stream'],c['old']))!=sorted(changes,key=lambda c:(c['stream'],c['old'])):
        raise ValueError('Repair evidence differs')
    with av.open(media['path']) as a,av.open(str(temporary)) as b:
        for before,after in itertools.zip_longest(a.decode(video=0),b.decode(video=0)):
            if before is None or after is None:raise ValueError('Frame count changed')
            if not (before.to_ndarray(format='rgb24')==after.to_ndarray(format='rgb24')).all():
                raise ValueError('Decoded pixels changed')
    checked=comparison.inspect_media(temporary)
    if checked['clock_errors']:raise ValueError('Repair still has bad timestamps')
    temporary.replace(target)
    record=dict(id='21',path=str(target),name=media['name']+' (修正副本)',sha256=comparison.sha(target),
                original_id=media['id'],**checked)
    records.append(record)
    for sample in list(samples):
        if sample['media_id']!=media['id']:continue
        copied=dict(sample,id='21'+sample['id'][2:],media_id='21')
        if 'pts' in copied:
            copied['pts'],copied['time_base']=checked['timestamps'][sample['frame_index']]
        samples.append(copied)
        shutil.copyfile(directory/'samples'/(sample['id']+'-original.png'),directory/'samples'/(copied['id']+'-original.png'))
    comparison.write(directory/'repair.json',dict(original_sha256=media['sha256'],repaired_sha256=record['sha256'],
          changes=changes,decoded_pixels_equal=True,audio_payload_equal=True,frames=checked['frames']))
    comparison.write(directory/'manifest.json',records)
    comparison.write(directory/'annotations.json',samples)
    run=comparison.read(directory/'run.json');run['manifest_sha256']=comparison.sha(directory/'manifest.json')
    run['sample_sources']=comparison.sample_sources(samples)
    comparison.write(directory/'run.json',run)


if __name__=='__main__':repair(Path(sys.argv[1]).resolve())
