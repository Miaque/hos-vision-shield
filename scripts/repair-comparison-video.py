"""User-authorized, lossless duplicate-clock repair; retains original evaluation failure."""
import hashlib
import importlib.util
import itertools
from pathlib import Path
import shutil
import sys

import av

spec=importlib.util.spec_from_file_location('comparison',Path(__file__).with_name('compare-yolo.py'))
comparison=importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def repair(directory):
    if (directory/'baseline').exists() or (directory/'candidate').exists():
        raise ValueError('Repair must precede either benchmark')
    records=comparison.read(directory/'manifest.json')
    samples=comparison.read(directory/'annotations.json')
    candidates=[m for m in records if m.get('clock_errors')]
    if len(candidates)!=1:
        raise ValueError('Expected exactly the confirmed duplicate-clock video')
    media=candidates[0]
    comparison.verify_source(media)
    target=directory/'repaired-input.mp4'
    changes=[]
    with av.open(media['path']) as reader, av.open(str(target),'w',format='mp4') as writer:
        streams={s.index:writer.add_stream_from_template(s) for s in reader.streams}
        last={}
        for packet in reader.demux():
            if packet.dts is None:continue
            index=packet.stream.index
            if index in last and packet.dts<=last[index]:
                if packet.dts!=last[index] or packet.pts!=packet.dts or packet.stream.type!='video':
                    raise ValueError('Not the authorized duplicate PTS/DTS case')
                changes.append(dict(old=packet.pts,new=packet.pts+1,time_base=str(packet.time_base)))
                packet.pts+=1;packet.dts+=1
            last[index]=packet.dts
            packet.stream=streams[index]
            writer.mux(packet)
    if len(changes)!=2:raise ValueError('Expected exactly two repairs')
    with av.open(media['path']) as a,av.open(str(target)) as b:
        for before,after in itertools.zip_longest(a.decode(video=0),b.decode(video=0)):
            if before is None or after is None:raise ValueError('Frame count changed')
            if not (before.to_ndarray(format='rgb24')==after.to_ndarray(format='rgb24')).all():
                raise ValueError('Decoded pixels changed')
    def audio_digest(path):
        digest=hashlib.sha256()
        with av.open(str(path)) as c:
            for p in c.demux(audio=0):
                if p.dts is not None:digest.update(bytes(p))
        return digest.hexdigest()
    if audio_digest(media['path'])!=audio_digest(target):raise ValueError('Audio payload changed')
    checked=comparison.inspect_media(target)
    if checked['clock_errors']:raise ValueError('Repair still has bad timestamps')
    record=dict(id='21',path=str(target),name=media['name']+' (修正副本)',sha256=comparison.sha(target),
                original_id=media['id'],**checked)
    records.append(record)
    for sample in list(samples):
        if sample['media_id']!=media['id']:continue
        copied=dict(sample,id='21'+sample['id'][2:],media_id='21')
        samples.append(copied)
        shutil.copyfile(directory/'samples'/(sample['id']+'-original.png'),directory/'samples'/(copied['id']+'-original.png'))
    comparison.write(directory/'repair.json',dict(original_sha256=media['sha256'],repaired_sha256=record['sha256'],
          changes=changes,decoded_pixels_equal=True,audio_payload_equal=True,frames=checked['frames']))
    comparison.write(directory/'manifest.json',records)
    comparison.write(directory/'annotations.json',samples)
    run=comparison.read(directory/'run.json');run['manifest_sha256']=comparison.sha(directory/'manifest.json')
    comparison.write(directory/'run.json',run)


if __name__=='__main__':repair(Path(sys.argv[1]).resolve())
