"""Bounded independent receive/decode evidence; this is not a player render clock."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from threading import Thread
import time

import av
import numpy as np

ROOT = Path(sys.argv[1]).resolve()
ROOT.mkdir(parents=True,exist_ok=True)
DURATION = int(sys.argv[2]) if len(sys.argv)>2 else 120
URLS = {'raw':'rtsp://127.0.0.1:18555/replay/rfm-01',
        'output':'rtsp://127.0.0.1:18554/redacted/27c39007-0aee-4185-b8fa-9d1ece4de058'}


def receive(role):
    end = time.monotonic()+DURATION
    connection, written = 0,0
    with (ROOT/f'av-{role}.jsonl').open('w',encoding='utf-8') as handle:
        def save(kind,**data):
            nonlocal written
            line=json.dumps(dict(kind=kind,connection=connection,local_monotonic=time.monotonic(),
                                 local_unix=time.time(),**data),separators=(',',':'))+'\n'
            written+=len(line.encode())
            if written>32*1024*1024:raise RuntimeError('AV capture capacity reached')
            handle.write(line);handle.flush()

        while time.monotonic()<end:
            connection+=1
            counts=Counter()
            try:
                with av.open(URLS[role],options={'rtsp_transport':'tcp','analyzeduration':'100000','probesize':'32768'},
                             timeout=(5,8)) as reader:
                    save('open',tracks=[dict(type=s.type,codec=s.codec_context.name,time_base=str(s.time_base),
                         rate=getattr(s,'rate',None),width=getattr(s.codec_context,'width',None),
                         height=getattr(s.codec_context,'height',None)) for s in reader.streams])
                    next_image=0
                    for packet in reader.demux():
                        if time.monotonic()>=end:break
                        save('packet',track=packet.stream.type,pts=packet.pts,dts=packet.dts,time_base=str(packet.time_base),
                             duration=packet.duration,bytes=packet.size,sha256=hashlib.sha256(bytes(packet)).hexdigest())
                        started=time.monotonic()
                        for frame in packet.decode():
                            counts[packet.stream.type]+=1
                            fields=dict(track=packet.stream.type,pts=frame.pts,time_base=str(frame.time_base),
                                        decode_seconds=time.monotonic()-started)
                            if packet.stream.type=='audio':
                                samples=frame.to_ndarray().astype(np.float64)
                                fields.update(samples=frame.samples,rate=frame.sample_rate,rms=float(np.sqrt(np.mean(samples*samples))))
                            elif time.monotonic()>=next_image:
                                name=f'{role}-{connection}-{counts["video"]}.png'
                                frame.to_image().save(ROOT/name)
                                fields['image']=name
                                next_image=time.monotonic()+15
                            save('decoded',**fields)
                    save('closed',counts=dict(counts))
            except (av.error.FFmpegError,OSError) as error:
                save('error',type=type(error).__name__,counts=dict(counts))
                time.sleep(1)
    print(role,'AV receive/decode complete',flush=True)


if __name__=='__main__':
    threads=[Thread(target=receive,args=(role,)) for role in URLS]
    for thread in threads:thread.start()
    for thread in threads:thread.join()
