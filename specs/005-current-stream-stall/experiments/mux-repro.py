"""Isolated localhost RTSP mux experiment; never connects to the deployed service."""
from collections import defaultdict
from fractions import Fraction
import json
from pathlib import Path
import socket
import struct
from threading import Event, Thread
import time

import av
import numpy as np

ROOT=Path(__file__).resolve().parent


def trial(options):
    server=socket.socket()
    server.bind(('127.0.0.1',0))
    server.listen(1)
    port=server.getsockname()[1]
    received=[]
    done=Event()
    errors=[]
    def receive():
        try:
            with server,server.accept()[0] as peer:
                peer.settimeout(3)
                buf=b''
                while not done.is_set():
                    chunk=peer.recv(65536)
                    if not chunk:break
                    buf+=chunk
                    while len(buf)>=4:
                        if buf[0]==36:
                            length=int.from_bytes(buf[2:4],'big')
                            if len(buf)<length+4:break
                            channel=buf[1];packet=buf[4:length+4];buf=buf[length+4:]
                            if channel==0 and len(packet)>=12 and packet[1]&128:
                                received.append(dict(at=time.perf_counter(),rtp=int.from_bytes(packet[4:8],'big')))
                            continue
                        if b'\r\n\r\n' not in buf:break
                        head,body=buf.split(b'\r\n\r\n',1)
                        lines=head.decode().split('\r\n');method=lines[0].split()[0]
                        fields={k.lower():v.strip() for k,v in (x.split(':',1) for x in lines[1:])}
                        length=int(fields.get('content-length',0))
                        if len(body)<length:break
                        buf=body[length:]
                        transport=fields.get('transport')
                        extra='Public: OPTIONS, ANNOUNCE, SETUP, RECORD, TEARDOWN, GET_PARAMETER\r\n'
                        if transport:extra+='Transport: '+transport+'\r\n'
                        peer.sendall((f'RTSP/1.0 200 OK\r\nCSeq: {fields["cseq"]}\r\nSession: test\r\n'
                                      f'{extra}Content-Length: 0\r\n\r\n').encode())
                        if method=='TEARDOWN':return
        except Exception as error:
            errors.append(type(error).__name__)
    thread=Thread(target=receive,daemon=True);thread.start()
    muxed=[]
    options={'rtsp_transport':'tcp',**options}
    with av.open(f'rtsp://127.0.0.1:{port}/test','w',format='rtsp',options=options) as writer:
        video=writer.add_stream('libx264',rate=25)
        video.width,video.height,video.pix_fmt=64,48,'yuv420p'
        video.options={'preset':'ultrafast','tune':'zerolatency','bf':'0'}
        video.time_base=video.codec_context.time_base=Fraction(1,25)
        audio=writer.add_stream('aac',rate=8000);audio.layout='mono'
        pending=[]
        for i in range(40):
            frame=av.AudioFrame(format='fltp',layout='mono',samples=1024)
            frame.sample_rate,frame.pts,frame.time_base=8000,i*1024,Fraction(1,8000)
            for plane in frame.planes:plane.update(bytes(plane.buffer_size))
            pending.extend(audio.encode(frame))
        pending.extend(audio.encode())
        started=time.perf_counter()
        for i in range(100):
            # Same ordering as write_output: due audio first, then video. Hold audio only in this synthetic window.
            if not 25<=i<50:
                while pending and pending[0].dts*pending[0].time_base<=Fraction(i,25):
                    writer.mux(pending.pop(0))
            frame=av.VideoFrame.from_ndarray(np.full((48,64,3),i,dtype=np.uint8),format='rgb24')
            frame.pts,frame.time_base=i,Fraction(1,25)
            for packet in video.encode(frame):writer.mux(packet)
            muxed.append(dict(frame=i,at=time.perf_counter()))
            delay=started+(i+1)/25-time.perf_counter()
            if delay>0:time.sleep(delay)
        for packet in video.encode():writer.mux(packet)
    done.set();thread.join(4)
    gaps=[b['at']-a['at'] for a,b in zip(received,received[1:])]
    return dict(options=options,frames=len(received),max_receive_gap_seconds=max(gaps,default=None),
                gaps_over_500ms=sum(x>.5 for x in gaps),errors=errors,muxed=muxed,received=received)


if __name__=='__main__':
    results=[]
    for options in ({},{'flush_packets':'1'},{'max_interleave_delta':'100000'}):
        result=trial(options);results.append(result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('muxed','received')}),flush=True)
    (ROOT/'mux-repro-results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
