"""真实本机TCP握手、RTP解包、容量与退出；不需要模型或外部服务器。"""
import socket
import struct
from threading import Event, Thread
import time

import pytest

from vision_shield.media.rtsp_observer import RTSPObserver, nal_parts, picture_digest
from vision_shield.media.stream_metrics import vcl_digest


def test_packetization_matches_the_publisher_digest_and_rejects_truncation():
    unit = b'\x65abc'
    assert nal_parts(unit) == [(True,unit)]
    assert nal_parts(b'\x78\x00\x04'+unit) == [(True,unit)]
    assert nal_parts(b'\x7c\x85ab')[0][1]+nal_parts(b'\x7c\x45c')[0][1] == unit
    assert picture_digest([bytearray(unit)]) == vcl_digest(b'\x00\x00\x01'+unit)
    with pytest.raises(ValueError,match='Truncated'):
        nal_parts(b'\x78\x00\x04\x65')


def test_real_rtsp_reader_is_bounded_and_stops_its_socket():
    complete, exit_server = Event(), Event()
    server = socket.socket()
    server.bind(('127.0.0.1',0))
    server.listen(1)
    port = server.getsockname()[1]
    methods = []
    def serve():
        with server,server.accept()[0] as client:
            client.settimeout(2)
            buf = b''
            for _ in range(3):
                while b'\r\n\r\n' not in buf:
                    buf += client.recv(4096)
                request,buf = buf.split(b'\r\n\r\n',1)
                method = request.decode().split()[0]
                methods.append(method)
                body = (b'v=0\r\nm=video 0 RTP/AVP 96\r\na=control:trackID=0\r\n'
                        if method=='DESCRIBE' else b'')
                client.sendall((f'RTSP/1.0 200 OK\r\nCSeq: {len(methods)}\r\nSession: test\r\n'
                    f'Content-Length: {len(body)}\r\n\r\n').encode()+body)
            for sequence in range(1,301):
                packet=struct.pack('!BBHII',128,224,sequence,sequence*3600,1)+b'\x65abc'
                client.sendall(b'$\x00'+len(packet).to_bytes(2,'big')+packet)
            complete.set()
            exit_server.wait(3)
    thread=Thread(target=serve)
    thread.start()
    observer=RTSPObserver(f'rtsp://127.0.0.1:{port}/redacted/test')
    try:
        assert complete.wait(3)
        deadline=time.monotonic()+2
        while time.monotonic()<deadline:
            data=observer.snapshot()
            if data['frames'] and data['frames'][-1]['sequence']==300:
                break
            time.sleep(.01)
        assert methods==['DESCRIBE','SETUP','PLAY']
        assert len(data['frames'])==256 and data['discarded']==44
        assert data['frames'][-1]['encoded_vcl_sha256']==vcl_digest(b'\x00\x00\x01\x65abc')
    finally:
        observer.close()
        exit_server.set()
        thread.join(3)
    assert not observer.thread.is_alive()


async def test_api_observers_use_only_our_output_and_evict_after_two_tasks(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from uuid import UUID
    import httpx
    from vision_shield.config import Settings
    from vision_shield.main import create_app
    from vision_shield.media import rtsp_observer
    created = []
    class Observer:
        def __init__(self,url):
            self.url,self.closed=url,False
            self.thread=SimpleNamespace(is_alive=lambda:True)
            created.append(self)
        def close(self):
            self.closed=True
        def snapshot(self):
            return {'frames':[]}
    monkeypatch.setattr(rtsp_observer,'RTSPObserver',Observer)
    settings=Settings(_env_file=None,debug=True,rtsp_publish_base_url='rtsp://mediamtx:8554')
    app=create_app(settings)
    app.state.settings=settings
    app.state.store=SimpleNamespace(get=AsyncMock(return_value=dict(kind='stream',stream_attempts=[{}],
        device_profile={'mode':'live'})))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
        for number in (1,1,2,3):
            r=await client.get(f'/v1/tasks/{UUID(int=number)}/diagnostics')
            assert r.status_code==200 and r.json()['receiver']=={'frames':[]}
    assert len(created)==3 and created[0].closed
    assert len(app.state.live_receivers)==2
    assert created[0].url==f'rtsp://mediamtx:8554/redacted/{UUID(int=1)}'
