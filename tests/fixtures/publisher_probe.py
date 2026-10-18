"""Local MediaMTX probe; no application implementation or user permissions."""

import base64
import json
import os
import shutil
import socket
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import yaml

from tests.conftest import NO_WINDOW, ROOT, stop_process, unused_port, wait_for_port


class PublisherProbe:
    def __init__(self, directory, redis_url=None):
        binary = os.getenv("TEST_MEDIAMTX_BIN") or shutil.which("mediamtx")
        if not binary:
            local = ROOT / ".tools/mediamtx" / ("mediamtx.exe" if os.name == "nt" else "mediamtx")
            if local.is_file():
                binary = str(local)
        if not binary:
            raise RuntimeError("Set TEST_MEDIAMTX_BIN to a MediaMTX executable")
        self.directory, self.binary = directory, binary
        from redis import Redis

        self.control = Redis.from_url(redis_url, decode_responses=True) if redis_url else None
        self.generation = "g1"
        self.approved = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.session_ids = set()
        self.rtsp_port, self.api_port = unused_port(), unused_port()
        probe = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                accepted = payload["action"] == "publish" and payload["user"] == probe.generation
                if accepted:
                    probe.session_ids.add(payload["id"])
                    probe.approved.set()
                    probe.release.wait(10)
                self.send_response(204 if accepted else 401)
                self.end_headers()

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def generation(self):
        return self.control.get("probe:generation") if self.control else self._generation

    @generation.setter
    def generation(self, value):
        if self.control:
            if value is None:
                self.control.delete("probe:generation")
            else:
                self.control.set("probe:generation", value)
        else:
            self._generation = value

    def __enter__(self):
        self.thread.start()
        config = yaml.safe_load((ROOT / "deploy/mediamtx.yml").read_text("utf-8"))
        config.update(
            rtspAddress=f"127.0.0.1:{self.rtsp_port}",
            apiAddress=f"127.0.0.1:{self.api_port}",
            authMethod="http",
            authHTTPAddress=f"http://127.0.0.1:{self.server.server_port}/check",
            authHTTPExclude=[{"action": "api"}, {"action": "read"}],
        )
        path = self.directory / "mediamtx.yml"
        path.write_text(yaml.safe_dump(config), "utf-8")
        self.log = (self.directory / "mediamtx.log").open("wb")
        self.process = subprocess.Popen(
            [self.binary, str(path)], stdout=self.log, stderr=self.log,
            cwd=self.directory, creationflags=NO_WINDOW,
        )
        try:
            wait_for_port(self.process, self.rtsp_port)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *args):
        self.release.set()
        stop_process(self.process)
        self.log.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        if self.control:
            self.control.close()

    def kick(self, session_id):
        return httpx.post(
            f"http://127.0.0.1:{self.api_port}/v3/rtspsessions/kick/{session_id}",
            timeout=5,
        )

    def announce(self, generation):
        sock = socket.create_connection(("127.0.0.1", self.rtsp_port), timeout=5)
        sock.settimeout(5)
        credentials = base64.b64encode(f"{generation}:probe".encode()).decode()
        sdp = (
            "v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=probe\r\n"
            "c=IN IP4 127.0.0.1\r\nt=0 0\r\nm=video 0 RTP/AVP 96\r\n"
            "a=rtpmap:96 H264/90000\r\na=fmtp:96 packetization-mode=1\r\n"
            "a=control:trackID=0\r\n"
        ).encode()
        headers = (
            f"ANNOUNCE rtsp://127.0.0.1:{self.rtsp_port}/redacted/probe RTSP/1.0\r\n"
            f"CSeq: 1\r\nAuthorization: Basic {credentials}\r\n"
            f"Content-Type: application/sdp\r\nContent-Length: {len(sdp)}\r\n\r\n"
        ).encode()
        sock.sendall(headers + sdp)
        return sock


def read_headers(sock):
    response = b""
    while b"\r\n\r\n" not in response:
        chunk = sock.recv(4096)
        if not chunk:
            return response
        response += chunk
        if len(response) > 16384:
            raise RuntimeError("Oversized RTSP response")
    return response
