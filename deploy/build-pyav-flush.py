"""Add only interleaver flushing to the pinned test PyAV output extension."""
from pathlib import Path
import os
import subprocess
import sys
import tarfile
import zipfile

import av


root = Path('/native')
assert av.__version__ == '19.0.1' and av.ffmpeg_version_info == '9.0.2'
for wheel in root.glob('*.whl'):
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(root / 'tools')
sys.path.insert(0, str(root / 'tools'))
for name in ('av.tar.gz', 'ffmpeg.tar.gz'):
    with tarfile.open(root / name) as archive:
        archive.extractall(root, filter='data')
headers, source = root / 'FFmpeg-n9.0.2', root / 'av-19.0.1'
subprocess.run([str(headers / 'configure'), '--disable-everything', '--disable-x86asm',
                '--disable-autodetect', '--disable-programs', '--disable-doc'], cwd=headers, check=True)
path = source / 'av/container/output.py'
text = path.read_text()
marker = '    def mux(self, packets):'
assert text.count(marker) == 1 and 'def flush_mux(' not in text
# Retain PyAV's existing open, timeout, concurrent-close and FFmpeg error handling.
method = '''    def flush_mux(self):
        """Release accepted interleaved packets without flushing encoders."""
        self._assert_open()
        self.start_encoding()
        ret: cython.int
        arm_write_timeout(self)
        self._blocking_depth += 1
        try:
            with cython.nogil:
                ret = lib.av_interleaved_write_frame(self.ptr, cython.NULL)
        finally:
            self._blocking_depth -= 1
            self.set_timeout(None)
        self.err_check(ret)

'''
path.write_text(text.replace(marker, method + marker))
import Cython
assert Cython.__version__ == '3.3.0'
libraries = [p for p in (Path(av.__file__).parent.parent / 'av.libs').glob('lib*.so*')
             if p.name.split('-')[0] in ('libavformat', 'libavcodec', 'libavutil', 'libavfilter',
                                         'libavdevice', 'libswresample', 'libswscale')]
assert len(libraries) == 7
sdk = root / 'sdk'
(sdk / 'lib').mkdir(parents=True)
(sdk / 'include').symlink_to(headers, target_is_directory=True)
for library in libraries:
    (sdk / 'lib' / (library.name.split('-')[0] + '.so')).symlink_to(library)
# PyAV's shared Cython module and limited-API flags require rebuilding all bindings together.
environment = dict(os.environ, PYTHONPATH=str(root / 'tools'),
                   LDFLAGS=f'-Wl,-rpath,{Path(av.__file__).parent.parent / "av.libs"}')
subprocess.run([sys.executable, 'setup.py', 'build_ext', '-j4', '--build-lib', str(root / 'patched'),
                '--ffmpeg-dir=' + str(sdk)], cwd=source, env=environment, check=True)
