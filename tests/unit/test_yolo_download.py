import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from tests.conftest import NO_WINDOW


@pytest.mark.parametrize('case',['existing','hash_mismatch','download_error'])
def test_download_never_replaces_existing_file_on_failure(tmp_path,case):
    root=Path(__file__).parents[2]
    scripts=tmp_path/'scripts';scripts.mkdir()
    models=tmp_path/'models';models.mkdir()
    script=scripts/'download-yolo-models.ps1'
    shutil.copyfile(root/'scripts'/script.name,script)
    original=b'valid-model' if case=='existing' else b'previous-model'
    target=models/'model.onnx';target.write_bytes(original)
    (models/'yolo-manifest.json').write_text(json.dumps(dict(models=[dict(path=target.name,
        source='https://example.invalid/model',sha256=hashlib.sha256(b'valid-model').hexdigest())])))
    wrapper=tmp_path/'check.ps1'
    wrapper.write_text("""param([string]$ScriptPath, [string]$Case)
$ErrorActionPreference = 'Stop'
function Invoke-WebRequest {
    param($Uri, $OutFile, $TimeoutSec)
    if ($Case -ne 'hash_mismatch') { throw 'Simulated download failure' }
    [System.IO.File]::WriteAllText($OutFile, 'wrong-content')
}
& $ScriptPath
""")
    result=subprocess.run([shutil.which('pwsh') or 'powershell','-NoProfile','-NonInteractive','-File',str(wrapper),str(script),case],
        capture_output=True,creationflags=NO_WINDOW,timeout=30)
    assert (result.returncode==0)==(case=='existing'),result.stderr.decode(errors='replace')
    assert target.read_bytes()==original
