import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from vision_shield.detection.yolo import Detector

@pytest.mark.integration
def test_pinned_models_run_on_cpu():
    root=Path(__file__).parents[2]
    manifest=json.loads((root/'models/yolo-manifest.json').read_text('utf-8-sig'))
    paths=[]
    for model in manifest['models']:
        path=root/'models'/model['path']
        with path.open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==model['sha256']
        paths.append(path)
    detector=Detector(*paths)
    assert all(m.get_providers()==['CPUExecutionProvider'] for m in detector.models)
    assert detector.detect(np.zeros((100,200,3),np.uint8))==[]
