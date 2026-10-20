import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import av
from threading import Event
from vision_shield.config import Settings
from vision_shield.detection.yolo import Detector

@pytest.mark.integration
def test_pinned_models_run_on_cpu(generated_media,tmp_path):
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
    from vision_shield.media.images import process_image
    from vision_shield.media.video import process_video,masked_frame
    from vision_shield.redaction import FrameRedactor
    process_image(generated_media/'no-target.jpeg',tmp_path/'image.jpg',detector,100000)
    config=Settings(_env_file=None).model_dump()
    assert process_video(generated_media/'silent.mp4',tmp_path/'video.mp4',detector,config,Event(),lambda e:None)==24
    frame=av.VideoFrame.from_ndarray(np.zeros((100,200,3),np.uint8),format='rgb24')
    from fractions import Fraction
    frame.pts=0;frame.time_base=Fraction(1,90000)
    assert masked_frame(frame,FrameRedactor(detector)).pts==0
