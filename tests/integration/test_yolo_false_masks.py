"""Opt-in replay of false masks and a small partially occluded face."""

import os
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from vision_shield.detection.yolo import Detector


@pytest.mark.integration
def test_background_masks_removed_without_losing_faces_or_plate():
    directory=os.getenv('YOLO_REGRESSION_SOURCE')
    if not directory:
        pytest.skip('Set YOLO_REGRESSION_SOURCE to the local original-image directory')
    root=Path(__file__).parents[2]
    detector=Detector(root/'models/yolov8n-face.onnx',root/'models/yolov11s-plate.onnx')
    cases=[
        ('0ab3b44e62f5fa5740ebc36b1a85216a48f05b9890ac8bbd772c205c2d298427.jpg',
         [(3000,300),(3700,480)],[(790,445),(1600,590)]),
        ('d3326728f1e40f2c749d4e669288d8b17f1dc27e13801649021937cf12ef5c56.jpg',
         [(3000,300),(267,487)],[(2440,410)]),
        ('20261014_145251_607523_visual_events.jpg',[],[(1750,420),(1800,530)]),
    ]
    incorrect=[]
    for filename,background,targets in cases:
        pixels=np.array(Image.open(Path(directory)/filename).convert('RGB'))
        assert pixels.shape==(1080,3840,3)
        boxes=detector.detect(pixels)
        def covered(point):
            x,y=point
            return any(x1<=x<x2 and y1<=y<y2 for x1,y1,x2,y2 in boxes)
        incorrect.extend((filename,point) for point in background if covered(point))
        assert all(covered(point) for point in targets),'A real face or plate lost its mask'
    assert not incorrect,f'Non-target background is masked: {incorrect}'
