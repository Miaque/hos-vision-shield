import numpy as np
import pytest
from vision_shield.config import Settings


def test_yolo_selection_and_cpu_only(monkeypatch):
    from vision_shield.detection import load_detector, yolo
    received=[]
    monkeypatch.setattr(yolo,'Detector',lambda *a: received.append(a) or 'yolo')
    cfg=Settings(_env_file=None,detection_backend='yolo',face_model_path='face.onnx',plate_model_path='plate.onnx')
    assert load_detector(cfg.model_dump())=='yolo'
    assert str(received[0][0])=='face.onnx'
    with pytest.raises(ValueError):
        Settings(_env_file=None,detection_backend='yolo',inference_device='cuda')


def test_yolo_letterbox_inverse_and_rgb():
    from vision_shield.detection.yolo import prepare, decode
    rgb=np.zeros((100,200,3),np.uint8); rgb[:]=(255,64,0)
    blob,geometry=prepare(rgb,640)
    assert blob.shape==(1,3,640,640)
    assert blob[0,:,320,320]==pytest.approx([1,64/255,0])
    # source (50,25)-(150,75), scaled 3.2 plus top pad 160
    rows=np.array([[[320],[320],[320],[160],[.9]]],np.float32)
    detections=decode(rows,geometry,200,100)
    assert detections[0][0]==pytest.approx((50,25,150,75))
    assert detections[0][1]==pytest.approx(.9)


@pytest.mark.parametrize('bad',[np.zeros((0,2,3),np.uint8),np.zeros((2,2),np.uint8),np.zeros((2,2,3),float)])
def test_invalid_pixels_fail(bad):
    from vision_shield.detection.yolo import prepare
    with pytest.raises(ValueError): prepare(bad,640)


def test_invalid_network_output_is_not_no_targets():
    from vision_shield.detection.yolo import decode
    for output in [np.zeros((1,6,10)),np.full((1,5,2),np.nan),np.array([[[1],[1],[2],[2],[1.1]]])]:
        with pytest.raises(ValueError): decode(output,(1,1,0,0),10,10)


def test_empty_and_overlapping_boxes():
    from vision_shield.detection.yolo import decode
    rows=np.array([[[10,10],[10,10],[10,10],[10,10],[.9,.8]]],np.float32)
    assert len(decode(rows,(1,1,0,0),20,20))==1
    rows[:,4]=.1
    assert decode(rows,(1,1,0,0),20,20)==[]


def test_face_post_nms_padding_rows_and_class_validation():
    from vision_shield.detection.yolo import decode
    output=np.zeros((1,300,21),np.float32)
    output[0,0,:6]=[2,3,8,9,.9,0]
    assert decode(output,(1,1,0,0),10,10,face=True)[0][0]==(2,3,8,9)
    output[0,0,5]=1
    with pytest.raises(ValueError): decode(output,(1,1,0,0),10,10,face=True)
