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


def test_detector_applies_padding_once_and_preserves_input():
    from types import SimpleNamespace
    from vision_shield.detection.yolo import Detector
    face=np.zeros((1,300,21),np.float32);face[0,0,:6]=[160,240,480,400,.9,0]
    plate=np.array([[[320],[320],[320],[160],[.9]]],np.float32)
    detector=Detector.__new__(Detector);detector.box_padding=.15
    detector.models=[SimpleNamespace(get_inputs=lambda:[SimpleNamespace(name='images')],run=lambda *a:[face]),
                     SimpleNamespace(get_inputs=lambda:[SimpleNamespace(name='images')],run=lambda *a:[plate])]
    pixels=np.full((100,200,3),40,np.uint8)
    boxes=detector.detect(pixels)
    assert boxes[0]==pytest.approx((35,17.5,165,82.5))
    assert boxes[1]==pytest.approx((20,17.5,180,82.5))
    assert (pixels==40).all()
    def fail(*args):raise RuntimeError('plate inference failed')
    detector.models[1].run=fail
    with pytest.raises(RuntimeError,match='plate'):detector.detect(pixels)


def test_tile_offset_maps_back_to_source(monkeypatch):
    from types import SimpleNamespace
    from vision_shield.detection.yolo import Detector,DetailDetector
    monkeypatch.setattr(DetailDetector,'_views',lambda w,h:[(20,10,120,110)])
    face=np.zeros((1,300,21),np.float32);face[0,0,:6]=[64,64,128,128,.9,0]
    detector=Detector.__new__(Detector);detector.box_padding=0
    detector.models=[SimpleNamespace(get_inputs=lambda:[SimpleNamespace(name='images')],run=lambda *a:[face])]
    assert detector.detect(np.zeros((200,200,3),np.uint8))==[(30,20,40,30)]


@pytest.mark.parametrize('face,score,confirmation,kept,calls',[
    (False,.572,False,False,2),
    (True,.31,False,False,3),
    (False,.3,True,True,2),
    (True,.3,True,True,2),
    (False,.8,False,True,1),
    (False,.6,False,True,1),
])
def test_uncertain_candidates_need_same_model_context(face,score,confirmation,kept,calls):
    from types import SimpleNamespace
    from vision_shield.detection.yolo import Detector
    face_output=np.zeros((1,300,21),np.float32)
    plate_output=np.zeros((1,5,1),np.float32)
    if face:face_output[0,0,:6]=[160,240,480,400,score,0]
    else:plate_output[0,:,0]=[320,320,320,160,score]
    counts=[0,0]
    def model(index,output):
        def run(*args):
            counts[index]+=1
            return [output if counts[index]==1 or confirmation else np.zeros_like(output)]
        return SimpleNamespace(get_inputs=lambda:[SimpleNamespace(name='images')],run=run)
    detector=Detector.__new__(Detector);detector.box_padding=.15
    detector.models=[model(0,face_output),model(1,plate_output)]
    boxes=detector.detect(np.zeros((100,200,3),np.uint8))
    assert bool(boxes)==kept
    assert counts[0 if face else 1]==calls
    assert counts[1 if face else 0]==1


def test_context_verification_error_is_not_partial_success():
    from types import SimpleNamespace
    from vision_shield.detection.yolo import Detector
    plate=np.array([[[320],[320],[320],[160],[.3]]],np.float32)
    count=0
    def infer(*args):
        nonlocal count
        count+=1
        if count>1:raise RuntimeError('Context inference failed')
        return [plate]
    detector=Detector.__new__(Detector);detector.box_padding=.15
    detector.models=[SimpleNamespace(get_inputs=lambda:[SimpleNamespace(name='images')],run=lambda *a:[np.zeros((1,300,21),np.float32)]),
                     SimpleNamespace(get_inputs=lambda:[SimpleNamespace(name='images')],run=infer)]
    with pytest.raises(RuntimeError,match='Context inference'):detector.detect(np.zeros((100,200,3),np.uint8))


def test_context_requires_overlap_with_the_original_candidate():
    from types import SimpleNamespace
    from vision_shield.detection.yolo import Detector
    proposals=iter([np.array([[[320],[320],[320],[160],[.3]]],np.float32),
                    np.array([[[32],[170],[32],[12],[.9]]],np.float32)])
    detector=Detector.__new__(Detector);detector.box_padding=.15
    detector.models=[SimpleNamespace(get_inputs=lambda:[SimpleNamespace(name='images')],run=lambda *a:[np.zeros((1,300,21),np.float32)]),
                     SimpleNamespace(get_inputs=lambda:[SimpleNamespace(name='images')],run=lambda *a:[next(proposals)])]
    assert detector.detect(np.zeros((100,200,3),np.uint8))==[]


def test_context_overlap_uses_crop_coordinates():
    from types import SimpleNamespace
    from vision_shield.detection.yolo import Detector
    output=np.zeros((1,300,21),np.float32);output[0,0,:6]=[310,300,330,340,.9,0]
    detector=Detector.__new__(Detector)
    detector.models=[SimpleNamespace(get_inputs=lambda:[SimpleNamespace(name='images')],run=lambda *a:[output])]
    assert detector._confirmed(np.zeros((1000,1000,3),np.uint8),(400,400,420,440),0)


def test_context_does_not_confirm_a_disjoint_fragment_below_a_face():
    from types import SimpleNamespace
    from vision_shield.detection.yolo import Detector
    output=np.zeros((1,300,21),np.float32);output[0,0,:6]=[300,260,340,305,.9,0]
    detector=Detector.__new__(Detector)
    detector.models=[SimpleNamespace(get_inputs=lambda:[SimpleNamespace(name='images')],run=lambda *a:[output])]
    assert not detector._confirmed(np.zeros((1000,1000,3),np.uint8),(400,430,420,440),0)


def test_face_context_retries_a_closer_view_for_a_small_occluded_face():
    from types import SimpleNamespace
    from vision_shield.detection.yolo import Detector
    output=np.zeros((1,300,21),np.float32);output[0,0,:6]=[306,293,334,347,.3,0]
    proposals=iter([np.zeros_like(output),output])
    detector=Detector.__new__(Detector)
    detector.models=[SimpleNamespace(get_inputs=lambda:[SimpleNamespace(name='images')],run=lambda *a:[next(proposals)])]
    assert detector._confirmed(np.zeros((1000,1000,3),np.uint8),(400,400,420,440),0)
