import importlib.util
from copy import deepcopy
from pathlib import Path
from threading import Event

import av
import pytest

from vision_shield.config import Settings
from vision_shield.media.video import process_video

spec=importlib.util.spec_from_file_location('comparison',Path(__file__).parents[2]/'scripts/compare-yolo.py')
compare=importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


@pytest.mark.parametrize('name',['silent.mp4','aac.mp4','vfr.mp4'])
def test_complete_media_and_audio_integrity(generated_media,tmp_path,name):
    class EmptyDetector:
        def detect(self,pixels):return []
    source=generated_media/name
    output=tmp_path/name
    process_video(source,output,EmptyDetector(),Settings(_env_file=None).model_dump(),Event(),lambda e:None)
    original,checked=compare.inspect_media(source),compare.inspect_media(output)
    compare.verify_output(original,checked)
    truncated=deepcopy(checked);truncated['frames']-=1
    with pytest.raises(ValueError,match='completeness'):compare.verify_output(original,truncated)
    shifted=deepcopy(checked);shifted['times'][0]+=.01
    with pytest.raises(ValueError,match='timestamp'):compare.verify_output(original,shifted)
    if original['audio']:
        shifted=deepcopy(checked);shifted['audio_packets'][0][0]+=.01
        with pytest.raises(ValueError,match='audio'):compare.verify_output(original,shifted)
        changed=deepcopy(checked);changed['audio_sha256']='bad'
        with pytest.raises(ValueError,match='audio'):compare.verify_output(original,changed)


def test_worker_records_hold_boxes_without_changing_production_pixels(generated_media,tmp_path,monkeypatch):
    from vision_shield import detection
    class BriefDetector:
        def __init__(self):self.count=0
        def detect(self,pixels):
            self.count+=1
            return [(10,10,30,30)] if self.count==1 else []
    monkeypatch.setattr(detection,'load_detector',lambda config:BriefDetector())
    source=generated_media/'silent.mp4'
    media=dict(id='00',name=source.name,path=str(source),sha256=compare.sha(source),**compare.inspect_media(source))
    samples=[dict(id=f'00-{i:06d}',media_id='00',frame_index=i,kind='video',width=320,height=240,
                  targets=[],review_status='pending') for i in [0,1,2,3]]
    (tmp_path/'samples').mkdir()
    compare.write(tmp_path/'run.json',dict(sample_sources=compare.sample_sources(samples)))
    compare.write(tmp_path/'manifest.json',[media])
    compare.write(tmp_path/'annotations.json',samples)
    compare.worker(tmp_path,'candidate')
    result=compare.read(tmp_path/'candidate/00.json')
    assert result['status']=='complete'
    assert [bool(e['boxes']) for e in result['evidence']]==[True,True,True,False]
    control=tmp_path/'control.mp4'
    process_video(source,control,BriefDetector(),Settings(_env_file=None).model_dump(),Event(),lambda e:None)
    assert compare.sha(control)==compare.sha(tmp_path/result['output'])
    assert all((tmp_path/'samples'/f'{s["id"]}-candidate.png').exists() for s in samples)


def test_black_source_pixels_are_not_detection_evidence():
    sample=dict(id='00-000000',media_id='00',frame_index=0,kind='image',width=10,height=10,
                targets=[dict(id='face',kind='face',polygon=[[1,1],[3,1],[3,3]])],
                review_status='human_confirmed',reviewer='tester',reviewed_at='2026-10-20T10:00:00+08:00')
    evidence=[dict(sample_id=sample['id'],variant=v,boxes=[]) for v in ('baseline','candidate')]
    result=compare.summarize([sample],evidence)
    assert result['variants']['baseline']['image']['face']['missed']==1


def test_image_worker_writes_complete_result_and_sample(tmp_path,monkeypatch):
    from PIL import Image
    from vision_shield import detection
    detections=[]
    class EmptyDetector:
        def detect(self,pixels):detections.append(1);return []
    monkeypatch.setattr(detection,'load_detector',lambda config:EmptyDetector())
    source=tmp_path/'原图.jpg';Image.new('RGB',(10,10),'black').save(source)
    (tmp_path/'samples').mkdir()
    media=dict(id='00',path=str(source),sha256=compare.sha(source),**compare.inspect_media(source))
    samples=[dict(id='00-000000',media_id='00',frame_index=0,targets=[],review_status='pending',kind='image',width=10,height=10)]
    compare.write(tmp_path/'run.json',dict(sample_sources=compare.sample_sources(samples)))
    compare.write(tmp_path/'manifest.json',[media])
    compare.write(tmp_path/'annotations.json',samples)
    compare.worker(tmp_path,'candidate')
    record=compare.read(tmp_path/'candidate/00.json')
    assert record['status']=='complete'
    assert record['evidence'][0]['boxes']==[]
    assert (tmp_path/'samples/00-000000-candidate.png').is_file()
    compare.worker(tmp_path,'candidate')
    assert len(detections)==1
    (tmp_path/record['output']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='output changed'):compare.worker(tmp_path,'candidate')


@pytest.mark.parametrize('error',[RuntimeError('inference failed'),KeyboardInterrupt()])
def test_worker_failure_and_cancel_restore_redaction_and_never_complete(tmp_path,monkeypatch,error):
    from PIL import Image
    from vision_shield import detection,redaction
    import vision_shield.media.images as images
    originals=redaction.redact,images.redact
    class BrokenDetector:
        def detect(self,pixels):raise error
    monkeypatch.setattr(detection,'load_detector',lambda config:BrokenDetector())
    source=tmp_path/'source.jpg';Image.new('RGB',(10,10),'black').save(source)
    compare.write(tmp_path/'run.json',dict(sample_sources=[]))
    compare.write(tmp_path/'manifest.json',[dict(id='00',path=str(source),sha256=compare.sha(source),**compare.inspect_media(source))])
    compare.write(tmp_path/'annotations.json',[])
    with pytest.raises(KeyboardInterrupt if isinstance(error,KeyboardInterrupt) else SystemExit):compare.worker(tmp_path,'candidate')
    assert compare.read(tmp_path/'candidate/00.json')['status']==('cancelled' if isinstance(error,KeyboardInterrupt) else 'failed')
    assert (redaction.redact,images.redact)==originals
    assert not (tmp_path/'candidate/00.jpg').exists()


def test_prepare_freezes_exact_inventory_and_vfr_samples(generated_media,tmp_path):
    from PIL import Image
    import shutil
    source=tmp_path/'中文素材';source.mkdir()
    for i in range(16):Image.new('RGB',(10,10),'black').save(source/f'{i:02d}.jpg')
    for i in range(5):shutil.copyfile(generated_media/'vfr.mp4',source/f'{i:02d}.mp4')
    directory=tmp_path/'run'
    compare.prepare(source,directory)
    records=compare.read(directory/'manifest.json')
    assert len(records)==21
    assert all(m['frames']==24 for m in records if m['kind']=='video')
    samples=compare.read(directory/'annotations.json')
    assert len(samples)==26 and all(s['review_status']=='pending' for s in samples)
    compare.verify_samples(directory,samples)
    with pytest.raises(ValueError,match='exists'):compare.prepare(source,directory)
    with pytest.raises(ValueError,match='separate'):compare.prepare(source,source/'output')
    (source/'00.jpg').unlink()
    with pytest.raises(ValueError,match='16 JPEGs'):compare.prepare(source,tmp_path/'other-run')


def test_images_only_prepare_never_opens_videos(tmp_path):
    from PIL import Image
    source=tmp_path/'source';source.mkdir()
    for i in range(16):Image.new('RGB',(10,10),'black').save(source/f'{i:02d}.jpg')
    for i in range(5):(source/f'{i:02d}.mp4').write_bytes(b'Video must not be decoded')
    directory=tmp_path/'images'
    compare.prepare(source,directory,images_only=True)
    records=compare.read(directory/'manifest.json')
    assert len(records)==16 and all(m['kind']=='image' for m in records)
    assert compare.read(directory/'run.json')['scope']=='images'
