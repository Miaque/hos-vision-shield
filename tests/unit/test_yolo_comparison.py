import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('compare',Path(__file__).parents[2]/'scripts/compare-yolo.py')
compare=importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


def test_sample_selection_includes_last_and_vfr():
    assert compare.sample_indices([0,.2,5.1,5.2,11,11.2])==[0,2,4,5]
    assert compare.sample_indices([0])==[0]


def test_mask_union_and_partial():
    masks=[(0,0,4,10),(2,0,6,10)]
    assert compare.judge((10,10),masks,[[0,0],[9,0],[9,9],[0,9]])=='partial'
    assert compare.judge((10,10),[(0,0,10,10)],[[0,0],[9,0],[9,9],[0,9]])=='full'
    assert compare.judge((10,10),[],[[0,0],[9,0],[9,9],[0,9]])=='missed'


def test_no_formal_statistics_from_pending():
    assert compare.summarize([],[])['formal_samples']==0
    assert compare.summarize([],[])['variants']['baseline']['image']['face']['miss_rate'] is None


def test_changed_content_rejected(tmp_path):
    p=tmp_path/'中文.jpg';p.write_bytes(b'one')
    record={'id':'test','path':str(p),'sha256':compare.sha(p)}
    p.write_bytes(b'two')
    with pytest.raises(ValueError):compare.verify_source(record)


def annotation(status='human_confirmed'):
    return dict(id='00-000000', media_id='00', frame_index=0, kind='image', width=10, height=10,
                review_status=status, reviewer='tester', reviewed_at='2026-10-20T10:00:00+08:00',
                targets=[dict(id='face-1', kind='face', polygon=[[1,1],[3,1],[3,3],[1,3]])])


@pytest.mark.parametrize('change', [
    {'review_status': 'invented'}, {'reviewed_at': 'yesterday'},
    {'targets': [dict(id='f',kind='dog',polygon=[[1,1],[2,1],[2,2]])]},
    {'targets': [dict(id='f',kind='face',polygon=[[1,1],[2,1],[2,2]],uncertain='false')]},
])
def test_all_annotations_validated_even_when_pending(change):
    sample=annotation('pending');sample.update(change)
    with pytest.raises(ValueError):compare.summarize([sample],[])


def test_duplicate_samples_and_targets_are_rejected():
    sample=annotation('pending')
    with pytest.raises(ValueError):compare.summarize([sample,sample],[])
    sample['targets']*=2
    with pytest.raises(ValueError):compare.summarize([sample],[])


def test_quality_counts_union_uncertain_and_new_miss():
    sample=annotation()
    sample['targets'].append(dict(id='uncertain',kind='plate',uncertain=True,polygon=[[8,8],[9,8],[9,9]]))
    evidence=[dict(sample_id=sample['id'],variant='baseline',boxes=[(1,1,4,4),(0,6,2,8),(1,6,3,8),(8,8,10,10)]),
              dict(sample_id=sample['id'],variant='candidate',boxes=[])]
    result=compare.summarize([sample],evidence)
    assert result['variants']['baseline']['image']['face']['full']==1
    assert result['variants']['candidate']['image']['face']['missed']==1
    assert result['variants']['baseline']['image']['plate']['miss_rate'] is None
    assert result['false_mask_regions']['baseline']==1
    assert result['new_misses']==[dict(sample_id=sample['id'],target_id='face-1',kind='face',baseline='full',candidate='missed')]
    assert compare.summarize([annotation('disputed')],[])['disputed_samples']==1


def test_one_variant_cannot_mark_whole_run_processed(tmp_path):
    compare.write(tmp_path/'manifest.json',[dict(id='00')])
    compare.write(tmp_path/'baseline/00.json',dict(status='complete'))
    assert compare.result_status(tmp_path)=='incomplete'
    compare.write(tmp_path/'candidate/00.json',dict(status='complete'))
    assert compare.result_status(tmp_path)=='processed'
    compare.write(tmp_path/'candidate/00.json',dict(status='failed'))
    assert compare.result_status(tmp_path)=='incomplete'


def test_invalid_mask_box_is_rejected():
    with pytest.raises(ValueError):compare.mask_pixels((10,10),[(4,1,2,3)])
    with pytest.raises(ValueError):compare.target_pixels((10,10),[[1,1],[2,2],[3,3]])


def test_reviewed_input_without_outputs_is_explicitly_unavailable():
    summary=compare.summarize([annotation()],[])
    assert summary['unavailable_samples']==1 and summary['formal_samples']==0


def test_worker_rejects_wrong_imported_source(tmp_path):
    compare.write(tmp_path/'run.json',{})
    with pytest.raises(ValueError,match='Wrong imported source'):compare.worker(tmp_path,'baseline')


def test_resume_runs_only_unfinished_media_and_checks_frozen_tools(tmp_path,monkeypatch):
    from types import SimpleNamespace
    run=dict(manifest_sha256='',lock_sha256=compare.sha(compare.ROOT/'uv.lock'),
             candidate_source=compare.source_fingerprint(compare.ROOT),tool_source=compare.tool_fingerprint(),
             baseline_commit=compare.BASE,parameters=compare.PARAMETERS,models={},sample_sources=[])
    compare.write(tmp_path/'manifest.json',[dict(id='00')])
    run['manifest_sha256']=compare.sha(tmp_path/'manifest.json')
    compare.write(tmp_path/'run.json',run)
    compare.write(tmp_path/'annotations.json',[])
    (tmp_path/'baseline-source').mkdir()
    compare.write(tmp_path/'baseline-source.json',{})
    compare.write(tmp_path/'baseline/00.json',dict(status='complete'))
    calls=[]
    def execute(args,**kwargs):
        calls.append(args)
        compare.write(tmp_path/'candidate/00.json',dict(status='complete'))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(compare.subprocess,'run',execute)
    compare.run_variant(tmp_path,'candidate')
    assert compare.read(tmp_path/'run.json')['status']=='processed'
    assert calls[0][-1]=='candidate' and '_worker' in calls[0]
    run['tool_source']={}
    compare.write(tmp_path/'run.json',run)
    with pytest.raises(ValueError,match='tools'):compare.run_variant(tmp_path,'candidate')


def test_report_escapes_names_and_lists_failures(tmp_path):
    from PIL import Image
    source=tmp_path/'input.jpg';Image.new('RGB',(10,10),'black').save(source)
    media=dict(id='00',name='<中文 & 图片>.jpg',kind='image',path=str(source),sha256=compare.sha(source))
    sample=annotation('pending')
    compare.write(tmp_path/'manifest.json',[media])
    compare.write(tmp_path/'annotations.json',[sample])
    compare.write(tmp_path/'run.json',dict(manifest_sha256=compare.sha(tmp_path/'manifest.json'),sample_sources=compare.sample_sources([sample])))
    compare.write(tmp_path/'baseline/00.json',dict(media_id='00',variant='baseline',status='failed',error='<bad>'))
    compare.report(tmp_path)
    page=(tmp_path/'index.html').read_text('utf-8')
    assert '&lt;中文 &amp; 图片&gt;.jpg' in page
    assert '&lt;bad&gt;' in page and '待人工复核' in page and '未运行' in page
    assert '<img src="originals/00.jpg"' in page and 'file:///' not in page
    assert compare.sha(tmp_path/'originals/00.jpg')==media['sha256']
    assert compare.read(tmp_path/'summary.json')['status']=='incomplete'
