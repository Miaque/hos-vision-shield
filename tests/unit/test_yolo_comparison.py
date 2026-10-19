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
