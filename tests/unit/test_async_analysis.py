"""Same-thread nested intervals must be subtracted before intersecting a future wait."""
import importlib.util
from pathlib import Path

import pytest


def test_nested_intervals_do_not_double_count_model_inside_wait():
    path = Path(__file__).resolve().parents[2]/'specs/007-model-async/experiments/analyze-tasks.py'
    spec = importlib.util.spec_from_file_location('async_analysis', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [dict(started=0,monotonic=10,sequence=4,stage='candidate'),
            dict(started=1,monotonic=8,sequence=3,stage='forward'),
            dict(started=2,monotonic=3,sequence=1,stage='copy'),
            dict(started=4,monotonic=5,sequence=2,stage='sync')]
    segments = list(module.exclusive_segments(rows))
    assert sum(b-a for a,b,_ in segments) == 10
    durations = {}
    for a,b,r in segments:
        durations[r['stage']] = durations.get(r['stage'], 0) + module.intersection(a,b,2.5,9)
    assert durations == pytest.approx(dict(candidate=1, forward=4, copy=.5, sync=1))
