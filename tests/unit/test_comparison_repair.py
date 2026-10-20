import importlib.util
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location('repair', Path(__file__).parents[2]/'scripts/repair-comparison-video.py')
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


@pytest.mark.parametrize('kind', ['video', 'audio'])
def test_duplicate_packet_clock_moves_one_source_tick(kind):
    packet = SimpleNamespace(pts=20, dts=20, time_base=Fraction(1, 8000),
                             stream=SimpleNamespace(type=kind, index=0))
    changes = []
    repair.repair_clock(packet, {0: Fraction(20, 8000)}, changes)
    assert (packet.pts, packet.dts) == (21, 21)
    assert changes == [dict(stream=kind, old=20, new=21, time_base='1/8000')]


@pytest.mark.parametrize('pts,dts', [(19, 19), (21, 20)])
def test_other_clock_damage_is_rejected(pts, dts):
    packet = SimpleNamespace(pts=pts, dts=dts, time_base=Fraction(1, 8000),
                             stream=SimpleNamespace(type='video', index=0))
    with pytest.raises(ValueError, match='duplicate'):
        repair.repair_clock(packet, {0: Fraction(20, 8000)}, [])


def test_clock_comparison_uses_time_base():
    packet = SimpleNamespace(pts=40, dts=40, time_base=Fraction(1, 16000),
                             stream=SimpleNamespace(type='audio', index=0))
    changes = []
    repair.repair_clock(packet, {0: Fraction(20, 8000)}, changes)
    assert packet.dts == 41
