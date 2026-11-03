from types import SimpleNamespace

import numpy as np
import pytest

from vision_shield.detection import Detection
from vision_shield.detection.quality import Detector


@pytest.mark.parametrize('fail', [False, True])
def test_delayed_plate_check_reuses_only_its_own_detail_and_restores_next_frame(fail):
    calls = []
    first, second = (np.full((100, 100, 3), n, np.uint8) for n in (0, 10))

    def candidates(pixels, *, kind=None, role):
        n = int(pixels[0, 0, 0])
        calls.append((role, n, kind))
        if fail and role == 'primary' and kind == 'plate':
            raise RuntimeError('primary failed')
        found = [Detection('face', (20, 20, 40, 50), .9),
                 Detection('plate', (50 + n, 60, 70 + n, 75), .9),
                 Detection('plate', (1, 1, 5, 5), .3)]
        return [d for d in found if kind is None or d.kind == kind]

    detector = Detector.__new__(Detector)
    detector.inference_device, detector.box_padding = 'cuda', 0
    detector.primary = SimpleNamespace(prepare=lambda p: p,
        candidates=lambda p, **kw: candidates(p, role='primary', **kw))
    detector.detail = SimpleNamespace(candidates=lambda p, **kw: candidates(p, role='detail', **kw))
    detector._confirmed = lambda *a, **kw: False
    detector.plate_proposals(first)
    detector.detect_faces(first)
    deferred = detector.plate_discovery_context(first)
    detector.plate_proposals(second)
    pending = detector._proposal_candidates
    try:
        if fail:
            with pytest.raises(RuntimeError, match='primary failed'):
                with deferred:
                    detector.detect_regions(first, kind='plate')
        else:
            with deferred:
                assert detector.detect_regions(first, kind='plate') == [Detection('plate', (50, 60, 70, 75), .9)] * 2
        assert detector._proposal_candidates is pending
        assert calls.count(('detail', 0, None)) == 1
        assert ('detail', 0, 'plate') not in calls
        assert calls.count(('primary', 0, 'plate')) == 1
        assert detector.detect_faces(second) == [Detection('face', (20, 20, 40, 50), .9)] * 2
    finally:
        detector._pool.shutdown()
