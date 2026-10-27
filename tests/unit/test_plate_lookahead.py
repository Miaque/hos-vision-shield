from types import SimpleNamespace

import numpy as np

from vision_shield.media.plate_lookahead import PendingMask, PlateLookahead
from vision_shield.redaction import FrameRedactor


def frame(index, moment, plates=(), confirmed=()):
    return PendingMask(moment, np.full((8, 8), index, np.uint8), list(plates), list(confirmed),
                       lambda extra: (index, list(extra)))


def redactor(match=lambda gray, box, reference: box):
    return SimpleNamespace(_plate_template_fallback=False, _follow=match,
                           _overlap=FrameRedactor._overlap, debug=True, detection_stats={})


def test_holds_half_second_and_flushes_all_original_frames_once_in_order():
    delay = PlateLookahead(redactor())
    outputs = []
    for index in range(20):
        outputs.extend(delay.push(frame(index, index * .04)))
    assert len(outputs) == 7
    outputs.extend(delay.flush())
    assert [index for index, _ in outputs] == list(range(20))
    assert not delay.flush()


def test_frame_cap_bounds_bursts_without_discarding_or_repeating_frames():
    delay = PlateLookahead(redactor())
    outputs = []
    for index in range(100):
        outputs.extend(delay.push(frame(index, index * .001)))
        assert len(delay.pending) <= 16
    outputs.extend(delay.flush())
    assert [index for index, _ in outputs] == list(range(100))


def test_future_confirmed_plate_repairs_prior_frames_without_changing_tracking():
    owner = redactor(lambda gray, box, reference: tuple(v - 1 if i % 2 == 0 else v for i, v in enumerate(box)))
    delay = PlateLookahead(owner)
    for index in range(8):
        assert not delay.push(frame(index, index * .04))
    box = (20, 2, 40, 6)
    delay.push(frame(8, .32, [box], [box]))
    outputs = delay.flush()
    assert outputs[0] == (0, [(12, 2, 32, 6)])
    assert outputs[-1] == (8, [])
    assert owner.detection_stats['backfilled_plate_masks'] == 8


def test_failed_reverse_match_does_not_invent_an_earlier_mask():
    delay = PlateLookahead(redactor(lambda *args: None))
    delay.push(frame(0, 0))
    box = (2, 2, 6, 4)
    delay.push(frame(1, .04, [box], [box]))
    assert delay.flush() == [(0, []), (1, [])]


def test_existing_plate_renewal_does_not_repaint_the_entire_window():
    box = (2, 2, 6, 4)
    owner = redactor()
    delay = PlateLookahead(owner)
    for index in range(4):
        delay.push(frame(index, index * .04, [box], [box]))
    assert all(not extra for _, extra in delay.flush())
    assert owner.detection_stats['backfilled_plate_masks'] == 0


def test_scene_reset_flushes_old_frames_without_backtracking_across_the_cut():
    delay = PlateLookahead(redactor())
    delay.push(frame(0, 0))
    box = (2, 2, 6, 4)
    next_frame = frame(1, .04, [box], [box])
    next_frame.reset = True
    assert delay.push(next_frame) == [(0, [])]
    assert delay.flush() == [(1, [])]
