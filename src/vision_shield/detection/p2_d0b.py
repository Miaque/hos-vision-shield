"""Experimental host-only P2-D0b observer; no candidate enters production state."""
import inspect
import json
import math
import sys
import textwrap
import time
from unittest.mock import patch

import cv2
import numpy as np

def shadow_function(redactor):
    original = inspect.unwrap(redactor._flow)
    source = textwrap.dedent(inspect.getsource(original))
    source = source[source.index('def _flow'):]
    before = 'inset_x, inset_y = (right - left) // 5, (bottom - top) // 5'
    assert source.count(before) == 1
    namespace = {}
    exec(compile(source.replace(before, 'inset_x, inset_y = 0, 0'), '<whole-plate-shadow>', 'exec'),
         original.__globals__, namespace)
    return original, namespace['_flow']

def observe(function, owner, gray, box, reference):
    captured = {}

    def returned(frame, event, value):
        if frame.f_code is function.__code__ and event == 'return':
            captured.update(frame.f_locals)

    assert sys.getprofile() is None
    started = time.perf_counter()
    sys.setprofile(returned)
    try:
        result = function(owner, gray, box, reference)
    finally:
        sys.setprofile(None)
    elapsed = (time.perf_counter() - started) * 1000
    points, valid, inliers = [captured.get(key) for key in ('points', 'valid', 'inliers')]
    count = 0 if points is None else len(points)
    transform = captured.get('transform')
    reason = ('accepted' if result is not None else
              'patch_small' if min(captured['patch'].shape) < 5 else
              'corners_lt6' if count < 6 else
              'forward_missing' if captured.get('following') is None else
              'backward_missing' if captured.get('backward') is None else
              'bidirectional_gate' if valid.sum() < 6 or valid.mean() < .65 else
              'spread_gate' if 'transform' not in captured else
              'ransac_gate' if transform is None or not np.isfinite(transform).all() or inliers.mean() < .7 else
              'motion_gate' if 'corners' not in captured else 'output_size')
    return dict(box=None if result is None else list(result), reason=reason, corners=count,
                valid=None if valid is None else int(valid.sum()),
                valid_ratio=None if valid is None else float(valid.mean()),
                inliers=None if inliers is None else int(inliers.sum()),
                inlier_ratio=None if inliers is None else float(inliers.mean()),
                spread=None if 'source' not in captured else np.ptp(captured['source'], axis=0).tolist(),
                transform=None if transform is None else transform.tolist(),
                scale=captured.get('scale'),
                angle=None if transform is None else math.atan2(transform[1, 0], transform[0, 0]),
                host_ms=elapsed)

TARGET_FRAMES = (2, 3, 192, 204, 308, 333, 530, 605, 845, 873, 879, 911, 1058, 1196, 1428)


def install(stack, collector):
    from vision_shield.redaction import FrameRedactor
    _, whole = shadow_function(FrameRedactor)
    detect = FrameRedactor._detections
    report = collector.report
    evidence, used, truncated = [], 0, False

    def detection(owner, *args, **kwargs):
        nonlocal used, truncated
        caller = sys._getframe(1)
        state = caller.f_locals
        raw = state.get('raw_flow_boxes', {}) if caller.f_code.co_name == 'regions' else {}
        selected = owner.frame_index in TARGET_FRAMES and raw and used < 48*1024
        row = None
        if selected:
            targets = []
            candidate_raw = {}
            for index in raw:
                track = owner.history[index]
                if track.kind != 'plate':
                    continue
                candidate = observe(whole, owner, state['gray'], track.box, track.reference)
                candidate.pop('host_ms')
                candidate_raw[index] = candidate['box']
                targets.append(dict(track=index, box=track.box, seen=track.seen,
                    raw=raw[index], final=state['flow_boxes'][index], whole=candidate,
                    original_blocked=owner._plate_template_blocked(track.box, owner.moment, raw)))
            for target in targets:
                target['whole_blocked'] = owner._plate_template_blocked(target['box'], owner.moment, candidate_raw)
            row = dict(frame=owner.frame_index, moment=owner.moment, targets=targets,
                pending_before=getattr(owner.detector, 'unconfirmed_plate_boxes', None),
                last_joint=owner._last_joint_detection, recheck=state['recheck'],
                face_only=state['face_only'], reset=bool(state['reset']))
        counts = collector.forward_counts.copy() if row is not None else None
        result = detect(owner, *args, **kwargs)
        if row is not None:
            row['actual_forwards'] = {k:collector.forward_counts[k]-counts[k] for k in collector.forward_counts}
            row['pending_after'] = getattr(owner.detector, 'unconfirmed_plate_boxes', None)
            row['accepted_plates'] = [d.box for d in result if d.kind == 'plate']
            # Bound the addition so the existing child-pipe report retains room.
            row = json.loads(json.dumps(row, default=float), parse_float=lambda x:round(float(x), 6))
            size = len(json.dumps(row).encode())
            if used + size <= 48*1024:
                evidence.append(row)
                used += size
            else:
                truncated = True
        elif owner.frame_index in TARGET_FRAMES and raw and used >= 48*1024:
            truncated = True
        return result

    def with_shadow(frames):
        result = report(frames)
        result['p2_d0b'] = dict(target_frames=TARGET_FRAMES, rows=evidence.copy(), truncated=truncated,
            byte_limit=48*1024, scope='Shadow only. P0 raw/final predictions, gates, discovery, lifetime and lookahead unchanged. Diagnostic FPS is not acceptance throughput. Actual forwards are original detection-pass calls, not savings.')
        while result['p2_d0b']['rows'] and len(json.dumps(result).encode()) > 200*1024:
            result['p2_d0b']['rows'].pop()
            result['p2_d0b']['truncated'] = True
        return result

    stack.enter_context(patch.object(FrameRedactor, '_detections', detection))
    stack.enter_context(patch.object(collector, 'report', with_shadow))
