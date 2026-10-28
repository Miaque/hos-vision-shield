"""Frozen P0 replay: change only corner ROI in an in-memory function copy."""
import argparse
from collections import Counter
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
import textwrap
import time
from types import SimpleNamespace

import cv2
import numpy as np


def read(path):
    return json.loads(path.read_text('utf-8'))


def rows(path):
    return [json.loads(line) for line in path.read_text('utf-8').splitlines()]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


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
    cv2.setRNGSeed(0)
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


def self_check(redactor, original, whole):
    owner = redactor(SimpleNamespace(), detection_interval=3, tracking='flow')
    previous = cv2.GaussianBlur(np.random.default_rng(91).integers(20, 220, (160, 220), np.uint8), (3, 3), 0)
    current = cv2.warpAffine(previous, np.float32([[1, 0, 4], [0, 1, 2]]), (220, 160))
    for function in (original, whole):
        result = observe(function, owner, current, (60, 50, 120, 100), previous)
        np.testing.assert_allclose(result['box'], (64, 52, 124, 102), atol=.5, rtol=0)
        assert result['valid'] >= 6 and result['inlier_ratio'] >= .7
        assert observe(function, owner, current, (60, 50, 120, 100), np.zeros_like(previous))['reason'] == 'corners_lt6'
        assert observe(function, owner, np.zeros_like(current), (60, 50, 120, 100), previous)['box'] is None
    sparse = np.zeros_like(previous)
    sparse[65:70, 80:85] = 255
    assert observe(whole, owner, sparse, (60, 50, 120, 100), sparse)['corners'] < 6
    assert observe(whole, owner, sparse, (60, 50, 120, 100), sparse)['box'] is None
    owner.detector = SimpleNamespace(has_unconfirmed_plates=True, unconfirmed_plate_boxes=((63, 53, 82, 64),))
    owner._last_joint_detection = 0
    assert owner._plate_template_blocked((60, 50, 120, 100), .04, {})
    owner.detector.unconfirmed_plate_boxes = ((180, 130, 200, 145),)
    assert owner._plate_template_blocked((0, 0, 20, 10), .04, {})
    assert not owner.history and owner.gray is None


def review_tile(rgb_before, rgb_now, event):
    x, y, a, b = event['box']
    predictions = [event[k]['box'] for k in ('original', 'whole') if event[k]['box'] is not None]
    bounds = np.array([event['box'], *predictions])
    left = max(0, math.floor(bounds[:, 0].min()) - 30)
    top = max(0, math.floor(bounds[:, 1].min()) - 20)
    right = min(rgb_now.shape[1], math.ceil(bounds[:, 2].max()) + 30)
    bottom = min(rgb_now.shape[0], math.ceil(bounds[:, 3].max()) + 20)
    views = []
    for pixels, boxes in ((rgb_before, [(event['box'], (255, 220, 0))]),
                          (rgb_now, [(event['box'], (255, 220, 0)),
                                     (event['original']['box'], (30, 140, 255)),
                                     (event['whole']['box'], (30, 240, 60))])):
        crop = cv2.cvtColor(pixels[top:bottom, left:right].copy(), cv2.COLOR_RGB2BGR)
        for box, color in boxes:
            if box is not None:
                cv2.rectangle(crop, (round(box[0]-left), round(box[1]-top)),
                              (round(box[2]-left), round(box[3]-top)), color, 1)
        crop = cv2.resize(crop, (360, 140), interpolation=cv2.INTER_NEAREST)
        views.append(crop)
    tile = np.zeros((175, 720, 3), np.uint8)
    tile[35:] = np.hstack(views)
    label = f"{event['clip']} f{event['frame']} t{event['track']} {event['whole']['reason']} gate={event['blocked']} tpl={event['template'] is not None}"
    cv2.putText(tile, label, (4, 22), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
    return tile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bundle', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    root, output = args.bundle.resolve(), args.output.resolve()
    sys.path.insert(0, str(root/'p0/src'))
    from vision_shield.redaction import FrameRedactor, _Track
    original, whole = shadow_function(FrameRedactor)
    self_check(FrameRedactor, original, whole)
    if args.check_only:
        print('self-check passed', flush=True)
        return
    import av
    import torch
    from vision_shield.media.cuda_frames import hwaccel, rgb_pixels, gray_pixels
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    torch.cuda.set_device(0)
    output.mkdir(parents=True, exist_ok=False)
    (output/'review').mkdir()
    manifest = read(root/'bundle.json')
    for relative, expected in manifest['sha256'].items():
        assert digest(root/relative) == expected, relative
    runtime = dict(gpu=torch.cuda.get_device_name(), uuid=str(torch.cuda.get_device_properties(0).uuid),
                   torch=torch.__version__, av=av.__version__, opencv=cv2.__version__,
                   torch_threads=torch.get_num_threads(), opencv_threads=cv2.getNumThreads(),
                   script_sha256=digest(Path(__file__)), bundle=manifest,
                   scope='Frozen historical LOCAL P0 state replayed on current GPU. No new model calls, trajectory updates, masks, scheduling changes or live service access. Host timings include observation overhead.')
    (output/'manifest.json').write_text(json.dumps(runtime, indent=2), 'utf-8')
    summary = {}
    for clip in ('a', 'b'):
        history = rows(root/f'evidence/{clip}/tracks.jsonl')
        decisions = {r['frame']:r for r in rows(root/f'evidence/{clip}/decisions.jsonl')}
        failures = rows(root/f'evidence/{clip}/flow-failures.jsonl')
        known = read(root/f'analysis/{clip}-corner-shadow-events.json')
        lookup = {(r['frame'], tuple(r['box'])):r for r in known}
        failure_keys = {(r['frame'], tuple(r['box'])) for r in failures}
        samples = {r['frame']:r for r in read(root/'analysis/d0-sampling-candidates.json')[clip]['candidates']}
        events, frame_rows, tiles, unknown = [], [], [], 0
        controls, reproduced = Counter(), 0
        last_joint, previous_gray, previous_rgb = None, None, None
        owner = FrameRedactor(SimpleNamespace(), detection_interval=3, tracking='flow')
        with av.open(str(root/f'inputs/{clip}.mp4'), hwaccel=hwaccel(0)) as reader:
            for number, frame in enumerate(reader.decode(video=0), 1):
                moment = float(frame.pts * frame.time_base)
                assert frame.format.name == 'cuda' and frame.sw_format.name == 'nv12' and not frame.rotation
                current = history[number-1]
                assert number == current['frame'] and abs(moment-current['moment']) < 1e-9
                y, uv = [torch.from_dlpack(p.__dlpack__()) for p in frame.planes]
                rgb = rgb_pixels(y, uv, frame.colorspace, frame.color_range == 2)
                gray = gray_pixels(rgb).cpu().numpy()
                rgb_cpu = rgb.cpu().numpy()
                before = history[number-2] if number > 1 else None
                frame_events, active, known_count = [], [], 0
                owner._last_joint_detection = last_joint
                if before is not None and (number-1) % 3 != 0 and not decisions.get(number, {}).get('reset'):
                    active = [(index,t) for index,t in enumerate(before['tracks']) if t['kind'] == 'plate' and moment-t['seen'] <= .5]
                    owner.history.clear()
                    owner.history.extend(_Track(t['seen'], t['seen'], tuple(t['box']), t['kind']) for t in before['tracks'])
                    owner.detector = SimpleNamespace(has_unconfirmed_plates=bool(before['pending']),
                                                     unconfirmed_plate_boxes=before['pending'])
                    raw = {}
                    for index, track in active:
                        if track['seen'] != before['moment']:
                            unknown += 1
                            continue
                        known_count += 1
                        box = track['box']
                        initial = observe(original, owner, gray, box, previous_gray)
                        candidate = observe(whole, owner, gray, box, previous_gray)
                        cv2.setRNGSeed(0)
                        direct = original(owner, gray, box, previous_gray)
                        assert (direct is None) == (initial['box'] is None)
                        if direct is not None:
                            np.testing.assert_allclose(direct, initial['box'], atol=1e-8, rtol=0)
                        key = (number, tuple(box))
                        assert (key in failure_keys) == (initial['box'] is None), key
                        if key in lookup:
                            prior = lookup[key]
                            assert initial['corners'] == prior['corners'], (key, 'inset corners')
                            assert candidate['corners'] == prior['whole_box_corners'], (key, 'whole corners')
                        template = owner._follow(gray, box, previous_gray) if initial['box'] is None or candidate['box'] is None else None
                        event = dict(clip=clip, frame=number, moment=moment, track=index, box=box,
                                     original=initial, whole=candidate, template=template,
                                     sample=number in samples, pending=before['pending'],
                                     template_d0a_reason=lookup.get(key, {}).get('template_shadow'))
                        frame_events.append(event)
                        raw[index] = initial['box']
                    for event in frame_events:
                        event['blocked'] = owner._plate_template_blocked(event['box'], moment, raw)
                        candidate_raw = {e['track']:e['whole']['box'] for e in frame_events}
                        event['whole_gate_blocked'] = owner._plate_template_blocked(event['box'], moment, candidate_raw)
                        event['gate_reason'] = ('clear' if not event['blocked'] else
                            'snapshot_missing_or_expired' if last_joint is None or moment-last_joint > .2 else
                            'search_competition' if any(p[0] < event['box'][2]+96 and p[2] > event['box'][0]-96
                                and p[1] < event['box'][3]+96 and p[3] > event['box'][1]-96 for p in before['pending']) else
                            'pending_needs_global_confirmation')
                        event['gate_complete'] = known_count == len(active)
                        event['baseline_unresolved'] = event['original']['box'] is None and (event['blocked'] or event['template'] is None)
                        event['candidate_unresolved'] = event['whole']['box'] is None and (event['whole_gate_blocked'] or event['template'] is None)
                        # Keep original safety obligations even when expanded ROI LK returns a box.
                        event['safe_unresolved'] = (event['original']['box'] is None and (event['blocked'] or event['whole_gate_blocked'])) or event['candidate_unresolved']
                        events.append(event)
                    if active and known_count == len(active):
                        for metric, count in (
                            ('template_attempts', sum(e['original']['box'] is None and not e['blocked'] for e in frame_events)),
                            ('template_rejections', sum(e['original']['box'] is None and not e['blocked'] and e['template'] is None for e in frame_events)),
                            ('template_blocked', sum(e['original']['box'] is None and e['blocked'] for e in frame_events))):
                            assert current['stats'][metric]-before['stats'][metric] == count, (clip, number, metric)
                        reproduced += 1
                    if number in samples:
                        assert known_count == len(active) == samples[number]['active_tracks']
                        assert sum(e['original']['box'] is None for e in frame_events) == samples[number]['raw_failures']
                        assert sum(e['baseline_unresolved'] for e in frame_events) == samples[number]['failed_tracks']
                    if frame_events:
                        decision = decisions.get(number, {})
                        frame_rows.append(dict(frame=number, moment=moment, sample=number in samples,
                            decision=decision, active=len(active), known=known_count,
                            pending=before['pending'], baseline_unresolved=[e['track'] for e in frame_events if e['baseline_unresolved']],
                            roi_unresolved=[e['track'] for e in frame_events if e['candidate_unresolved']],
                            safe_unresolved=[e['track'] for e in frame_events if e['safe_unresolved']],
                            unknown_tracks=[i for i,t in active if t['seen'] != before['moment']],
                            current_confirmed_unassociated=[t['box'] for t in current['tracks'] if t['kind'] == 'plate' and t['seen'] == moment
                                and not any(owner._overlap(t['box'], old['box']) >= .2 for _,old in active)],
                            discovery='Required full-frame detail proposals and primary faces retained; newly reachable proposal/primary-plate confirmation and ROI costs unmeasured. No forward savings claimed.'))
                    for event in frame_events:
                        box = event['box']
                        tags = [name for name, condition in (
                            ('original_success', event['original']['box'] is not None),
                            ('medium_large', box[3]-box[1] >= 20),
                            ('border', min(box[0], box[1], 640-box[2], 480-box[3]) < 3),
                            ('weak_pending', bool(event['pending']) and event['blocked']),
                            ('reverse_reject', event['template_d0a_reason'] == 'reverse_displacement')) if condition and controls[name] < 3]
                        event['controls'] = tags
                        controls.update(tags)
                        regression = event['original']['box'] is not None and event['whole']['box'] is None
                        if event['sample'] or tags or regression or (event['whole']['box'] is not None and event['baseline_unresolved'] and not event['blocked']):
                            event['review_tile'] = len(tiles)
                            tiles.append(review_tile(previous_rgb, rgb_cpu, event))
                if number in decisions and not decisions[number]['face_only']:
                    last_joint = moment
                previous_gray, previous_rgb = gray, rgb_cpu
                if number % 500 == 0:
                    print(json.dumps(dict(clip=clip, decoded=number, events=len(events))), flush=True)
        assert number == len(history)
        assert len([e for e in events if (e['frame'], tuple(e['box'])) in lookup]) == len(known)
        assert sum(f['sample'] for f in frame_rows) == len(samples)
        for start in range(0, len(tiles), 8):
            sheet = np.vstack(tiles[start:start+8])
            assert cv2.imwrite(str(output/f'review/{clip}-{start//8:02d}.png'), sheet)
        (output/f'{clip}-events.json').write_text(json.dumps(events, indent=2), 'utf-8')
        (output/f'{clip}-frames.json').write_text(json.dumps(frame_rows, indent=2), 'utf-8')
        sample_events = [e for e in events if e['sample']]
        summary[clip] = dict(frames=number, events=len(events), unknown_reference_tracks=unknown,
            complete_frames_gate_counters_reproduced=reproduced, controls=dict(controls),
            original_reasons=dict(Counter(e['original']['reason'] for e in events)),
            whole_reasons=dict(Counter(e['whole']['reason'] for e in events)),
            sample_frames=len(samples), sample_events=len(sample_events),
            sample_whole_reasons=dict(Counter(e['whole']['reason'] for e in sample_events)),
            sample_raw_failed=len([e for e in sample_events if e['original']['box'] is None]),
            original_success_regressions=[dict(frame=e['frame'], track=e['track']) for e in events if e['original']['box'] is not None and e['whole']['box'] is None],
            newly_returned=len([e for e in events if e['original']['box'] is None and e['whole']['box'] is not None]),
            beyond_template_unblocked=[dict(frame=e['frame'], track=e['track'], complete=e['gate_complete']) for e in events if e['baseline_unresolved'] and not e['blocked'] and e['whole']['box'] is not None],
            sample_all_roi_resolved=[f['frame'] for f in frame_rows if f['sample'] and not f['roi_unresolved']],
            sample_all_safe_resolved=[f['frame'] for f in frame_rows if f['sample'] and not f['safe_unresolved']],
            frozen_opportunity_frames=[f['frame'] for f in frame_rows if not f['unknown_tracks'] and not f['safe_unresolved'] and f['decision'].get('recheck') and f['decision'].get('failed_tracks', 0) > 0 and not f['decision'].get('reset') and not f['decision'].get('periodic')],
            quality='Predictions are unapproved until identity and character coverage review; no model detections used as truth.')
        (output/'summary.json').write_text(json.dumps(summary, indent=2), 'utf-8')
        print(json.dumps({clip:summary[clip]}), flush=True)


if __name__ == '__main__':
    main()
