"""Host-diagnostic observation of original tracking returns and frame decisions."""

import ast
from collections import Counter
from functools import wraps
import inspect
import sys
import textwrap
from threading import Lock
from unittest.mock import patch


class FlowProfile:
    def __init__(self):
        self.active = None
        self.lock = Lock()
        self.counts, self.raw_reasons, self.full_reasons = Counter(), Counter(), Counter()
        self.reason_frames, self.full_models = Counter(), Counter()
        self.samples, self.sample_counts = [], Counter()

    def install(self, stack):
        from vision_shield.redaction import FrameRedactor

        original = FrameRedactor.regions
        self.regions_code = inspect.unwrap(original).__code__
        flow = inspect.unwrap(FrameRedactor._flow)
        self.flow_code = flow.__code__
        source, first = inspect.getsourcelines(flow)
        returns = sorted(node.lineno for node in ast.walk(ast.parse(textwrap.dedent(''.join(source))))
                         if isinstance(node, ast.Return) and isinstance(node.value, ast.Constant) and node.value.value is None)
        # shortcut: labels follow the current return sites; update and test them when flow gates change.
        reasons = ('missing_reference', 'patch_too_small', 'not_enough_features', 'forward_lk_missing',
                   'backward_lk_missing', 'inconsistent_points', 'insufficient_spread', 'affine_rejected', 'scale_rotation_rejected')
        if len(returns) != len(reasons):
            raise RuntimeError('Tracking failure observer does not match the current flow function')
        self.return_reasons = dict(zip((first + line - 1 for line in returns), reasons, strict=True))

        @wraps(original)
        def observed(owner, pixels, moment, **kwargs):
            if getattr(owner, 'tracking', None) != 'flow':
                return original(owner, pixels, moment, **kwargs)
            record = dict(owner=owner, frame=owner.frame_index + 1, moment=moment,
                          scheduled=owner.frame_index % owner.detection_interval == 0,
                          tracks=list(owner.history), flows={}, models=Counter())
            self.active = record
            previous = sys.gettrace()
            try:
                sys.settrace(self._trace)
                result = original(owner, pixels, moment, **kwargs)
            finally:
                sys.settrace(previous)
                self.active = None
            self._finish(record)
            return result

        stack.enter_context(patch.object(FrameRedactor, 'regions', observed))

    def _trace(self, frame, event, value):
        if frame.f_code not in (self.flow_code, self.regions_code):
            return None
        frame.f_trace_lines = False
        if event == 'return':
            local = frame.f_locals
            if frame.f_code is self.regions_code:
                self.active['state'] = {key: local[key] for key in ('flow_boxes', 'reset', 'detect', 'face_only')}
            else:
                caller = frame.f_back
                while caller is not None and caller.f_code is not self.regions_code:
                    caller = caller.f_back
                if caller is None:
                    raise RuntimeError('Flow observation has no frame caller')
                points, valid = local.get('points'), local.get('valid')
                self.active['flows'][caller.f_locals['old']] = dict(
                    reason=self.return_reasons.get(frame.f_lineno, 'invalid_bounds') if value is None else None,
                    features=None if 'points' not in local else 0 if points is None else len(points),
                    valid_points=None if valid is None else int(valid.sum()),
                    valid_fraction=None if valid is None else float(valid.mean()),
                    patch_shape=list(local['patch'].shape) if 'patch' in local else None,
                    fallback_enabled=bool(self.active['owner']._plate_template_fallback),
                    unconfirmed=bool(getattr(self.active['owner'].detector, 'has_unconfirmed_plates', False)))
        return self._trace

    def forward(self, stage):
        if self.active is not None:
            with self.lock:
                self.active['models'][stage] += 1

    def _finish(self, record):
        state = record['state']
        boxes, flows = state['flow_boxes'], record['flows']
        failed = {old for old, box in boxes.items() if box is None}
        extra = bool(state['detect'] and not state['face_only'] and not record['scheduled'])
        cause = ('scene_reset' if state['reset'] else 'track_failure' if failed else 'new_proposal') if extra else None
        self.counts['frames'] += 1
        self.counts['scheduled_full_frames'] += int(record['scheduled'])
        self.counts['raw_flow_calls'] += len(flows)
        if cause:
            self.full_reasons[cause] += 1
        if cause == 'track_failure':
            self.full_models.update(record['models'])
            if len(failed) == 1 and record['tracks'][next(iter(failed))].kind == 'plate':
                self.counts['single_failed_plate_frames'] += 1
                self.counts['single_failed_plate_with_healthy_sibling_frames'] += int(any(box is not None for box in boxes.values()))
            self.reason_frames.update({flows[old]['reason'] or 'stale_template_rejected' for old in failed})
        for old, detail in flows.items():
            reason = detail['reason']
            if reason:
                self.raw_reasons[reason] += 1
            elif old in failed:
                reason = 'stale_template_rejected'
                self.counts['stale_template_rejections'] += 1
            else:
                continue
            rescued = old not in failed
            self.counts['template_rescues' if rescued else 'unresolved_track_failures'] += 1
            track = record['tracks'][old]
            age = record['moment'] - track.seen
            if not rescued and detail['reason'] and track.kind == 'plate' and age <= .2 and detail['fallback_enabled'] and detail['unconfirmed']:
                self.counts['fallback_blocked_by_unconfirmed_plate'] += 1
            key = (reason, rescued)
            if self.sample_counts[key] < 2 and len(self.samples) < 64:
                self.samples.append(dict(frame=record['frame'], template_rescued=rescued,
                    scheduled=record['scheduled'], full_trigger=cause, kind=track.kind,
                    box=[float(value) for value in track.box], age_seconds=age,
                    healthy_siblings=sum(box is not None for index, box in boxes.items() if index != old), **(detail | {'reason': reason})))
                self.sample_counts[key] += 1

    def report(self):
        return dict(**self.counts, raw_failure_reasons=dict(self.raw_reasons), additional_full_frames=dict(self.full_reasons),
                    failure_reason_full_frames=dict(self.reason_frames), track_failure_full_model_calls=dict(self.full_models),
                    samples=self.samples,
                    notes=['Raw failures and template rescues count targets; full triggers count unique frames.',
                           'Failure reason frame counts can overlap when multiple tracks fail.',
                           'Model calls include confirmation work inside failure-triggered frames; these are not counterfactual extra calls.',
                           'Reference-only gates do not establish real motion failure or safe removability of detection.'])
