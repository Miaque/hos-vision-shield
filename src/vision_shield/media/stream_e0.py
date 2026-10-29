"""Bounded worker-clock evidence; counters never depend on retaining a trace sample."""
from collections import Counter, OrderedDict, defaultdict, deque
from contextlib import contextmanager
from fractions import Fraction
import json
import hashlib
from threading import Lock, get_ident, local
import time


def runtime_fingerprint(runtime):
    identity = {key: runtime.get(key) for key in ('source_sha256', 'effective_config', 'output_mux_options')}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


class StreamE0:
    def __init__(self, *, clock=time.monotonic):
        self.clock, self.lock, self.context = clock, Lock(), local()
        self.main_thread, self.started, self.ended = get_ident(), clock(), None
        self.frames, self.encoder = OrderedDict(), {}
        self.pacer, self.audio, self.video, self.events = (deque(maxlen=128), deque(maxlen=128),
                                                        deque(maxlen=128), deque(maxlen=64))
        self.audio_outputs = deque(maxlen=128)
        self.frame_count = self.event_count = 0
        self.pacer_totals, self.fifo_totals, self.clock_totals = Counter(), Counter(), Counter()
        self.budgets, self.scope_counts = defaultdict(Counter), Counter()
        self.blocked, self.previous_clocks = {}, {}
        self.shadow_due = self.epoch_moment = self.epoch_actual = None
        self.first_output = self.last_output = None
        self.packet_clock, self.full_since = None, None
        self.main_stack = []
        self.runtime_fingerprint = None

    @staticmethod
    def media_clock(value):
        base = getattr(value, 'time_base', None)
        return dict(pts=getattr(value, 'pts', None), dts=getattr(value, 'dts', None),
                    time_base=None if base is None else str(base),
                    duration=getattr(value, 'duration', None))

    def _check_clock(self, key, value, field='pts'):
        if value.get(field) is None or value.get('time_base') is None:
            return
        moment = value[field] * Fraction(value['time_base'])
        previous = self.previous_clocks.get(key)
        if previous is not None and moment <= previous:
            self.clock_totals[key + '_nonincreasing'] += 1
        if previous is not None and moment > previous:
            delta = float(moment - previous)
            self.clock_totals[key + '_forward_jumps_over_500ms'] += delta > .5
            self.clock_totals[key + '_max_forward_step_seconds'] = max(
                self.clock_totals[key + '_max_forward_step_seconds'], delta)
        self.previous_clocks[key] = moment

    def frame_start(self, frame, received_at, *, decoded_at=None, packet_clock=None):
        raw = self.media_clock(frame)
        with self.lock:
            self.frame_count += 1
            fid = self.frame_count
            self.frames[fid] = dict(frame_id=fid, revision=1, raw_frame_clock=raw,
                received_monotonic=received_at, decoded_monotonic=decoded_at,
                decode_trigger_packet_clock=packet_clock)
            if len(self.frames) > 256:
                self.frames.popitem(last=False)
            self._check_clock('video_raw_frame_pts', raw)
        return fid

    def update_frame(self, fid, **data):
        with self.lock:
            if fid in self.frames:
                self.frames[fid].update(data)
                self.frames[fid]['revision'] += 1

    def video_clock(self, fid, kind, value):
        raw = self.media_clock(value)
        self.update_frame(fid, **{kind: raw})
        with self.lock:
            self._check_clock('video_' + kind + '_pts', raw)

    def audio_clock(self, before, after):
        with self.lock:
            self.clock_totals['audio_packets'] += 1
            self._check_clock('audio_raw_dts', before, 'dts')
            self._check_clock('audio_normalized_dts', after, 'dts')
            self._check_clock('audio_raw_pts', before)
            self._check_clock('audio_normalized_pts', after)
            self.audio.append(dict(sequence=self.clock_totals['audio_packets'],
                monotonic=self.clock(), raw=before, normalized=after))

    def video_packet(self, value):
        with self.lock:
            self.clock_totals['video_demux_packets'] += 1
            self._check_clock('video_demux_pts', value)
            self._check_clock('video_demux_dts', value, 'dts')
            self.video.append(dict(sequence=self.clock_totals['video_demux_packets'],
                                   monotonic=self.clock(), raw=value))

    def audio_publish(self, before_origin, output, origin, transcoded):
        with self.lock:
            self.clock_totals['audio_mux_attempts'] += 1
            self._check_clock('audio_mux_pts', output)
            self._check_clock('audio_mux_dts', output, 'dts')
            row = dict(sequence=self.clock_totals['audio_mux_attempts'],
                monotonic=self.clock(), normalized_input=before_origin, mux_clock=output,
                origin=str(origin), transcoded=transcoded)
            self.audio_outputs.append(row)
            return row

    def audio_accepted(self, row):
        with self.lock:
            self.clock_totals['audio_mux_packets'] += 1
            row['mux_completed_monotonic'] = self.clock()

    def fifo_full(self, fid, depth):
        now = self.clock()
        with self.lock:
            self.blocked[fid] = now
            self.fifo_totals['full_events'] += 1
        self.update_frame(fid, full_started_monotonic=now, full_depth=depth)

    def fifo_unblocked(self, fid):
        now = self.clock()
        with self.lock:
            started = self.blocked.pop(fid, None)
            seconds = 0 if started is None else now - started
            self.fifo_totals['backpressure_seconds'] += seconds
        self.update_frame(fid, backpressure_seconds=seconds, full_ended_monotonic=now)

    def fifo(self, fid, action, depth, capacity=2):
        now = self.clock()
        with self.lock:
            self.fifo_totals[action] += 1
            self.fifo_totals['capacity'] = capacity
            if action == 'dequeued' and fid in self.frames:
                self.fifo_totals['queue_wait_seconds'] += now - self.frames[fid]['enqueue_monotonic']
            if depth == capacity and self.full_since is None:
                self.full_since = now
            elif depth < capacity and self.full_since is not None:
                self.fifo_totals['full_occupancy_seconds'] += now - self.full_since
                self.full_since = None
        self.update_frame(fid, **{('enqueue' if action == 'enqueued' else 'dequeue') + '_monotonic': now,
                                 action + '_depth': depth})

    @contextmanager
    def span(self, name):
        stack = getattr(self.context, 'stack', None)
        if stack is None:
            stack = self.context.stack = self.main_stack if get_ident() == self.main_thread else []
        item = [self.clock(), 0., name]
        stack.append(item)
        try:
            yield
        finally:
            elapsed = self.clock() - item[0]
            stack.pop()
            if stack:
                stack[-1][1] += elapsed
            with self.lock:
                self.budgets[get_ident()][name] += max(0., elapsed - item[1])
                self.scope_counts[name] += 1

    def event(self, name, **data):
        with self.lock:
            self.event_count += 1
            self.events.append(dict(sequence=self.event_count, name=name, monotonic=self.clock(), **data))

    def encode_context(self, moment, fid, **data):
        with self.lock:
            # The existing encoder guard limits the corresponding media backlog to 512 frames.
            row = self.frames.get(fid, {})
            clocks = {key: row.get(key) for key in ('raw_frame_clock', 'input_clock',
                                                   'normalized_clock', 'timestamp_repaired')}
            self.encoder[moment] = dict(clocks, frame_id=fid, **data)

    def pacing(self, sample):
        sample = dict(sample)
        sample['rebase_reason'] = ('initial' if sample['previous_moment'] is None
            else 'late-beyond-half-interval' if sample['rebased'] else None)
        sample['selected_constraint'] = ('anti-burst' if sample['anti_burst_due'] > sample['media_due']
            else 'media-schedule' if sample['media_due'] > sample['anti_burst_due'] else 'tie')
        moment, interval = Fraction(sample['moment']), sample['interval']
        if self.shadow_due is None:
            self.shadow_due = sample['due']
            self.epoch_moment, self.epoch_actual = moment, sample['actual']
            anti, media, shadow_request = sample['due'], sample['due'], 0.
        else:
            media = self.shadow_due + interval
            anti = sample['previous_actual'] + interval / 2
            self.shadow_due = max(media, anti)
            shadow_request = max(0., self.shadow_due - sample['decision_at'])
        requested = max(0., sample['remaining'])
        uplift = max(0., requested - shadow_request)
        retained_anti = min(shadow_request, max(0., anti - sample['decision_at']))
        retained_media = max(0., requested - uplift - retained_anti)
        actual = sample['actual_wait_seconds']
        # Partition measured wait against the requested interval; timer overrun is separate.
        anti_actual = min(actual, retained_anti)
        media_actual = min(max(0., actual - anti_actual), retained_media)
        uplift_actual = min(max(0., actual - anti_actual - media_actual), uplift)
        lag = sample['decision_at'] - (self.epoch_actual + float(moment - self.epoch_moment))
        sample.update(conditional_no_rebase_due=self.shadow_due,
            conditional_rebase_request_seconds=uplift, retained_anti_burst_request_seconds=retained_anti,
            retained_media_request_seconds=retained_media, lag_to_initial_epoch_seconds=lag)
        with self.lock:
            totals = self.pacer_totals
            if actual > 0:
                stack = getattr(self.context, 'stack', [])
                if stack:
                    stack[-1][1] += actual
                self.budgets[get_ident()]['wait:pacer'] += actual
                self.scope_counts['wait:pacer'] += 1
            totals['decisions'] += 1
            totals['rebases'] += sample['rebased']
            totals['positive_waits'] += requested > 0
            totals['stopped_decisions'] += not sample['allowed']
            totals['late_epoch_positive_waits'] += lag > interval and requested > 0
            totals['conditional_rebase_waits'] += uplift > 1e-9
            totals['conditional_rebase_served_waits'] += uplift_actual > 1e-9
            totals[sample['selected_constraint'] + '_decisions'] += 1
            totals[sample['selected_constraint'] + '_actual_wait_seconds'] += actual
            totals['clamped_media_intervals'] += sample.get('raw_media_interval', 0) > .5
            for key, value in dict(requested_wait_seconds=requested, actual_wait_seconds=actual,
                conditional_rebase_request_seconds=uplift, conditional_rebase_wait_seconds=uplift_actual,
                retained_anti_burst_wait_seconds=anti_actual, retained_media_wait_seconds=media_actual,
                timer_overrun_seconds=max(0., actual - requested)).items():
                totals[key] += value
            sample.update(self.encoder.get(moment, {}))
            sample['encoder_packet_clock'] = self.packet_clock
            sample['sequence'] = totals['decisions']
            self.pacer.append(sample)

    def written(self, moment, at):
        with self.lock:
            self.clock_totals['video_mux_packets'] += 1
            context = self.encoder.pop(moment, {})
            if self.pacer and self.pacer[-1]['moment'] == str(moment) and self.pacer[-1]['allowed']:
                self.pacer[-1]['mux_completed_monotonic'] = at
            source = context.get('raw_frame_clock')
            if source and source['pts'] is not None and source['time_base'] is not None:
                normalized = context.get('normalized_clock')
                point = dict(monotonic=at, source_seconds=float(source['pts'] * Fraction(source['time_base'])),
                    normalized_seconds=(None if not normalized or normalized['pts'] is None else
                                        float(normalized['pts'] * Fraction(normalized['time_base']))),
                    encoder_seconds=float(moment))
                if self.first_output is None:
                    self.first_output = point
                self.last_output = point
        self.update_frame(context.get('frame_id'), mux_completed_monotonic=at)

    def finish(self):
        self.ended = self.clock()

    def summary(self):
        result = self.snapshot()
        for key in ('frames', 'pacer', 'audio_clocks', 'video_clocks', 'audio_outputs', 'events'):
            result.pop(key)
        return result

    def snapshot(self):
        now = self.clock()
        with self.lock:
            main = dict(self.budgets.get(self.main_thread, {}))
            cutoff = now if self.ended is None else self.ended
            for i, item in enumerate(self.main_stack):
                active_child = cutoff - self.main_stack[i+1][0] if i+1 < len(self.main_stack) else 0
                main[item[2]] = main.get(item[2], 0) + max(0., cutoff-item[0]-item[1]-active_child)
            wall = cutoff - self.started
            result = dict(schema_version=1, clock_origin='worker_pyav_demux_and_decode',
                runtime_fingerprint=self.runtime_fingerprint,
                started_monotonic=self.started, sampled_monotonic=now, ended_monotonic=self.ended,
                pacer_totals=dict(self.pacer_totals), fifo_totals=dict(self.fifo_totals),
                clock_totals=dict(self.clock_totals), scope_counts=dict(self.scope_counts),
                main_exclusive_seconds=main, main_wall_seconds=wall,
                main_unattributed_seconds=max(0., wall - sum(main.values())),
                other_thread_exclusive_seconds={str(tid): dict(values) for tid, values in self.budgets.items()
                                                if tid != self.main_thread},
                active_backpressure_seconds=sum(cutoff - start for start in self.blocked.values()),
                active_full_occupancy_seconds=0 if self.full_since is None else cutoff - self.full_since,
                publication_boundary='video_mux_return_not_rtp',
                frames=[dict(row) for row in list(self.frames.values())[-64:]], pacer=list(self.pacer),
                audio_clocks=list(self.audio), video_clocks=list(self.video), events=list(self.events),
                audio_outputs=list(self.audio_outputs),
                first_output=self.first_output, last_output=self.last_output,
                ring_discarded=dict(frames=self.frame_count-len(self.frames),
                    pacer=self.pacer_totals['decisions']-len(self.pacer),
                    audio_clocks=self.clock_totals['audio_packets']-len(self.audio),
                    video_clocks=self.clock_totals['video_demux_packets']-len(self.video),
                    audio_outputs=self.clock_totals['audio_mux_attempts']-len(self.audio_outputs),
                    events=self.event_count-len(self.events)),
                discarded={}, totals=dict(frames=self.frame_count, pacer=self.pacer_totals['decisions'],
                    audio_clocks=self.clock_totals['audio_packets'],
                    video_clocks=self.clock_totals['video_demux_packets'],
                    audio_outputs=self.clock_totals['audio_mux_attempts'], events=self.event_count))
        # Serialize each candidate record once; repeatedly serializing the whole tree blocks the media loop.
        shares = dict(frames=.30, pacer=.40, audio_clocks=.07, video_clocks=.07, audio_outputs=.09, events=.07)
        records = {key: result[key] for key in shares}
        result.update({key: [] for key in shares})
        available = 128 * 1024 - 1024 - len(json.dumps(result).encode())
        for key, share in shares.items():
            used, kept = 0, []
            for row in reversed(records[key]):
                size = len(json.dumps(row).encode()) + 2
                if used + size > available * share:
                    break
                kept.append(row)
                used += size
            result[key] = list(reversed(kept))
        result['discarded'] = {key: result['totals'][key] - len(result[key]) for key in result['totals']}
        result['wire_omitted'] = {key: result['discarded'][key] - result['ring_discarded'][key]
                                  for key in result['totals']}
        return result
