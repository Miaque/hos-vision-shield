"""Throwaway P2 thread/queue experiment. Not imported by deployed media code."""

from dataclasses import dataclass, field
from fractions import Fraction
import json
from pathlib import Path
from queue import Empty, Full, Queue
import sys
from threading import Event, Thread
from time import monotonic

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
from vision_shield.media.stream_pacing import StreamPacer


@dataclass(frozen=True)
class FinalFrame:
    """Caller may construct this only from a completed lookahead render result."""
    sequence: int
    pts: Fraction
    masked_pixels: bytes
    ready_at: float = field(default_factory=monotonic)


class FinalPublisher:
    def __init__(self, publish, *, capacity=4, budget=.5):
        self.publish, self.budget = publish, budget
        self.queue, self.stop = Queue(maxsize=capacity), Event()
        self.error = None
        self.stats = dict(normal_frames=0, protected_frames=0, dropped_frames=0, blocked_frames=0, peak_depth=0)
        self.worker = Thread(target=self.run, daemon=True)
        self.worker.start()

    def submit(self, frame):
        if not isinstance(frame, FinalFrame):
            raise TypeError('Pending/raw frames cannot enter the final-frame queue')
        if self.error:
            raise self.error
        if not self.worker.is_alive() or self.stop.is_set():
            raise RuntimeError('Publisher is closed')
        try:
            self.queue.put(frame, timeout=self.budget)
        except Full as error:
            self.stats['blocked_frames'] += 1
            raise TimeoutError('Final-frame queue is full; caller retains the frame') from error
        self.stats['peak_depth'] = max(self.stats['peak_depth'], self.queue.qsize())

    def run(self):
        pacer = StreamPacer(self.stop)
        try:
            while not self.stop.is_set():
                try:
                    frame = self.queue.get(timeout=.05)
                except Empty:
                    continue
                if frame is None:
                    return
                if monotonic() - frame.ready_at > self.budget:
                    raise TimeoutError('Final-frame age budget exceeded; output blocked')
                if not pacer.wait(frame.pts):
                    return
                if monotonic() - frame.ready_at > self.budget:
                    raise TimeoutError('Publishing wait exceeded frame age budget')
                self.publish(frame)
                self.stats['normal_frames'] += 1
        except Exception as error:
            self.stats['blocked_frames'] += 1
            self.error = error
            self.stop.set()

    def close(self):
        if self.error is None:
            try:
                self.queue.put(None, timeout=self.budget)
            except Full as error:
                self.stop.set()
                raise TimeoutError('Publisher cannot drain within close budget') from error
        self.worker.join(self.budget + .1)
        if self.worker.is_alive():
            self.stop.set()
            raise TimeoutError('Blocked publisher callback requires process reclamation')
        if self.error:
            raise self.error


def self_check():
    def frame(index, **kwargs):
        return FinalFrame(index, Fraction(index, 25), bytes([64]) * 12, **kwargs)

    outputs, progressed = [], Event()

    def publish(value):
        assert value.masked_pixels == bytes([64]) * 12
        outputs.append((value.sequence, value.pts, monotonic()))
        if len(outputs) == 3:
            progressed.set()

    publisher = FinalPublisher(publish)
    try:
        publisher.submit(object())
        raise AssertionError('Raw input was accepted')
    except TypeError:
        pass
    for index in range(4):
        publisher.submit(frame(index))
    # The producer is blocked here; already final frames continue on the publisher thread.
    assert progressed.wait(1)
    publisher.close()
    assert [row[0] for row in outputs] == list(range(4))
    assert [row[1] for row in outputs] == [Fraction(i, 25) for i in range(4)]
    assert publisher.stats['peak_depth'] <= 4 and publisher.stats['normal_frames'] == 4
    results = [dict(case='producer_blocked_final_frames_continue', **publisher.stats)]
    try:
        publisher.submit(frame(5))
        raise AssertionError('Closed publisher accepted another frame')
    except RuntimeError:
        pass

    written = []
    publisher = FinalPublisher(written.append)
    publisher.submit(frame(0, ready_at=monotonic() - 1))
    try:
        publisher.close()
        raise AssertionError('Expired final frame was published')
    except TimeoutError:
        pass
    assert not written and publisher.stats['blocked_frames'] == 1
    results.append(dict(case='expired_frame_blocks_output', **publisher.stats))

    def failed_publish(value):
        raise OSError('Injected publish failure')

    publisher = FinalPublisher(failed_publish)
    publisher.submit(frame(0))
    try:
        publisher.close()
        raise AssertionError('Publish failure was lost')
    except OSError:
        pass
    assert publisher.stats['normal_frames'] == 0 and publisher.stats['blocked_frames'] == 1
    results.append(dict(case='publish_failure_reaches_owner', **publisher.stats))

    entered, resume = Event(), Event()

    def blocked(value):
        entered.set()
        assert resume.wait(1)

    publisher = FinalPublisher(blocked, capacity=1, budget=.1)
    publisher.submit(frame(0))
    assert entered.wait(1)
    publisher.submit(frame(1))
    try:
        publisher.submit(frame(2))
        raise AssertionError('Capacity was exceeded')
    except TimeoutError:
        pass
    finally:
        resume.set()
    try:
        publisher.close()
    except TimeoutError:
        pass
    assert publisher.stats['peak_depth'] <= 1 and publisher.stats['dropped_frames'] == 0
    results.append(dict(case='capacity_backpressure_no_silent_drop', **publisher.stats))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    self_check()
