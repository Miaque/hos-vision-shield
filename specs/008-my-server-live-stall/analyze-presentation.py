"""Summarize actual compositor callbacks; missed callbacks remain an evidence limit."""
from collections import Counter
from bisect import bisect_right
import importlib.util
import json
from pathlib import Path
import sys


spec = importlib.util.spec_from_file_location('analysis', Path(__file__).with_name('analyze.py'))
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def summarize(evidence):
    start, end = evidence['startedUnix'], evidence['campaign']['deadline_unix']
    result = dict(start_unix=start, deadline_unix=end, stopped_unix=evidence.get('endedUnix'),
                  seconds=end-start, config=evidence['config'], streams={},
                  event_counts=dict(Counter(row['kind'] for row in evidence['events'])))
    for index in (0, 1):
        frames = [f for f in evidence['frames'] if f['stream'] == index and start <= f['unix'] <= end]
        events = [e for e in evidence['events'] if e.get('stream') == index and e['unix'] <= end]
        times = [f['presentationTime'] for f in frames]
        gaps = [b-a for a, b in zip(times, times[1:])]
        missed = sum(max(0, b['presentedFrames']-a['presentedFrames']-1)
                     for a, b in zip(frames, frames[1:]) if a['epoch'] == b['epoch'])
        windows = {}
        for seconds in (1, 5):
            boundaries = [start+i/10 for i in range(max(0, int((end-start-seconds)*10)))]
            arrivals = [f['unix'] for f in frames]
            windows[str(seconds)] = analysis.distribution(
                [(bisect_right(arrivals, t+seconds)-bisect_right(arrivals, t))/seconds for t in boundaries])
        waiting, intervals = None, []
        for event in events:
            if event['kind'] == 'waiting' and waiting is None:
                waiting = event['unix']
            elif event['kind'] == 'playing' and waiting is not None:
                intervals.append(event['unix']-waiting)
                waiting = None
        if waiting is not None:
            intervals.append(end-waiting)
        final = [row for row in evidence['streams'] if row['stream'] == index]
        result['streams'][str(index)] = dict(
            callback_frames=len(frames), callback_fps=len(frames)/(end-start),
            missed_presented_callbacks=missed, presentation_gaps=analysis.distribution(gaps),
            stalls={str(t): sum(g > t for g in gaps) for t in (.2, .5, 1)},
            callback_window_fps=windows, waiting_seconds=intervals,
            event_counts=dict(Counter(e['kind'] for e in events)),
            visibility=dict(Counter(f['visibility'] for f in frames)),
            final_player_status=final,
            media_time_non_increases=sum(b['mediaTime'] <= a['mediaTime'] for a, b in zip(frames, frames[1:])
                                         if a['epoch'] == b['epoch']))
    result['limit'] = ('Callbacks prove new frames submitted for composition, not panel scanout. '
                       'A gap containing missed callbacks is only an upper bound on a presentation stall. '
                       'Callback FPS excludes frames whose callbacks were missed. Player seeks and dropped '
                       'frames are reported separately and never count as application integrity acceptance.')
    return result


def self_check():
    frames = [dict(stream=0, epoch=epoch, unix=t, presentationTime=t, presentedFrames=count,
                   mediaTime=t, visibility='visible')
              for t, epoch, count in [(1, 1, 1), (2, 1, 4), (3, 2, 1), (4, 2, 2)]]
    result = summarize(dict(startedUnix=0, campaign=dict(deadline_unix=6), config={}, frames=frames,
                            streams=[], events=[dict(stream=0, kind='waiting', unix=1),
                                                dict(stream=0, kind='playing', unix=3)]))
    first, empty = result['streams']['0'], result['streams']['1']
    assert first['callback_frames'] == 4 and first['missed_presented_callbacks'] == 2
    assert first['waiting_seconds'] == [2] and first['media_time_non_increases'] == 0
    assert first['stalls']['0.5'] == 3 and first['stalls']['1'] == 0
    assert empty['callback_frames'] == 0 and empty['callback_fps'] == 0


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-check']:
        self_check()
        print('presentation self-check passed')
        raise SystemExit(0)
    source = Path(sys.argv[1])
    report = summarize(json.loads(source.read_text(encoding='utf-8')))
    assert report['seconds'] > 0
    assert all(value['missed_presented_callbacks'] >= 0 for value in report['streams'].values())
    source.with_name(source.stem+'-summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
