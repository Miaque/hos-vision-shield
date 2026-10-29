"""Correlate VCL hashes plus RTP/media-clock offsets; never subtract host clocks."""
from collections import Counter, defaultdict
from fractions import Fraction
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []


def summary(values):
    values = sorted(values)
    def q(fraction):
        if not values:
            return None
        index = (len(values) - 1) * fraction
        left = int(index)
        return values[left] + (values[min(left + 1, len(values) - 1)] - values[left]) * (index - left)
    return dict(n=len(values), p50_ms=None if not values else q(.5)*1000,
                p95_ms=None if not values else q(.95)*1000, p99_ms=None if not values else q(.99)*1000,
                max_ms=None if not values else max(values)*1000,
                over_ms={str(round(t*1000)): sum(x > t for x in values) for t in (.1,.2,.5,1)})


def server(frames):
    attempts = defaultdict(list)
    for frame in frames:
        attempts[frame['attempt_id']].append(frame)
    gaps, receive, decode_to_process, process_to_mux, media_gaps = [], [], [], [], []
    result = {}
    for aid, rows in attempts.items():
        rows.sort(key=lambda x: x['sequence'])
        sequence_gaps = 0
        for a,b in zip(rows, rows[1:]):
            if b['sequence'] != a['sequence'] + 1:
                sequence_gaps += 1
                continue
            gaps.append(b['encoded_monotonic'] - a['encoded_monotonic'])
            if a.get('received_monotonic') is not None and b.get('received_monotonic') is not None:
                receive.append(b['received_monotonic'] - a['received_monotonic'])
            media_gaps.append(float(Fraction(b.get('timeline_pts') or b['source_pts']) -
                                    Fraction(a.get('timeline_pts') or a['source_pts'])))
        for row in rows:
            if row.get('decoded_monotonic') is not None and row.get('processing_monotonic') is not None:
                decode_to_process.append(row['processing_monotonic'] - row['decoded_monotonic'])
                process_to_mux.append(row['encoded_monotonic'] - row['processing_monotonic'])
        elapsed = rows[-1]['encoded_monotonic'] - rows[0]['encoded_monotonic']
        media = float(Fraction(rows[-1].get('timeline_pts') or rows[-1]['source_pts']) -
                      Fraction(rows[0].get('timeline_pts') or rows[0]['source_pts']))
        result[aid] = dict(frames=len(rows), first_sequence=rows[0]['sequence'], last_sequence=rows[-1]['sequence'],
                           capture_sequence_gaps=sequence_gaps, wall_progress_seconds=elapsed,
                           media_progress_seconds=media, accumulated_lateness_increase_seconds=elapsed-media)
    return dict(attempts=result, mux_completion_intervals=summary(gaps), app_read_intervals=summary(receive),
                decoded_to_process=summary(decode_to_process), process_start_to_mux_completion=summary(process_to_mux),
                media_intervals=summary(media_gaps), timestamp_repaired_samples=sum(x['timestamp_repaired'] for x in frames),
                repeated_vcl_hashes=len(frames)-len({x.get('encoded_vcl_sha256') for x in frames}))


def rtp_summary(rows):
    frames = [x for x in rows if x['kind']=='frame']
    gaps, discontinuities = [], []
    for a,b in zip(frames, frames[1:]):
        if a['connection'] == b['connection']:
            gaps.append(b['last_packet_perf']-a['last_packet_perf'])
            media = ((b['rtp_timestamp']-a['rtp_timestamp']) % 2**32)/90000
            if media > .1:
                discontinuities.append(dict(local_unix=b['local_unix'], media_gap=media,
                                            receipt_gap=b['last_packet_perf']-a['last_packet_perf']))
    return dict(frames=len(frames), connections=len({x['connection'] for x in frames}),
                intervals=summary(gaps), errors=[x for x in rows if x['kind']=='error'],
                timestamp_discontinuities=discontinuities,
                max_rtp_sequence_gaps=max((x['sequence_gaps_total'] for x in frames),default=None))


def correlate(server_frames, output, raw):
    by_hash = defaultdict(list)
    for row in server_frames:
        if row.get('encoded_vcl_sha256'):
            by_hash[row['encoded_vcl_sha256']].append(row)
    output = [x for x in output if x['kind']=='frame']
    raw = [x for x in raw if x['kind']=='frame']
    votes = defaultdict(Counter)
    def offset(rtp, row):
        pts = Fraction(row.get('timeline_pts') or row['source_pts'])
        return (rtp['rtp_timestamp'] - round(pts * 90000)) % 2**32
    for rtp in output:
        for row in by_hash.get(rtp.get('vcl_sha256'), []):
            votes[rtp['connection']][(row['attempt_id'], offset(rtp,row))] += 1
    winners = {connection: counts.most_common(1)[0][0] for connection,counts in votes.items()}
    matched = []
    for rtp in output:
        winner = winners.get(rtp['connection'])
        candidates = [row for row in by_hash.get(rtp.get('vcl_sha256'), [])
                      if (row['attempt_id'],offset(rtp,row)) == winner]
        if len(candidates) == 1:
            matched.append((rtp,candidates[0]))
    events = []
    for (a,x),(b,y) in zip(matched,matched[1:]):
        if a['connection'] != b['connection'] or x['attempt_id'] != y['attempt_id'] or y['sequence'] != x['sequence']+1:
            continue
        receiver_gap=b['last_packet_perf']-a['last_packet_perf']
        if receiver_gap <= .1:
            continue
        server_gap=y['encoded_monotonic']-x['encoded_monotonic']
        present=[row for row in raw if a['last_packet_perf'] < row['last_packet_perf'] <= b['last_packet_perf']]
        events.append(dict(event_id=len(events)+1, local_end_unix=b['local_unix'],
            attempt_id=y['attempt_id'], previous_sequence=x['sequence'],sequence=y['sequence'],
            previous_source_pts=x['source_pts'],source_pts=y['source_pts'],
            previous_received_monotonic=x['received_monotonic'], received_monotonic=y['received_monotonic'],
            previous_processing_monotonic=x.get('processing_monotonic'),processing_monotonic=y.get('processing_monotonic'),
            previous_mux_monotonic=x['encoded_monotonic'],mux_monotonic=y['encoded_monotonic'],
            receiver_gap_seconds=receiver_gap, mux_gap_seconds=server_gap,
            interval_expansion_seconds=receiver_gap-server_gap,
            raw_frames_received_in_same_local_window=len(present),
            previous_output_rtp_timestamp=a['rtp_timestamp'],output_rtp_timestamp=b['rtp_timestamp'],
            frame_hash=y['encoded_vcl_sha256'],
            classification='accepted_by_mux_but_delivery_delayed' if server_gap <= .2 and receiver_gap > .5 else 'mixed_or_processing'))
    return dict(matched_frames=len(matched), received_frames=len(output),
                media_clock_calibration={str(k):dict(attempt_id=v[0],offset_90khz_ticks=v[1],votes=votes[k][v]) for k,v in winners.items()},
                severe_events_over_500ms=[x for x in events if x['receiver_gap_seconds']>.5], events=events)


def build():
    baseline=read(ROOT/'frames.jsonl')
    live=read(ROOT/'observed/live.jsonl')
    frames=[x for x in live if x['kind']=='frame']
    operations=list({(x['attempt_id'],x['sequence']):x for x in live if x['kind']=='operation'}.values())
    out=dict(baseline=server(baseline),observed=server(frames) if frames else {},
        operations={kind:summary([x['seconds'] for x in operations if x['operation_kind']==kind])
                    for kind in sorted({x['operation_kind'] for x in operations})})
    for folder in [ROOT, ROOT/'experiment']:
        name=folder.name if folder!=ROOT else 'baseline_rtp'
        raw=read(folder/'rtp-raw.jsonl');output=read(folder/'rtp-output.jsonl')
        out[name]=dict(raw=rtp_summary(raw),output=rtp_summary(output),
                      correlation=correlate(baseline+frames,output,raw))
    # The first RTP probe omitted keepalive; use only in-connection matched intervals, never its reconnect counts.
    oldraw=read(ROOT/'rtp-raw-initial-no-keepalive.jsonl')
    oldout=read(ROOT/'rtp-output-initial-no-keepalive.jsonl')
    out['initial_probe_matched_intervals_only']=correlate(baseline,oldout,oldraw)
    (ROOT/'analysis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print('baseline',out['baseline']['mux_completion_intervals'])
    print('operations',out['operations'])
    for key in ['baseline_rtp','experiment','initial_probe_matched_intervals_only']:
        c=out[key].get('correlation',out[key]);print(key,'matched',c['matched_frames'],'severe',len(c['severe_events_over_500ms']),
             'delivery_delayed',sum(x['classification']=='accepted_by_mux_but_delivery_delayed' for x in c['severe_events_over_500ms']))


if __name__=='__main__':
    assert summary([.1,.2,.5,1])['over_ms']=={'100':3,'200':2,'500':1,'1000':0}
    build()
