"""Export compact, auditable records without copying raw recordings into main."""
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone, timedelta
from fractions import Fraction
import hashlib
import html
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
OUT = Path('specs/005-current-stream-stall/evidence').resolve()
OUT.mkdir(parents=True, exist_ok=True)
PHASES = {'internal':'266f14d','interleave-b':'dd3474e','restored-a':'77f98a9'}


def read(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def write_csv(path, records):
    with path.open('w',encoding='utf-8-sig',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(dict.fromkeys(k for row in records for k in row)))
        writer.writeheader()
        writer.writerows(records)


def dump(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    summaries, events, timeline, reconnects = {}, [], [], []
    for name,revision in PHASES.items():
        folder = ROOT/name
        phase = json.loads((folder/'phase-analysis.json').read_text())
        supplement = json.loads((folder/'supplement-analysis.json').read_text())
        runtime = phase['runtime']
        verification = {aid:{module:hashlib.sha256(subprocess.check_output([
            'git','show',f'{revision}:src/vision_shield/'+module.replace('.','/')+'.py'])).hexdigest()==digest
            for module,digest in values['source_sha256'].items()} for aid,values in runtime.items()}
        assert all(all(matches.values()) for matches in verification.values())
        dump(OUT/f'{name}-runtime.json',runtime)
        rows = read(folder/'live.jsonl')
        frames = list({(x['attempt_id'],x['sequence']):x for x in rows if x['kind']=='frame' and x['attempt_id'] in runtime}.values())
        operations = {(x['attempt_id'],x.get('frame_sequence'),x['operation_kind']):x for x in rows
                      if x['kind']=='operation' and x['attempt_id'] in runtime and x.get('frame_sequence')}
        receipts = list({(x['observer_id'],x['sequence']):x for x in rows if x['kind']=='internal_frame'}.values())
        hashes = defaultdict(list)
        for frame in frames:
            hashes[frame['encoded_vcl_sha256']].append(frame)
        votes = defaultdict(Counter)
        def offset(receipt,frame):
            return (receipt['rtp_timestamp']-round(Fraction(frame['timeline_pts'])*90000))%2**32
        for receipt in receipts:
            for frame in hashes.get(receipt['encoded_vcl_sha256'],[]):
                votes[(receipt['observer_id'],receipt['connection'])][(frame['attempt_id'],offset(receipt,frame))]+=1
        winners = {k:v.most_common(1)[0][0] for k,v in votes.items()}
        matched = {}
        for receipt in receipts:
            candidates = [f for f in hashes.get(receipt['encoded_vcl_sha256'],[]) if
                          (f['attempt_id'],offset(receipt,f))==winners.get((receipt['observer_id'],receipt['connection']))]
            if len(candidates)==1:
                matched[(candidates[0]['attempt_id'],candidates[0]['sequence'])]=receipt
        for frame in frames:
            aid,sequence = frame['attempt_id'],frame['sequence']
            record = dict(phase=name,**{k:v for k,v in frame.items() if k not in ('kind','captured_unix','local_perf')})
            for kind in ('process','render'):
                op = operations.get((aid,sequence,kind))
                if op:
                    record.update({kind+'_'+key:op[key] for key in ('started_monotonic','ended_monotonic','seconds')})
            receipt = matched.get((aid,sequence))
            if receipt:
                record.update(internal_rtp_received_monotonic=receipt['received_monotonic'],
                              internal_rtp_first_packet_monotonic=receipt['first_packet_monotonic'],
                              output_rtp_timestamp=receipt['rtp_timestamp'])
            timeline.append(record)
        raw = [x for x in read(folder/'rtp-raw.jsonl') if x['kind']=='frame']
        external = [x for x in read(folder/'rtp-output.jsonl') if x['kind']=='frame']
        raw_gaps = [(b['last_packet_perf'],b['last_packet_perf']-a['last_packet_perf']) for a,b in zip(raw,raw[1:])
                    if a['connection']==b['connection']]
        external_lookup = {(x['rtp_timestamp'],x['vcl_sha256']):x for x in external}
        for event in phase['events']:
            event = dict(phase=name,**event)
            external_frame = external_lookup.get((event['rtp_timestamp'],event['frame_hash']))
            remote_gap = event['remote_rtp_interval_seconds']
            if external_frame and remote_gap is not None:
                end = external_frame['last_packet_perf']; start = end-remote_gap
                event['raw_frames_in_same_client_window'] = sum(start<x['last_packet_perf']<=end for x in raw)
                event['raw_max_interval_in_same_client_window_seconds'] = max((gap for at,gap in raw_gaps if start<at<=end),default=None)
            op = operations.get((event['attempt_id'],event['sequence'],'process'))
            if op:
                event['process_seconds'] = op['seconds']
            event.update(same_source_frame_wire_input_status='not_measured; app demux interval retained',
                         single_frame_compute_over_500ms=bool(op and op['seconds']>.5),
                         sustained_backlog='confirmed at attempt level; see wall/media progress and backpressure counters',
                         cpu_gpu_wait_cause='not isolated; host scopes are not GPU kernel timings',
                         client_only='no: server internal receiver independently observes stalls' if event['internal_rtp_interval_seconds']>.5 else 'not inferred',
                         pts_reconnect_scope='within same attempt and consecutive frame sequence')
            events.append(event)
        ordered = sorted(frames,key=lambda x:x['encoded_monotonic'])
        for a,b in zip(ordered,ordered[1:]):
            if a['attempt_id']!=b['attempt_id']:
                reconnects.append(dict(phase=name,previous_attempt=a['attempt_id'],next_attempt=b['attempt_id'],
                    last_mux_monotonic=a['encoded_monotonic'],first_mux_monotonic=b['encoded_monotonic'],
                    mux_progress_gap_seconds=b['encoded_monotonic']-a['encoded_monotonic']))
        gaps = phase['internal_rtp_intervals']
        summaries[name] = dict(revision=revision,source_fingerprints=verification,
            **{k:v for k,v in phase.items() if k not in ('runtime','events','slow_scopes')},supplement=supplement,
            severe_per_1000_intervals=1000*gaps['over_ms']['500']/gaps['n'],
            post_mux_severe_per_1000_intervals=1000*phase['post_mux_severe_count']/gaps['n'])
    dump(OUT/'summary.json',summaries)
    dump(OUT/'reconnections.json',reconnects)
    write_csv(OUT/'stall-events.csv',events)
    write_csv(OUT/'frame-timeline.csv',timeline)
    repro = json.loads((ROOT/'mux-repro-results.json').read_text())
    dump(OUT/'mux-repro-summary.json',[{k:v for k,v in item.items() if k not in ('muxed','received')} for item in repro])
    dump(OUT/'experiment-plan.json',json.loads((ROOT/'experiment-plan.json').read_text()))
    files = []
    for path in sorted(ROOT.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:
            with path.open('rb') as handle:
                digest = hashlib.file_digest(handle,'sha256').hexdigest()
            files.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=digest))
    dump(OUT/'raw-artifact-manifest.json',files)
    selected = sorted([e for e in events if e['phase']=='internal' and e.get('remote_rtp_interval_seconds')],
                      key=lambda e:e['internal_rtp_interval_seconds'],reverse=True)[:4]
    dump(OUT/'selected-events.json',selected)
    sections = []
    for event in selected:
        seq = event['sequence']; aid = event['attempt_id']
        local = [x for x in timeline if x['phase']=='internal' and x['attempt_id']==aid and seq-1<=x['sequence']<=seq+8]
        origin = local[0]['encoded_monotonic']
        end = max(x.get('internal_rtp_received_monotonic',x['encoded_monotonic']) for x in local)-origin+.1
        left,width=165,900
        def xpos(moment):return left+(moment-origin+.2)/(end+.2)*width
        svg = [f'<svg viewBox="0 0 1120 260" role="img" aria-label="事件 {event["event_id"]} 的服务器同帧时间线">']
        lanes = [('received_monotonic','After demux'),('decoded_monotonic','Decoded'),('processing_monotonic','Process start'),
                 ('encoded_monotonic','Mux accepted'),('internal_rtp_received_monotonic','Internal RTP')]
        for index,(key,label) in enumerate(lanes):
            y=35+index*38
            svg.append(f'<text x="5" y="{y+5}">{label}</text><path d="M165 {y}H1080" stroke="#ddd"/>')
        for frame in local:
            color='#c42626' if frame['sequence']==seq else '#637e94'
            points=[(xpos(frame[key]),35+i*38) for i,(key,_) in enumerate(lanes) if key in frame]
            svg.append('<polyline fill="none" stroke="'+color+'" opacity=".65" points="'+' '.join(f'{x},{y}' for x,y in points)+'"/>')
            for x,y in points:svg.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{color}"/>')
            x,y=points[-1];svg.append(f'<text x="{x+4}" y="{y+12}" font-size="10">{frame["sequence"]}</text>')
        for value in range(int(end)+1):
            x=xpos(origin+value);svg.append(f'<text x="{x}" y="235">{value}s</text>')
        svg.append('</svg>')
        stamp=datetime.fromtimestamp(event['remote_end_unix'],timezone(timedelta(hours=8))).isoformat(timespec='milliseconds')
        sections.append(f'<section><h2>A event {event["event_id"]} / frame {seq} / {stamp}</h2><p>Source PTS: {html.escape(event["source_pts"])}; '
                        f'media interval {event["media_delta_seconds"]*1000:.1f}ms; mux interval {event["mux_completion_interval_seconds"]*1000:.1f}ms; '
                        f'internal RTP interval {event["internal_rtp_interval_seconds"]*1000:.1f}ms; local RTP interval {event["remote_rtp_interval_seconds"]*1000:.1f}ms.</p>'
                        +''.join(svg)+'<p>横轴：服务器单调时钟相对前一帧 mux 完成。红线表示当前帧；连线只关联同一源帧。客户端时间没有放进这个时轴。</p></section>')
    table=''.join(f'<tr><td>{n}</td><td>{d["internal_rtp_intervals"]["n"]}</td><td>{d["internal_rtp_intervals"]["p99_ms"]:.1f}</td><td>{d["severe_per_1000_intervals"]:.2f}</td></tr>' for n,d in summaries.items())
    document='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>实时流卡顿证据</title><style>body{font:16px system-ui;max-width:1180px;margin:32px auto;padding:0 20px;color:#203040}section{margin:32px 0;border-top:1px solid #ddd}svg{width:100%;height:auto}td,th{padding:10px;border:1px solid #ccc}table{border-collapse:collapse}p{line-height:1.65}</style><h1>3c66326a：A/B/A 与同帧时间线</h1><p>A=默认交织；B=仅 max_interleave_delta=100000µs；restored-a=已恢复默认。接收间隔不是渲染间隔。源帧用 attempt、sequence、媒体PTS和编码VCL哈希关联，服务器时钟已核对 boot ID 与命名空间偏移。</p><table><tr><th>Phase</th><th>有效相邻间隔数</th><th>内部 RTP P99 (ms)</th><th>&gt;500ms /1000 intervals</th></tr>'+table+'</table>'+''.join(sections)+'<p>完整记录见同目录 stall-events.csv、frame-timeline.csv、summary.json。独立原流接收仍有约3秒停顿；原流与任务之间未建立输入包级同帧映射，不能用相近墙钟推断直接因果。未采集实际播放器渲染。</p></html>'
    (OUT/'timeline.html').write_text(document,encoding='utf-8')
    print(json.dumps(dict(frames=len(timeline),events=len(events),reconnections=reconnects,selected=[(x['event_id'],x.get('raw_frames_in_same_client_window'),x.get('raw_max_interval_in_same_client_window_seconds')) for x in selected],rates={n:d['severe_per_1000_intervals'] for n,d in summaries.items()}),ensure_ascii=False))


if __name__=='__main__':
    main()
