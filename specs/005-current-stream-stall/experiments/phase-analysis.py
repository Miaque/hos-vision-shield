from collections import Counter,defaultdict
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('analysis',ROOT/'analyze.py')
analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)


def phase(folder,expected):
    rows=analysis.read(folder/'live.jsonl')
    runtime={x['attempt_id']:x['data'] for x in rows if x['kind']=='runtime' and x['data'].get('clock_identity')}
    if expected=='100000':
        runtime={k:v for k,v in runtime.items() if v.get('output_mux_options',{}).get('max_interleave_delta')=='100000'}
    if expected=='restored':
        runtime={k:v for k,v in runtime.items() if v.get('output_mux_options')=={'rtsp_transport':'tcp'}}
    frames=list({(x['attempt_id'],x['sequence']):x for x in rows if x['kind']=='frame' and x['attempt_id'] in runtime}.values())
    operations=list({(x['attempt_id'],x['sequence']):x for x in rows if x['kind']=='operation' and x['attempt_id'] in runtime}.values())
    receipts=list({(x['observer_id'],x['sequence']):x for x in rows if x['kind']=='internal_frame'}.values())
    states=[x['data'] for x in rows if x['kind']=='internal_state']
    if not frames or not receipts:return {'status':'waiting_for_current_runtime'}
    clocks=[x['clock'] for x in states if x.get('clock',{}).get('boot_id')]
    assert clocks and all(v['clock_identity']==clocks[-1] for v in runtime.values()),'unverified cross-process clock'
    by_hash=defaultdict(list)
    for f in frames:by_hash[f.get('encoded_vcl_sha256')].append(f)
    def group(r):return r['observer_id'],r['connection']
    def offset(r,f):return (r['rtp_timestamp']-round(Fraction(f.get('timeline_pts') or f['source_pts'])*90000))%2**32
    votes=defaultdict(Counter)
    for r in receipts:
        for f in by_hash.get(r.get('encoded_vcl_sha256'),[]):
            votes[group(r)][(f['attempt_id'],offset(r,f))]+=1
    winner={g:v.most_common(1)[0][0] for g,v in votes.items()}
    matched=[]
    for r in receipts:
        candidates=[f for f in by_hash.get(r.get('encoded_vcl_sha256'),[]) if (f['attempt_id'],offset(r,f))==winner.get(group(r))]
        if len(candidates)==1:matched.append((r,candidates[0]))
    external=[x for x in analysis.read(folder/'rtp-output.jsonl') if x['kind']=='frame']
    raw=analysis.read(folder/'rtp-raw.jsonl')
    ext_lookup={(x['rtp_timestamp'],x['vcl_sha256']):x for x in external}
    packet_gaps,latency,events=[],[],[]
    for r,f in matched:latency.append(r['received_monotonic']-f['encoded_monotonic'])
    for (a,x),(b,y) in zip(matched,matched[1:]):
        if group(a)!=group(b) or x['attempt_id']!=y['attempt_id'] or y['sequence']!=x['sequence']+1:continue
        gap=b['received_monotonic']-a['received_monotonic'];packet_gaps.append(gap)
        if gap<=.1:continue
        writer=y['encoded_monotonic']-x['encoded_monotonic']
        exta=ext_lookup.get((a['rtp_timestamp'],a['encoded_vcl_sha256']))
        extb=ext_lookup.get((b['rtp_timestamp'],b['encoded_vcl_sha256']))
        remote_gap=(extb['last_packet_perf']-exta['last_packet_perf'] if exta and extb and exta['connection']==extb['connection'] else None)
        event=dict(event_id=len(events)+1,attempt_id=y['attempt_id'],previous_sequence=x['sequence'],sequence=y['sequence'],
            previous_source_pts=x['source_pts'],source_pts=y['source_pts'],
            media_delta_seconds=float(Fraction(y['source_pts'])-Fraction(x['source_pts'])),
            app_read_interval_seconds=y['received_monotonic']-x['received_monotonic'],
            processing_start_interval_seconds=y['processing_monotonic']-x['processing_monotonic'],
            mux_completion_interval_seconds=writer,internal_rtp_interval_seconds=gap,
            remote_rtp_interval_seconds=remote_gap,mux_to_internal_seconds=b['received_monotonic']-y['encoded_monotonic'],
            received_monotonic=y['received_monotonic'],decoded_monotonic=y['decoded_monotonic'],
            processing_monotonic=y['processing_monotonic'],mux_monotonic=y['encoded_monotonic'],
            internal_monotonic=b['received_monotonic'],rtp_timestamp=b['rtp_timestamp'],frame_hash=y['encoded_vcl_sha256'],
            remote_end_unix=extb['local_unix'] if extb else None,
            post_mux_expansion=gap-writer,
            classification='post_mux_buffering' if gap>.5 and writer<.2 else 'mixed_or_source_or_processing')
        events.append(event)
    paired={}
    for op in operations:
        if op['operation_kind'] in ('process','render') and op.get('frame_sequence'):
            paired.setdefault((op['attempt_id'],op['frame_sequence']),{})[op['operation_kind']]=op['seconds']
    combined=[sum(v.values()) for v in paired.values() if set(v)=={'process','render'}]
    scopes=list({(x['attempt_id'],x['thread_id'],x['stage'],x['ended_monotonic']):x for x in rows if x['kind']=='scope' and x['attempt_id'] in runtime}.values())
    result=dict(status='sampled',runtime=runtime,clock_verified=clocks[-1],server=analysis.server(frames),
        operations={k:analysis.summary([x['seconds'] for x in operations if x['operation_kind']==k]) for k in sorted({x['operation_kind'] for x in operations})},
        paired_frame_compute=analysis.summary(combined),internal_rtp_intervals=analysis.summary(packet_gaps),
        mux_return_to_internal_receipt=analysis.summary(latency),matched_internal_frames=len(matched),
        raw_rtp=analysis.rtp_summary(raw),remote_rtp=analysis.rtp_summary(analysis.read(folder/'rtp-output.jsonl')),
        post_mux_severe_count=sum(x['classification']=='post_mux_buffering' for x in events),
        events=events,slow_scopes=scopes,
        source_attempt_ends=list({(x['run_id'],x.get('ended_at_unix')):x for r in rows if r['kind']=='attempts'
            for x in r['data'].get('attempts',[]) if x.get('attempt_id') in runtime and x.get('ended_at_unix')}.values()))
    (folder/'phase-analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(folder.name,'matched',len(matched),'gap',result['internal_rtp_intervals'],'mux_receipt',result['mux_return_to_internal_receipt'],'post_mux_severe',result['post_mux_severe_count'])
    return result


if __name__=='__main__':
    phase(ROOT/sys.argv[1],sys.argv[2] if len(sys.argv)>2 else 'baseline')
