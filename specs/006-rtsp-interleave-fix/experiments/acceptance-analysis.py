"""Summarize direct reception, application progress and separately observed playback."""
from collections import Counter,defaultdict
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('base',ROOT/'analyze.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
spec=importlib.util.spec_from_file_location('extra',ROOT/'supplement-analysis.py')
extra=importlib.util.module_from_spec(spec);spec.loader.exec_module(extra)


def windows(frames,size):
    stamps=sorted(x['last_packet_perf'] for x in frames)
    if not stamps:return {}
    start,end=stamps[0],stamps[-1]
    left,right=0,0
    rates=[]
    while start+size<=end:
        while left<len(stamps) and stamps[left]<start:left+=1
        right=max(right,left)
        while right<len(stamps) and stamps[right]<start+size:right+=1
        rates.append((right-left)/size)
        start+=.5
    return extra.numeric(rates)


def decoded(role):
    rows=base.read(ROOT/'candidate'/f'av-{role}.jsonl')
    result=dict(errors=[x for x in rows if x['kind']=='error'],tracks={})
    for track in ('video','audio'):
        values=[x for x in rows if x['kind']=='decoded' and x['track']==track]
        diffs=[float(Fraction(b['pts'])*Fraction(b['time_base'])-Fraction(a['pts'])*Fraction(a['time_base']))
               for a,b in zip(values,values[1:]) if a['connection']==b['connection'] and a['pts'] is not None and b['pts'] is not None]
        packets=[x for x in rows if x['kind']=='packet' and x['track']==track and x['bytes']]
        pts=[x for x in packets if x['pts'] is not None]
        dts=[x for x in packets if x['dts'] is not None]
        def bad(items,key):
            return sum(Fraction(b[key])*Fraction(b['time_base'])<=Fraction(a[key])*Fraction(a['time_base'])
                       for a,b in zip(items,items[1:]) if a['connection']==b['connection'])
        result['tracks'][track]=dict(decoded_frames=len(values),packets=len(packets),
            decoded_pts_nonpositive=sum(x<=0 for x in diffs),decoded_pts_interval_seconds=extra.numeric(diffs),
            pts_available=len(pts),dts_available=len(dts),packet_pts_nonpositive=bad(pts,'pts'),packet_dts_nonpositive=bad(dts,'dts'))
        if track=='audio':
            result['tracks'][track]['samples']=sum(x['samples'] for x in values)
            result['tracks'][track]['rms']=extra.numeric([x['rms'] for x in values])
    return result


def audio_wire(role):
    rows=base.read(ROOT/'candidate'/f'rtp-av-{role}.jsonl')
    audio=[x for x in rows if x['kind']=='audio_au']
    steps=[(((b['rtp_timestamp']-a['rtp_timestamp']+2**31)%2**32)-2**31)/b['rate']
           for a,b in zip(audio,audio[1:]) if a['connection']==b['connection']]
    return dict(audio_aus=len(audio),connections=len({x['connection'] for x in audio}),
                errors=[x for x in rows if x['kind']=='error'],rtcp_sender_reports=sum(x['kind']=='sender_report' for x in rows),
                rtp_sequence_gaps=max((x['sequence_gaps_total'] for x in audio),default=0),
                media_nonpositive_steps=sum(x<=0 for x in steps),media_step_seconds=extra.numeric(steps),
                media_step_counts=dict(Counter(str(round(x,6)) for x in steps)))


def build():
    output={}
    for name,commit in [('default-current-load','77f98a9'),('candidate','1c9be54')]:
        folder=ROOT/name
        phase=json.loads((folder/'phase-analysis.json').read_text())
        supplement=json.loads((folder/'supplement-analysis.json').read_text())
        rows=base.read(folder/'live.jsonl')
        runtime=phase['runtime']
        checks={aid:{key:hashlib.sha256(subprocess.check_output(['git','show',commit+':src/vision_shield/'+key.replace('.','/')+'.py'])).hexdigest()==value
                     for key,value in metadata['source_sha256'].items()} for aid,metadata in runtime.items()}
        assert checks and all(all(v.values()) for v in checks.values())
        loads=list({json.dumps(x['data'],sort_keys=True):x['data'] for x in rows if x['kind']=='load'}.values())
        roles={}
        for role in ('raw','output'):
            records=base.read(folder/f'rtp-{role}.jsonl')
            frames=[x for x in records if x['kind']=='frame']
            summary=base.rtp_summary(records)
            gaps=[]
            for a,b in zip(frames,frames[1:]):
                if a['connection']!=b['connection']:
                    gaps.append(dict(before_connection=a['connection'],after_connection=b['connection'],
                                     gap_seconds=b['last_packet_perf']-a['last_packet_perf'],
                                     last_packet_unix=a['local_unix'],first_packet_unix=b['local_unix']))
            summary.update(delivered_picture_rate=len(frames)/(frames[-1]['last_packet_perf']-frames[0]['last_packet_perf']),
                           low_windows_1s=windows(frames,1),low_windows_5s=windows(frames,5),connection_gaps=gaps)
            roles[role]=summary
        attempts={}
        for aid,progress in phase['server']['attempts'].items():
            attempts[aid]=dict(progress,lag_growth_per_wall_second=progress['accumulated_lateness_increase_seconds']/progress['wall_progress_seconds'])
        output[name]=dict(commit=commit,checks=checks,loads=loads,receivers=roles,
            **{key:value for key,value in phase.items() if key not in ('runtime','events','slow_scopes')},
            progress=attempts,supplement=supplement,
            internal_over500_per1000=phase['internal_rtp_intervals']['over_ms']['500']*1000/phase['internal_rtp_intervals']['n'],
            remote_over500_per1000=roles['output']['intervals']['over_ms']['500']*1000/roles['output']['intervals']['n'])
    unique={}
    maps={}
    for role in ('raw','output'):
        mapping=defaultdict(list)
        for row in base.read(ROOT/'candidate'/f'rtp-av-{role}.jsonl'):
            if row['kind']=='audio_au':mapping[row['sha256']].append(row)
        maps[role]=mapping
    unique['byte_identical_unique_aus']=sum(len(values)==len(maps['output'].get(key,[]))==1 for key,values in maps['raw'].items())
    output['av_checks']=dict(decoded={role:decoded(role) for role in ('raw','output')},
                             wire={role:audio_wire(role) for role in ('raw','output')},content=unique)
    output['player_observations']=base.read(ROOT/'player-observations-candidate.jsonl')
    (ROOT/'acceptance-summary.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({name:{'external':v['receivers']['output']['intervals'],'internal500per1000':v['internal_over500_per1000'],
                            'remote500per1000':v['remote_over500_per1000'],'progress':v['progress'],
                            'connection_gaps':v['receivers']['output']['connection_gaps']} for name,v in output.items() if name in ('candidate','default-current-load')},indent=2))


if __name__=='__main__':
    assert windows([dict(last_packet_perf=x/10) for x in range(30)],1)['min']==10
    build()
