"""Reproducible local CPU comparison; media stays local and reviews stay human-owned."""
import argparse
from datetime import datetime
import hashlib
import html
import io
from importlib.metadata import version
import json
import math
import os
import platform
from pathlib import Path
from urllib.parse import quote
import shutil
import subprocess
import sys
import threading
import time
import zipfile

import av
import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
BASE = '0664be4ae9edb6d0eac7bb2635d5563dad9d7956'
PARAMETERS=dict(device='cpu',sample_interval_seconds=5,max_side=1280,padding=.15,
                yolo_network_side=640,opencv_threads=2,ort_threads=2,torch_threads=4)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.part')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), 'utf-8')
    temporary.replace(path)


def source_fingerprint(root):
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted((root/'src').rglob('*.py'))}


def tool_fingerprint():
    return {name:sha(ROOT/name) for name in ('scripts/compare-yolo.py','scripts/repair-comparison-video.py',
        'pyproject.toml','models/manifest.json','models/cpu-manifest.json','models/yolo-manifest.json')}


def sample_sources(samples):
    return [{k:s[k] for k in ('id','media_id','frame_index','kind','width','height','pts','time_base') if k in s} for s in samples]


def verify_samples(directory, samples):
    validate_annotations(samples)
    if sample_sources(samples)!=read(directory/'run.json')['sample_sources']:
        raise ValueError('Frozen sample identity or geometry changed')


def sample_indices(times):
    selected, threshold = [], 0
    for index, moment in enumerate(times):
        if moment >= threshold:
            selected.append(index)
            threshold = (math.floor(moment / 5) + 1) * 5
    if times and (not selected or selected[-1] != len(times)-1):
        selected.append(len(times)-1)
    return selected


def inspect_media(path):
    path=Path(path)
    if Path(path.name.removesuffix('.part')).suffix.lower() in {'.jpg', '.jpeg', '.png'}:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            return {'kind': 'image', 'width': image.width, 'height': image.height, 'frames': 1, 'times': [0], 'audio': False}
    times, timestamps, width, height = [], [], None, None
    clock_errors = []
    with av.open(str(path)) as reader:
        audio = bool(reader.streams.audio)
        for frame in reader.decode(video=0):
            if frame.pts is None or frame.time_base is None:
                raise ValueError('Missing video timestamp')
            moment = float(frame.pts * frame.time_base)
            if times and moment <= times[-1]:
                clock_errors.append(len(times))
            times.append(moment)
            timestamps.append([frame.pts,str(frame.time_base)])
            w, h = (frame.height, frame.width) if frame.rotation % 180 else (frame.width, frame.height)
            if width is not None and (w, h) != (width, height):
                raise ValueError('Changing frame dimensions')
            width, height = w, h
    if not times:
        raise ValueError('Empty video')
    digest=hashlib.sha256();audio_packets=[]
    with av.open(str(path)) as reader:
        audio_info=[dict(codec=s.codec_context.name,rate=s.codec_context.sample_rate,channels=s.codec_context.channels) for s in reader.streams.audio]
        if len(audio_info)>1:raise ValueError('Multiple audio streams')
        if audio:
            for packet in reader.demux(audio=0):
                if packet.dts is None:continue
                if packet.pts is None or packet.time_base is None:raise ValueError('Missing audio timestamp')
                digest.update(bytes(packet))
                audio_packets.append([float(t*packet.time_base) for t in (packet.pts,packet.dts,packet.duration)])
    return dict(kind='video',width=width,height=height,frames=len(times),times=times,timestamps=timestamps,
                audio=audio,audio_info=audio_info,audio_packets=audio_packets,audio_sha256=digest.hexdigest(),clock_errors=clock_errors)


def verify_output(original, checked):
    if any(checked[k]!=original[k] for k in ('kind','width','height','frames','audio')):
        raise ValueError('Output completeness mismatch')
    if checked.get('clock_errors') or not np.allclose(checked['times'],original['times'],atol=.001,rtol=0):
        raise ValueError('Output timestamps differ')
    if original['audio']:
        if (checked['audio_info']!=original['audio_info'] or checked['audio_sha256']!=original['audio_sha256']
                or len(checked['audio_packets'])!=len(original['audio_packets'])
                or not np.allclose(checked['audio_packets'],original['audio_packets'],atol=.001,rtol=0)):
            raise ValueError('Output audio payload, duration or timestamps differ')


def save_samples(output, directory, selected, variant):
    if output.suffix=='.jpg':
        with Image.open(output) as image:
            image.save(directory/'samples'/(selected[0]['id']+'-'+variant+'.png'))
    else:
        with av.open(str(output)) as reader:
            for index,frame in enumerate(reader.decode(video=0)):
                if index in selected:
                    frame.to_image().save(directory/'samples'/(selected[index]['id']+'-'+variant+'.png'))


def verify_source(record):
    if sha(record['path']) != record['sha256']:
        raise ValueError('Source changed: ' + record['id'])


def prepare(source, directory, *, images_only=False):
    source, directory = source.resolve(), directory.resolve()
    if directory == source or source in directory.parents or directory in source.parents:
        raise ValueError('Run directory must be separate from inputs')
    if directory.exists():
        raise ValueError('Run directory already exists')
    files = sorted(p for p in source.iterdir() if p.is_file())
    if images_only:
        files=[p for p in files if p.suffix.lower()=='.jpg']
    if sum(p.suffix.lower() == '.jpg' for p in files) != 16 or (not images_only and (sum(p.suffix.lower() == '.mp4' for p in files) != 5 or len(files) != 21)):
        raise ValueError('Expected exactly 16 JPEGs and 5 MP4s')
    directory.mkdir(parents=True)
    (directory/'samples').mkdir()
    records, samples = [], []
    for index, path in enumerate(files):
        media = dict(id=f'{index:02d}', path=str(path), name=path.name, sha256=sha(path), **inspect_media(path))
        records.append(media)
        indices = sample_indices([t-media['times'][0] for t in media['times']])
        if media['kind'] == 'image':
            with Image.open(path) as image:
                images = [(0, np.array(ImageOps.exif_transpose(image).convert('RGB')))]
            for frame_index, pixels in images:
                sample_id = f'{media["id"]}-{frame_index:06d}'
                Image.fromarray(pixels).save(directory/'samples'/f'{sample_id}-original.png')
                samples.append(dict(id=sample_id, media_id=media['id'], frame_index=frame_index, kind='image',
                                    width=media['width'], height=media['height'], targets=[], review_status='pending'))
        else:
            with av.open(str(path)) as reader:
                for frame_index, frame in enumerate(reader.decode(video=0)):
                    if frame_index not in indices:
                        continue
                    pixels = frame.to_ndarray(format='rgb24')
                    if frame.rotation:
                        pixels = np.rot90(pixels, frame.rotation//90).copy()
                    sample_id = f'{media["id"]}-{frame_index:06d}'
                    Image.fromarray(pixels).save(directory/'samples'/f'{sample_id}-original.png')
                    samples.append(dict(id=sample_id, media_id=media['id'], frame_index=frame_index, kind='video',
                                        pts=frame.pts, time_base=str(frame.time_base), width=media['width'], height=media['height'],
                                        targets=[], review_status='pending'))
        print('prepared', media['id'], media['frames'], flush=True)
    models = [m for name in ('manifest.json','cpu-manifest.json','yolo-manifest.json') for m in read(ROOT/'models'/name)['models']]
    for model in models:
        if sha(ROOT/'models'/model['path']) != model['sha256']:
            raise ValueError('Model hash mismatch')
    write(directory/'manifest.json', records)
    write(directory/'annotations.json', samples)
    write(directory/'run.json', dict(baseline_commit=BASE, candidate_source=source_fingerprint(ROOT),
         candidate_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         candidate_diff_sha256=hashlib.sha256(subprocess.check_output(['git','diff','HEAD','--','src'],cwd=ROOT)).hexdigest(),
         models={m['path']:m['sha256'] for m in models}, manifest_sha256=sha(directory/'manifest.json'),
         status='prepared',scope='images' if images_only else 'all', python=sys.version, executable=sys.executable,parameters=PARAMETERS,tool_source=tool_fingerprint(),
         environment=dict(cpu=platform.processor(),os=platform.platform(),dependencies={p:version(p) for p in
             ('av','numpy','opencv-python-headless','onnxruntime','Pillow','torch','torchvision','pydantic-settings')}),
         sample_sources=sample_sources(samples),lock_sha256=sha(ROOT/'uv.lock')))


def mask_pixels(size, boxes):
    width, height = size
    mask = np.zeros((height,width),np.uint8)
    for x1,y1,x2,y2 in boxes:
        if not all(math.isfinite(v) for v in (x1,y1,x2,y2)):
            raise ValueError('Non-finite mask')
        if x2<=x1 or y2<=y1:
            raise ValueError('Invalid mask dimensions')
        left,top=max(0,math.floor(x1)),max(0,math.floor(y1))
        right,bottom=min(width,math.ceil(x2)),min(height,math.ceil(y2))
        if right>left and bottom>top:
            mask[top:bottom,left:right]=1
    return mask


def target_pixels(size, polygon):
    width,height=size
    points=np.asarray(polygon,dtype=float)
    if points.ndim!=2 or points.shape[1]!=2 or len(points)<3 or not np.isfinite(points).all():
        raise ValueError('Invalid target polygon')
    if cv2.contourArea(points.astype(np.float32))==0:
        raise ValueError('Empty target polygon')
    if np.any(points<0) or np.any(points[:,0]>=width) or np.any(points[:,1]>=height):
        raise ValueError('Target outside image')
    mask=np.zeros((height,width),np.uint8)
    cv2.fillPoly(mask,[np.rint(points).astype(np.int32)],1)
    return mask


def judge(size, boxes, polygon):
    target=target_pixels(size,polygon).astype(bool)
    covered=mask_pixels(size,boxes)[target]
    return 'full' if covered.all() else 'partial' if covered.any() else 'missed'


def validate_annotations(samples):
    seen=set()
    for sample in samples:
        if sample['id'] in seen or sample['kind'] not in ('image','video') or sample['review_status'] not in ('pending','human_confirmed','disputed'):
            raise ValueError('Invalid or duplicate sample')
        seen.add(sample['id'])
        if any(type(sample[k]) is not int or sample[k]<=0 for k in ('width','height')) or type(sample['frame_index']) is not int or sample['frame_index']<0:
            raise ValueError('Invalid sample geometry')
        if sample.get('reviewed_at'):
            if datetime.fromisoformat(sample['reviewed_at']).utcoffset() is None:
                raise ValueError('Review timestamp needs timezone')
        if sample['review_status']=='human_confirmed' and (not isinstance(sample.get('reviewer'),str) or not sample['reviewer'].strip() or not sample.get('reviewed_at')):
            raise ValueError('Human review needs actual reviewer and timestamp')
        targets=set()
        for target in sample['targets']:
            if not isinstance(target['id'],str) or not target['id'] or target['id'] in targets or target['kind'] not in ('face','plate') or type(target.get('uncertain',False)) is not bool:
                raise ValueError('Invalid or duplicate target')
            targets.add(target['id'])
            target_pixels((sample['width'],sample['height']),target['polygon'])
        for case in sample.get('oversized_cases',[]):
            if case['variant'] not in ('baseline','candidate') or not isinstance(case.get('note'),str) or not case['note'].strip():
                raise ValueError('Invalid oversized-mask review')


def summarize(samples, evidence):
    validate_annotations(samples)
    summary={'formal_samples':0,'pending_samples':0,'disputed_samples':0,'unavailable_samples':0,'variants':{},'new_misses':[],'false_mask_cases':[]}
    lookup={(item['sample_id'],item['variant']):item for item in evidence}
    if len(lookup)!=len(evidence) or any(e['variant'] not in ('baseline','candidate') for e in evidence):
        raise ValueError('Invalid or duplicate evidence')
    for variant in ('baseline','candidate'):
        summary['variants'][variant]={kind:{cls:dict(full=0,partial=0,missed=0,miss_rate=None) for cls in ('face','plate')} for kind in ('image','video')}
    summary['false_mask_regions']={'baseline':0,'candidate':0}
    summary['judgments']=[]
    for sample in samples:
        if sample['review_status']!='human_confirmed':
            summary['pending_samples']+=1
            summary['disputed_samples']+=sample['review_status']=='disputed'
            continue
        if not sample.get('reviewer') or not sample.get('reviewed_at'):
            raise ValueError('Human review needs actual reviewer and timestamp')
        if any((sample['id'],v) not in lookup for v in ('baseline','candidate')):
            summary['unavailable_samples']+=1
            continue
        summary['formal_samples']+=1
        size=sample['width'],sample['height']
        for variant in ('baseline','candidate'):
            boxes=lookup[(sample['id'],variant)]['boxes']
            all_targets=np.zeros((size[1],size[0]),np.uint8)
            for target in sample['targets']:
                all_targets|=target_pixels(size,target['polygon'])
                if target.get('uncertain'):
                    continue
                if target['kind'] not in ('face','plate'):
                    raise ValueError('Invalid target kind')
                outcome=judge(size,boxes,target['polygon'])
                summary['variants'][variant][sample['kind']][target['kind']][outcome]+=1
                summary['judgments'].append(dict(sample_id=sample['id'],variant=variant,target_id=target['id'],outcome=outcome))
            count,labels=cv2.connectedComponents(mask_pixels(size,boxes))
            for i in range(1,count):
                if not np.any(all_targets[labels==i]):
                    summary['false_mask_regions'][variant]+=1
                    ys,xs=np.where(labels==i)
                    summary['false_mask_cases'].append(dict(sample_id=sample['id'],variant=variant,kind='unknown',
                        box=[int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1]))
        outcomes={(j['variant'],j['target_id']):j['outcome'] for j in summary['judgments'] if j['sample_id']==sample['id']}
        for target in sample['targets']:
            if target.get('uncertain'):continue
            before,after=(outcomes[v,target['id']] for v in ('baseline','candidate'))
            if ('full','partial','missed').index(after)>('full','partial','missed').index(before):
                summary['new_misses'].append(dict(sample_id=sample['id'],target_id=target['id'],kind=target['kind'],baseline=before,candidate=after))
    for groups in summary['variants'].values():
        for group in groups.values():
            for counts in group.values():
                total=counts['full']+counts['partial']+counts['missed']
                counts['miss_rate']=(counts['partial']+counts['missed'])/total if total else None
    summary['false_mask_kind']='unknown: baseline masks do not expose classes'
    summary['unreviewed_fraction']=summary['pending_samples']/len(samples) if samples else None
    summary['oversized_cases']=[dict(sample_id=s['id'],**case) for s in samples for case in s.get('oversized_cases',[])]
    return summary


def worker(directory, variant):
    run=read(directory/'run.json')
    source_root=directory/'baseline-source' if variant=='baseline' else ROOT
    sys.path.insert(0,str(source_root/'src'))
    from vision_shield.config import Settings
    from vision_shield.detection import load_detector
    import vision_shield.redaction as redaction
    import vision_shield.media.images as images
    from vision_shield.media.video import process_video
    actual=Path(sys.modules['vision_shield'].__file__).resolve()
    if source_root.resolve() not in actual.parents:
        raise ValueError('Wrong imported source: '+str(actual))
    for field in Settings.model_fields:
        os.environ.pop(field.upper(),None)
    options=dict(inference_device='cpu',detection_backend='quality' if variant=='baseline' else 'yolo',
                 detection_max_side=1280,detection_box_padding=.15,
                 face_model_path=ROOT/'models'/('yunet_face.onnx' if variant=='baseline' else 'yolov8n-face.onnx'),
                 plate_model_path=ROOT/'models'/('hyperlpr_plate.onnx' if variant=='baseline' else 'yolov11s-plate.onnx'),
                 egoblur_face_model_path=ROOT/'models/face.jit',egoblur_plate_model_path=ROOT/'models/license_plate.jit')
    config=Settings(_env_file=None,**options).model_dump()
    started=time.perf_counter();detector=load_detector(config);init=time.perf_counter()-started
    write(directory/variant/'initialization.json',dict(seconds=init,source=str(actual),config={k:str(v) for k,v in options.items()}))
    samples=read(directory/'annotations.json')
    verify_samples(directory,samples)
    records=sorted(read(directory/'manifest.json'),key=lambda m:m['kind']!='image')
    original_redact=redaction.redact
    original_image_redact=images.redact
    stop=threading.Event()
    failed=False
    for media in records:
        verify_source(media)
        record_path=directory/variant/(media['id']+'.json')
        if record_path.exists():
            previous=read(record_path)
            if previous['status']=='complete':
                if sha(directory/previous['output'])!=previous['output_sha256']:
                    raise ValueError('Completed output changed: '+media['id'])
                continue
        suffix='.jpg' if media['kind']=='image' else '.mp4'
        output=directory/variant/(media['id']+suffix);output.parent.mkdir(exist_ok=True)
        temporary=output.with_name(output.name+'.part')
        selected={s['frame_index']:s for s in samples if s['media_id']==media['id']}
        evidence=[];frame_index=0
        def recording_redact(pixels,boxes):
            nonlocal frame_index
            boxes=list(boxes)
            result=original_redact(pixels,boxes)
            if frame_index in selected:
                sample=selected[frame_index]
                evidence.append(dict(sample_id=sample['id'],variant=variant,boxes=boxes))
            frame_index+=1
            return result
        redaction.redact=recording_redact
        images.redact=recording_redact
        started=time.perf_counter()
        result=dict(media_id=media['id'],variant=variant,status='running',output=str(output.relative_to(directory)))
        write(record_path,result)
        try:
            if media.get('clock_errors'):
                raise ValueError('Preflight: duplicate/non-increasing source timestamps at frames '+str(media['clock_errors']))
            if media['kind']=='image':
                images.process_image(media['path'],temporary,detector,config['max_image_pixels'])
            else:
                process_video(media['path'],temporary,detector,config,stop,lambda event: None)
            elapsed=time.perf_counter()-started
            checked=inspect_media(temporary)
            verify_output(media,checked)
            if len(evidence)!=len(selected):raise ValueError('Missing sample masking evidence')
            temporary.replace(output)
            save_samples(output,directory,selected,variant)
            result.update(status='complete',processing_seconds=elapsed,frames=frame_index,
                          output_sha256=sha(output),evidence=evidence,output_integrity=checked,
                          timing_scope='decode+detect+mask+encode; excludes initialization, verification, sample PNGs and report',measurement_count=1)
            print(variant,media['id'],'complete',round(elapsed,2),flush=True)
        except KeyboardInterrupt:
            result.update(status='cancelled');write(record_path,result);raise
        except Exception as exc:
            failed=True
            result.update(status='failed',error=str(exc),processing_seconds=time.perf_counter()-started)
            print(variant,media['id'],'FAILED',str(exc),flush=True)
        finally:
            redaction.redact=original_redact
            images.redact=original_image_redact
        write(record_path,result)
    if failed:
        raise SystemExit(1)


def result_status(directory):
    for media in read(directory/'manifest.json'):
        for variant in ('baseline','candidate'):
            path=directory/variant/(media['id']+'.json')
            if not path.exists() or read(path)['status']!='complete':
                return 'incomplete'
    return 'processed'


def run_variant(directory, variant):
    run=read(directory/'run.json')
    if sha(directory/'manifest.json')!=run['manifest_sha256'] or sha(ROOT/'uv.lock')!=run['lock_sha256']:
        raise ValueError('Frozen manifest or dependencies changed')
    if source_fingerprint(ROOT)!=run['candidate_source']:
        raise ValueError('Candidate source changed; prepare a new run')
    if tool_fingerprint()!=run['tool_source'] or PARAMETERS!=run['parameters'] or run['baseline_commit']!=BASE:
        raise ValueError('Frozen comparison tools or parameters changed; prepare a new run')
    verify_samples(directory,read(directory/'annotations.json'))
    for name,digest in run['models'].items():
        if sha(ROOT/'models'/name)!=digest:raise ValueError('Model changed')
    snapshot=directory/'baseline-source'
    if not snapshot.exists():
        archive=subprocess.check_output(['git','archive','--format=zip',BASE,'src'],cwd=ROOT)
        with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
            bundle.extractall(snapshot)
        write(directory/'baseline-source.json',source_fingerprint(snapshot))
    if source_fingerprint(snapshot)!=read(directory/'baseline-source.json'):
        raise ValueError('Baseline snapshot changed')
    run['status']='running';write(directory/'run.json',run)
    try:
        result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'_worker','--run-dir',str(directory),'--variant',variant],
                              cwd=ROOT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    finally:
        run['status']=result_status(directory)
        write(directory/'run.json',run)
    if result.returncode:raise SystemExit(result.returncode)


def report(directory):
    import csv
    records=read(directory/'manifest.json');samples=read(directory/'annotations.json')
    run=read(directory/'run.json')
    if sha(directory/'manifest.json')!=run['manifest_sha256']:raise ValueError('Frozen manifest changed')
    verify_samples(directory,samples)
    results=[read(p) for v in ('baseline','candidate') for p in (directory/v).glob('[0-9][0-9].json')]
    evidence=[e for r in results if r['status']=='complete' for e in r['evidence']]
    summary=summarize(samples,evidence)
    summary.update(completed=sum(r['status']=='complete' for r in results),expected=len(records)*2,
                   annotation_revision=sha(directory/'annotations.json'),status=result_status(directory),scope=run.get('scope','all'),
                   timing_scope='single batch run; initialization separate; decode+detect+mask+encode; excludes integrity verification, sample PNGs and report')
    original_ids={m['id'] for m in records if not m.get('original_id')}
    summary['original_outputs']=dict(completed=sum(r['status']=='complete' and r['media_id'] in original_ids for r in results),expected=len(original_ids)*2)
    summary['supplemental_outputs']=dict(completed=sum(r['status']=='complete' and r['media_id'] not in original_ids for r in results),expected=(len(records)-len(original_ids))*2)
    write(directory/'summary.json',summary)
    with (directory/'timings.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.writer(stream);writer.writerow(['variant','media_id','status','seconds','frames'])
        for r in results:writer.writerow([r['variant'],r['media_id'],r['status'],r.get('processing_seconds'),r.get('frames')])
    lookup={(r['media_id'],r['variant']):r for r in results}
    text=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>CPU 脱敏对比 · '+('待人工复核' if summary['pending_samples'] else '已复核')+'</title>',
          '<style>body{font:16px system-ui;background:#f4f6f8;color:#182433;margin:24px}section{background:white;padding:16px;margin:16px 0;border-radius:8px}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}img,video{width:100%;background:#ddd}h1{font-size:26px}pre{white-space:pre-wrap} @media(max-width:800px){.grid{grid-template-columns:1fr}}</style>',
          '<h1>main quality CPU / YOLO CPU'+(' · 待人工复核' if summary['pending_samples'] else '')+'</h1>',
          f'<p>完整产物 {summary["completed"]}/{summary["expected"]}；人工确认样本 {summary["formal_samples"]}；待复核 {summary["pending_samples"]}。未复核结果不代表准确率。</p>',
          '<p>'+('本次仅比较16张图片；视频未纳入本轮运行。' if run.get('scope')=='images' else '成功视频均通过全长完整性检查，失败或未完成项单列；下方固定采样每5秒及末帧。')+'效果优先，耗时为单次辅助记录。</p>']
    for media in records:
        original=directory/'originals'/(media['id']+Path(media['path']).suffix.lower())
        if not original.exists():
            verify_source(media)
            original.parent.mkdir(exist_ok=True)
            shutil.copyfile(media['path'],original)
        if sha(original)!=media['sha256']:raise ValueError('Original preview changed')
        text.append('<section><h2>'+html.escape(media['name'])+'</h2>')
        if media.get('original_id'):
            text.append('<p>补充修正副本：原始素材 '+html.escape(media['original_id'])+' 的失败保留；此副本单列。</p>')
        text.append('<div class="grid">')
        tag='img' if media['kind']=='image' else 'video controls preload="metadata"'
        text.append(f'<div>原始<{tag} src="{quote(original.relative_to(directory).as_posix())}"></'+('div>' if media['kind']=='image' else 'video></div>'))
        for variant in ('baseline','candidate'):
            r=lookup.get((media['id'],variant))
            if r and r['status']=='complete':
                src=html.escape(r['output'].replace('\\','/'),quote=True)
                close='' if media['kind']=='image' else '</video>'
                text.append(f'<div>{variant} · {r["processing_seconds"]:.2f}s<{tag} src="{src}">{close}</div>')
            else:text.append('<div>'+variant+' · '+html.escape(str(r.get('error',r['status']) if r else '未运行'))+'</div>')
        text.append('</div>')
        for sample in (s for s in samples if s['media_id']==media['id']):
            text.append(f'<details id="{html.escape(sample["id"],quote=True)}"><summary>{html.escape(sample["id"])} · 帧 {sample["frame_index"]} · {html.escape(sample["review_status"])}</summary><div class="grid">')
            for variant in ('original','baseline','candidate'):
                name=f'samples/{sample["id"]}-{variant}.png'
                url=html.escape(quote(name),quote=True)
                text.append(f'<div>{variant}<a href="{url}"><img loading="lazy" src="{url}"></a></div>' if (directory/name).exists() else '<div>未生成</div>')
            text.append('</div></details>')
        text.append('</section>')
    text.append('<h2>已确认新增漏遮</h2>')
    for miss in summary['new_misses']:
        text.append('<p><a href="#'+html.escape(miss['sample_id'],quote=True)+'">'+html.escape(miss['sample_id']+' / '+miss['target_id'])+'</a> '+miss['baseline']+' → '+miss['candidate']+'</p>')
    text.append('<h2>复核统计</h2><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre></html>')
    (directory/'index.html').write_text('\n'.join(text),'utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run','report','_worker'])
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--variant',choices=['baseline','candidate'])
    parser.add_argument('--images-only',action='store_true',help='prepare only: compare the 16 JPEGs without opening videos')
    args=parser.parse_args();directory=args.run_dir.resolve()
    if args.images_only and args.command!='prepare':parser.error('--images-only is only used by prepare; run uses the frozen manifest')
    if args.command=='prepare':
        if args.source is None:parser.error('--source required')
        prepare(args.source,directory,images_only=args.images_only)
    elif args.command=='report':report(directory)
    else:
        if args.variant is None:parser.error('--variant required')
        (worker if args.command=='_worker' else run_variant)(directory,args.variant)
