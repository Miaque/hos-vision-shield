"""Reproducible local CPU comparison; media stays local and reviews stay human-owned."""
import argparse
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
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
    if path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            return {'kind': 'image', 'width': image.width, 'height': image.height, 'frames': 1, 'times': [0], 'audio': False}
    times, width, height = [], None, None
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
            w, h = (frame.height, frame.width) if frame.rotation % 180 else (frame.width, frame.height)
            if width is not None and (w, h) != (width, height):
                raise ValueError('Changing frame dimensions')
            width, height = w, h
    if not times:
        raise ValueError('Empty video')
    return {'kind': 'video', 'width': width, 'height': height, 'frames': len(times), 'times': times, 'audio': audio, 'clock_errors': clock_errors}


def verify_source(record):
    if sha(record['path']) != record['sha256']:
        raise ValueError('Source changed: ' + record['id'])


def prepare(source, directory):
    source, directory = source.resolve(), directory.resolve()
    if directory == source or source in directory.parents or directory in source.parents:
        raise ValueError('Run directory must be separate from inputs')
    if directory.exists():
        raise ValueError('Run directory already exists')
    files = sorted(p for p in source.iterdir() if p.is_file())
    if sum(p.suffix.lower() == '.jpg' for p in files) != 16 or sum(p.suffix.lower() == '.mp4' for p in files) != 5 or len(files) != 21:
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
         models={m['path']:m['sha256'] for m in models}, manifest_sha256=sha(directory/'manifest.json'),
         status='prepared', python=sys.version, executable=sys.executable, device='cpu', sample_interval_seconds=5,
         lock_sha256=sha(ROOT/'uv.lock'), max_side=1280, padding=.15, opencv_threads=2, ort_threads=2, torch_threads=4))


def mask_pixels(size, boxes):
    width, height = size
    mask = np.zeros((height,width),np.uint8)
    for x1,y1,x2,y2 in boxes:
        if not all(math.isfinite(v) for v in (x1,y1,x2,y2)):
            raise ValueError('Non-finite mask')
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
    if np.any(points<0) or np.any(points[:,0]>=width) or np.any(points[:,1]>=height):
        raise ValueError('Target outside image')
    mask=np.zeros((height,width),np.uint8)
    cv2.fillPoly(mask,[np.rint(points).astype(np.int32)],1)
    return mask


def judge(size, boxes, polygon):
    target=target_pixels(size,polygon).astype(bool)
    covered=mask_pixels(size,boxes)[target]
    return 'full' if covered.all() else 'partial' if covered.any() else 'missed'


def summarize(samples, evidence):
    summary={'formal_samples':0,'pending_samples':0,'variants':{}}
    lookup={(item['sample_id'],item['variant']):item for item in evidence}
    for variant in ('baseline','candidate'):
        summary['variants'][variant]={kind:{cls:dict(full=0,partial=0,missed=0,miss_rate=None) for cls in ('face','plate')} for kind in ('image','video')}
    summary['false_mask_regions']={'baseline':0,'candidate':0}
    summary['judgments']=[]
    for sample in samples:
        if sample['review_status']!='human_confirmed':
            summary['pending_samples']+=1
            continue
        if not sample.get('reviewer') or not sample.get('reviewed_at'):
            raise ValueError('Human review needs actual reviewer and timestamp')
        if any((sample['id'],v) not in lookup for v in ('baseline','candidate')):
            raise ValueError('Reviewed sample missing output evidence')
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
            summary['false_mask_regions'][variant]+=sum(not np.any(all_targets[labels==i]) for i in range(1,count))
    for groups in summary['variants'].values():
        for group in groups.values():
            for counts in group.values():
                total=counts['full']+counts['partial']+counts['missed']
                counts['miss_rate']=(counts['partial']+counts['missed'])/total if total else None
    summary['false_mask_kind']='unknown: baseline masks do not expose classes'
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
    records=sorted(read(directory/'manifest.json'),key=lambda m:m['kind']!='image')
    original_redact=redaction.redact
    stop=threading.Event()
    failed=False
    for media in records:
        verify_source(media)
        record_path=directory/variant/(media['id']+'.json')
        if record_path.exists():
            previous=read(record_path)
            if previous['status']=='complete' and sha(directory/previous['output'])==previous['output_sha256']:
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
                Image.fromarray(result).save(directory/'samples'/(sample['id']+'-'+variant+'.png'))
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
            # Inspect using the actual format suffix; the production processing already closed the file.
            temporary.replace(output)
            checked=inspect_media(output)
            if any(checked[k]!=media[k] for k in ('kind','width','height','frames','audio')):
                raise ValueError('Output completeness mismatch')
            if not np.allclose(checked['times'],media['times'],atol=.001,rtol=0):
                raise ValueError('Output timestamps differ')
            result.update(status='complete',processing_seconds=elapsed,frames=frame_index,
                          output_sha256=sha(output),evidence=evidence)
            print(variant,media['id'],'complete',round(elapsed,2),flush=True)
        except KeyboardInterrupt:
            result.update(status='cancelled');write(record_path,result);raise
        except Exception as exc:
            failed=True
            result.update(status='failed',error=str(exc),processing_seconds=time.perf_counter()-started)
            print(variant,media['id'],'FAILED',str(exc),flush=True)
        write(record_path,result)
    if failed:
        raise SystemExit(1)


def run_variant(directory, variant):
    run=read(directory/'run.json')
    if sha(directory/'manifest.json')!=run['manifest_sha256'] or sha(ROOT/'uv.lock')!=run['lock_sha256']:
        raise ValueError('Frozen manifest or dependencies changed')
    if source_fingerprint(ROOT)!=run['candidate_source']:
        raise ValueError('Candidate source changed; prepare a new run')
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
    result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'_worker','--run-dir',str(directory),'--variant',variant],
                          cwd=ROOT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    run['status']='processed' if result.returncode==0 else 'incomplete'
    write(directory/'run.json',run)
    if result.returncode:raise SystemExit(result.returncode)


def report(directory):
    import csv
    records=read(directory/'manifest.json');samples=read(directory/'annotations.json')
    results=[read(p) for v in ('baseline','candidate') for p in (directory/v).glob('[0-9][0-9].json')]
    evidence=[e for r in results if r['status']=='complete' for e in r['evidence']]
    summary=summarize(samples,evidence)
    summary.update(completed=sum(r['status']=='complete' for r in results),expected=len(records)*2,
                   timing_scope='single batch run; initialization separate; decode+detect+mask+encode+sample evidence capture; excludes integrity verification and report')
    write(directory/'summary.json',summary)
    with (directory/'timings.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.writer(stream);writer.writerow(['variant','media_id','status','seconds','frames'])
        for r in results:writer.writerow([r['variant'],r['media_id'],r['status'],r.get('processing_seconds'),r.get('frames')])
    lookup={(r['media_id'],r['variant']):r for r in results}
    text=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>CPU 脱敏对比</title>',
          '<style>body{font:16px system-ui;background:#f4f6f8;color:#182433;margin:24px}section{background:white;padding:16px;margin:16px 0;border-radius:8px}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}img,video{width:100%;background:#ddd}h1{font-size:26px}pre{white-space:pre-wrap} @media(max-width:800px){.grid{grid-template-columns:1fr}}</style>',
          '<h1>main quality CPU / YOLO CPU</h1>',
          f'<p>完整产物 {summary["completed"]}/{summary["expected"]}；人工确认样本 {summary["formal_samples"]}；待复核 {summary["pending_samples"]}。未复核结果不代表准确率。</p>',
          '<p>视频均为完整处理；下方固定采样每5秒及末帧。效果优先，耗时为单次辅助记录。</p>']
    for media in records:
        text.append('<section><h2>'+html.escape(media['name'])+'</h2><div class="grid">')
        tag='img' if media['kind']=='image' else 'video controls preload="metadata"'
        text.append(f'<div>原始<{tag} src="{html.escape(Path(media["path"]).as_uri(),quote=True)}"></'+('div>' if media['kind']=='image' else 'video></div>'))
        for variant in ('baseline','candidate'):
            r=lookup.get((media['id'],variant))
            if r and r['status']=='complete':
                src=html.escape(r['output'].replace('\\','/'),quote=True)
                close='' if media['kind']=='image' else '</video>'
                text.append(f'<div>{variant} · {r["processing_seconds"]:.2f}s<{tag} src="{src}">{close}</div>')
            else:text.append('<div>'+variant+' · '+html.escape(str(r.get('error',r['status']) if r else '未运行'))+'</div>')
        text.append('</div>')
        for sample in (s for s in samples if s['media_id']==media['id']):
            text.append(f'<details><summary>{sample["id"]} · 帧 {sample["frame_index"]} · {html.escape(sample["review_status"])}</summary><div class="grid">')
            for variant in ('original','baseline','candidate'):
                name=f'samples/{sample["id"]}-{variant}.png'
                text.append(f'<div>{variant}<a href="{name}"><img loading="lazy" src="{name}"></a></div>' if (directory/name).exists() else '<div>未生成</div>')
            text.append('</div></details>')
        text.append('</section>')
    text.append('<h2>复核统计</h2><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre></html>')
    (directory/'index.html').write_text('\n'.join(text),'utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run','report','_worker'])
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--variant',choices=['baseline','candidate'])
    args=parser.parse_args();directory=args.run_dir.resolve()
    if args.command=='prepare':
        if args.source is None:parser.error('--source required')
        prepare(args.source,directory)
    elif args.command=='report':report(directory)
    else:
        if args.variant is None:parser.error('--variant required')
        (worker if args.command=='_worker' else run_variant)(directory,args.variant)
