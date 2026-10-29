"""Package metadata and hashes only; never copy media, models or per-pixel data."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    normal, profiled = [read(args.source/p/'outcome.json') for p in ('normal-full-a','async-full-b')]
    assert normal['output_sha256']==profiled['output_sha256'] and normal['source_sha256']==profiled['source_sha256']
    assert normal['result']['detection_stats']==profiled['result']['detection_stats']
    summary = dict(source_sha256=normal['source_sha256'], output_sha256=normal['output_sha256'],
        frames=normal['frames'], audio_packets=normal['audio_packets'],
        video_pts_and_audio_payload_equal=True, complete_output_bytes_equal=True,
        detection_stats=normal['result']['detection_stats'],
        modes=[{k:v for k,v in run.items() if k!='result'} | dict(
            media_elapsed_seconds=run['result']['stats']['elapsed_seconds'],
            video_decoder=run['result']['video_decoder'],video_encoder=run['result']['video_encoder']) for run in (normal,profiled)],
        total_wall_difference_percent=100*(profiled['total_wall_seconds']/normal['total_wall_seconds']-1),
        limits=['One full-file pair, not a statistically isolated observer effect',
                'E0 StreamProbe is inactive in process_video; no E0 on/off measurement',
                'The earlier async-full-a was intentionally aborted; overlapped an E1 GPU test and is excluded',
                'After the full-file run, only an async report byte-budget fail-closed guard was added; focused regression passed',
                'No actual RTX, real-time backlog, player, or new visual-quality ground truth acceptance'])
    write(args.output/'local-validation.json',summary)
    profile = read(args.source/'async-full-b/host-profile/report.json')
    write(args.output/'async-runtime.json',dict(runtime=profile['runtime'], async_tasks=profile['profile']['async_tasks']))
    path = Path('src/vision_shield/detection/async_profile.py')
    current = path.read_bytes()
    lines = current.splitlines(keepends=True)
    start = next(i for i,line in enumerate(lines) if b'# Future callbacks log exceptions' in line)
    end = next(i for i in range(start,len(lines)) if b'if not self.closed:' in lines[i])
    measured = b''.join(lines[:start]+lines[end:])
    assert hashlib.sha256(measured).hexdigest() == profile['profile']['async_tasks']['source_sha256']
    write(args.output/'post-run-change.json',dict(measured_source_sha256=hashlib.sha256(measured).hexdigest(),
        final_source_sha256=hashlib.sha256(current).hexdigest(), focused_tests='7 passed',
        patch=''.join(difflib.unified_diff(measured.decode().splitlines(True),current.decode().splitlines(True),
                                         fromfile='measured',tofile='final'))))
    write(args.output/'async-analysis.json',read(args.source/'async-analysis.json'))
    kernels = read(args.source/'kernel-probe/report.json')
    write(args.output/'kernel-summary.json',dict(gpu=kernels['gpu'],capability=kernels['capability'],
        notes=kernels['notes'],probes=[dict(model=p['model'],view=p['view'],shape=p['shape'],
            samples=p['report']['device_samples']) for p in kernels['probes']]))
    overhead = read(args.source/'observer-cost-final/report.json')
    timed = [r for r in overhead if r['measurement']=='timing']
    write(args.output/'observer-cost.json',dict(runs=overhead,
        median_cpu_increment_ms_per_fixture_frame=1000*(statistics.median(r['cpu_seconds'] for r in timed if r['enabled'])
            -statistics.median(r['cpu_seconds'] for r in timed if not r['enabled']))/1000,
        note='24 nested stages and one model future per fixture frame; memory and timing are separate runs; excludes E0/GPU/Redis/HTTP'))
    write(args.output/'scheduling-probe.json',read(args.source/'scheduling-probe.json'))
    write(args.output/'server-baseline.json',read(args.source/'server-current.json'))
    raw = []
    for path in sorted(args.source.rglob('*')):
        if path.is_file() and path.suffix in ('.json','.jsonl','.log'):
            raw.append(dict(path=path.as_posix(),bytes=path.stat().st_size,sha256=digest(path)))
    write(args.output/'raw-artifact-manifest.json',dict(files=raw,
        excluded_probe='async-full-a: intentionally aborted and concurrent, not used for performance or completeness'))
    print(json.dumps(dict(frames=normal['frames'],files=len(raw),output_hash_equal=True)))
