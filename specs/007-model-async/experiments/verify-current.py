"""Read-only HTTP verification against the pinned E0 deployment, without SSH."""
import argparse
import json
from pathlib import Path
import sys
import time

import httpx


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--base', default='http://127.0.0.1:18000')
    args = parser.parse_args()
    directory = Path(__file__).resolve().parents[2]/'006-stream-backlog-pacing/experiments'
    sys.path.insert(0, str(directory))
    from e0_analyze import verify_runtime
    with httpx.Client(base_url=args.base, trust_env=False, timeout=8) as client:
        task_id = '27c39007-0aee-4185-b8fa-9d1ece4de058'
        task = client.get('/v1/tasks/'+task_id)
        task.raise_for_status()
        response = client.get('/v1/tasks/'+task_id+'/diagnostics')
        response.raise_for_status()
        runtime = response.json()['live']['runtime']
        ref = '21bd631d5a3d02de052974b57f73428560944804'
        verify_runtime(runtime, ref)
        result = dict(captured_unix=time.time(), ref=ref, matched_source_modules=20,
                      state=task.json()['state'], ready=task.json()['ready'], runtime=runtime)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps({k:v for k,v in result.items() if k!='runtime'}))
