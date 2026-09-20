"""Consume completed runs in this bounded experiment; exits with the train queue."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/20260914_head_gaze_ablation'
status_path=OUT/'evaluation_status.json'
state={'status':'RUNNING','jobs':[]}
def save():status_path.write_text(json.dumps(state,indent=2),encoding='utf-8')
save()
try:
    while True:
        train=json.loads((OUT/'train_status.json').read_text())
        if train['status']=='FAILED':raise RuntimeError('Training queue failed; inspect train_status.json')
        completed={j['label'] for j in state['jobs'] if j['status']=='COMPLETE'}
        for job in train['jobs']:
            if not job['status'].startswith('COMPLETE'):continue
            label=f"s{job['seed']}_{job['arm']}"
            if label in completed:continue
            run=Path(job['run']);dest=OUT/'evaluation'/label
            item={'label':label,'status':'RUNNING','started_unix':time.time(),'out':str(dest)}
            state['jobs'].append(item);save()
            cmd=[sys.executable,'-u',str(ROOT/'research/evaluate_gaze_ablation.py'),
                '--policy',f'{label}={run / "candidate.onnx"}','--out',str(dest),'--resume']
            with (OUT/f'eval_{label}.log').open('w') as log,(OUT/f'eval_{label}.err.log').open('w') as err:
                code=subprocess.call(cmd,cwd=ROOT,stdout=log,stderr=err,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            if code:raise RuntimeError(f'Evaluation failed: {label}')
            item.update(status='COMPLETE',finished_unix=time.time());save()
            print(json.dumps(item),flush=True)
        if train['status']=='COMPLETE' and len(state['jobs'])==6:
            state.update(status='COMPLETE',finished_unix=time.time());save();break
        time.sleep(3)
except Exception as exc:
    state.update(status='FAILED',error=str(exc));save();raise
