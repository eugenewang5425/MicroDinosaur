"""Bounded three-arm study; train sequentially and evaluate completed arms."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'20260914_demonstrations'
RL=Path('D:/microduck_rl/logs/rsl_rl')


def main():
    arms=('legacy','archive','demonstration');evaluations=[]
    state=dict(status='RUNNING',started_unix=time.time(),training={},evaluation={})
    def save():(OUT/'study_status.json').write_text(json.dumps(state,indent=2))
    for arm in arms:
        smoke=RL/f'microdinosaur_demo_{arm}/20260914_smoke_64x5/run_provenance.json'
        assert json.loads(smoke.read_text())['status']=='COMPLETE'
    save()
    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    try:
        for arm in arms:
            run=RL/f'microdinosaur_demo_{arm}/20260914_train_512x101'
            state['training'][arm]='RUNNING';save()
            with (OUT/f'train_{arm}.log').open('w',encoding='utf-8') as log:
                subprocess.run([sys.executable,str(ROOT/'train_demonstration.py'),'--arm',arm,'--out',str(run),
                    '--envs','512','--iterations','101'],cwd=ROOT.parent,stdout=log,stderr=subprocess.STDOUT,check=True,creationflags=creationflags)
            state['training'][arm]='COMPLETE';save()
            log=(OUT/f'eval_{arm}.log').open('w',encoding='utf-8')
            process=subprocess.Popen([sys.executable,str(ROOT/'evaluate_demonstrations.py'),'--policy',str(run/'candidate.onnx'),
                '--label',arm],cwd=ROOT.parent,stdout=log,stderr=subprocess.STDOUT,creationflags=creationflags)
            evaluations.append((arm,process,log));state['evaluation'][arm]='RUNNING';save()
            print(json.dumps(dict(training_completed=arm,evaluation_started=arm)),flush=True)
        for arm,process,log in evaluations:
            code=process.wait();log.close();state['evaluation'][arm]='COMPLETE' if code==0 else 'FAILED';save()
            if code:raise RuntimeError(f'{arm} evaluation exited {code}')
        state.update(status='COMPLETE',finished_unix=time.time());save()
    except Exception as error:
        state.update(status='FAILED',error=str(error));save()
        for _,process,log in evaluations:
            # Keep already authorized evaluations alive; final status and logs
            # remain inspectable if a later training arm fails.
            log.close()
        raise


if __name__=='__main__':main()
