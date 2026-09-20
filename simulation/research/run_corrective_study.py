"""Two finite GPU runs, each followed by its predeclared CPU confirmation."""
import json
import os
import subprocess
import sys
import time
from corrective_common import OUT,ROOT,RUNS


def main():
    assert json.loads((OUT/'dataset_audit.json').read_text())['status']=='PASS'
    state=dict(status='RUNNING',started_unix=time.time(),training={},evaluation={})
    def save():(OUT/'study_status.json').write_text(json.dumps(state,indent=2))
    flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0;pending=[]
    def training(arm,envs,iterations,kind):
        run=RUNS/f'microdinosaur_corrective_{arm}'/(f'20260914_{kind}_{envs}x{iterations}'+('_v2' if kind=='smoke' else ''))
        if kind=='smoke' and run.exists():
            assert json.loads((run/'run_provenance.json').read_text())['status']=='COMPLETE';return
        with (OUT/f'{kind}_{arm}.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,str(ROOT/'train_corrective.py'),'--arm',arm,'--out',str(run),
                '--envs',str(envs),'--iterations',str(iterations)],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT.parent,check=True,creationflags=flags)
        return run
    try:
        save()
        for arm in ('control','corrective'):training(arm,64,5,'smoke')
        for arm in ('control','corrective'):
            state['training'][arm]='RUNNING';save();run=training(arm,512,201,'train')
            state['training'][arm]='COMPLETE';state['evaluation'][arm]='RUNNING';save()
            log=(OUT/f'eval_{arm}.log').open('w',encoding='utf-8')
            process=subprocess.Popen([sys.executable,str(ROOT/'evaluate_corrective.py'),'--policy',str(run/'candidate.onnx'),
                '--label',arm],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT.parent,creationflags=flags)
            pending.append((arm,process,log));print('Training complete; evaluation started:',arm,flush=True)
        for arm,process,log in pending:
            code=process.wait();log.close();state['evaluation'][arm]='COMPLETE' if code==0 else 'FAILED';save()
            if code:raise RuntimeError(f'{arm} evaluation failed')
        state.update(status='COMPLETE',finished_unix=time.time());save()
    except Exception as error:
        state.update(status='FAILED',error=str(error));save();raise


if __name__=='__main__':main()
