"""Archive source/model provenance and verify the complete demonstration study."""
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'20260914_demonstrations'
RL=Path('D:/microduck_rl')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    assert read(OUT/'study_status.json')['status']=='COMPLETE'
    assert read(OUT/'dataset_audit.json')['status']=='PASS'
    assert read(OUT/'summary.json')['formal_trials']==135
    assert all(r['status']=='PASS' for r in read(OUT/'training_audit.json').values())
    source_files=set();trace_count=0;names=None
    for arm in ('legacy','archive','demonstration'):
        folder=RL/f'logs/rsl_rl/microdinosaur_demo_{arm}/20260914_train_512x101'
        provenance=read(folder/'run_provenance.json')
        for p,digest in provenance['source_hashes'].items():
            path=Path(p);assert sha(path)==digest;source_files.add(path)
        model_dir=OUT/'models'/arm;model_dir.mkdir(parents=True,exist_ok=False)
        for name in ('candidate.onnx','run_provenance.json'):shutil.copy2(folder/name,model_dir/name)
        assert sha(model_dir/'candidate.onnx')==provenance['onnx_sha256']
        rows=read(OUT/f'evaluation/{arm}/matrix.json');assert len(rows)==45
        current={r['name'] for r in rows};assert len(current)==45
        if names is None:names=current
        else:assert names==current
        for r in rows:
            assert r['policy_sha256']==provenance['onnx_sha256']
            assert r==read(OUT/f"evaluation/{arm}/{r['name']}.json")
            for name,digest in r['source_hashes'].items():
                path=ROOT/name;assert sha(path)==digest;source_files.add(path)
            with np.load(OUT/f"evaluation/{arm}/{r['name']}.npz") as z:
                for name in z.files:assert np.isfinite(z[name]).all();trace_count+=1
    for pattern in ('*demonstration*.py','audit_step_dataset.py','demonstration_expert.py'):
        source_files.update(ROOT.glob(pattern))
    source_files.add(ROOT/'probe_expert_recovery.py')
    dest=OUT/'source_snapshot';dest.mkdir(exist_ok=False);sources=[]
    for p in sorted(source_files):
        relative=Path('rl')/p.relative_to(RL) if p.is_relative_to(RL) else Path('research')/p.relative_to(ROOT)
        target=dest/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
        assert sha(target)==sha(p)
        if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
        sources.append(dict(original=str(p),snapshot=str(target.relative_to(OUT)),sha256=sha(p)))
    for report in (OUT/'RESULTS.md',OUT/'TEACHER_AUDIT.md'):
        for target in re.findall(r'\]\(([^)]+)\)',report.read_text(encoding='utf-8')):
            if not target.startswith(('http://','https://','#')):assert (report.parent/target).exists(),target
    production=RL/'microdinosaur_p2.onnx';assert sha(production)=='0804114efd1e457ec2c297d709c0464fd0c2bfe251cbb36db7de74543857b0e3'
    result=dict(status='PASS',formal_trials=135,finite_trace_arrays=trace_count,source_files=sources,production_unchanged=True,
        artifacts={str(p.relative_to(OUT)):dict(bytes=p.stat().st_size,sha256=sha(p))
            for p in sorted(OUT.rglob('*')) if p.is_file() and p.suffix!='.log' and p.name!='artifact_manifest.json'})
    (OUT/'artifact_manifest.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(status='PASS',formal_trials=135,finite_arrays=trace_count,source_files=len(sources),artifacts=len(result['artifacts'])),indent=2))


if __name__=='__main__':main()
