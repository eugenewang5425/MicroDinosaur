"""Archive research sources/models and hash every delivered study artifact."""
import ast
import json
from pathlib import Path
import shutil
import numpy as np
from corrective_common import ROOT,OUT,RUNS
from evaluate_policy import sha


def main():
    assert json.loads((OUT/'training_audit.json').read_text())['status']=='PASS'
    assert json.loads((OUT/'dataset_audit.json').read_text())['status']=='PASS'
    sources=set(ROOT.glob('*corrective*.py'))|{ROOT/'step_render_overlay.py'}
    for arm in ('control','corrective'):
        run=RUNS/f'microdinosaur_corrective_{arm}/20260914_train_512x201'
        provenance=json.loads((run/'run_provenance.json').read_text())
        sources.update(Path(p) for p in provenance['source_hashes'])
        dest=OUT/'models'/arm;dest.mkdir(parents=True,exist_ok=True)
        for name in ('candidate.onnx','run_provenance.json'):shutil.copy2(run/name,dest/name)
        assert sha(dest/'candidate.onnx')==provenance['onnx_sha256']
    snapshot=OUT/'source_snapshot';snapshot.mkdir(exist_ok=True);source_records=[]
    for i,path in enumerate(sorted(sources)):
        if path.suffix=='.py':ast.parse(path.read_text(encoding='utf-8-sig'))
        dest=snapshot/f'{i}_{path.name}';shutil.copy2(path,dest)
        source_records.append(dict(original=str(path),snapshot=str(dest.relative_to(OUT)),sha256=sha(path)))
    arrays=0
    for folder in ('collection','evaluation','rough_probe','offline_probe/evaluation'):
        for path in (OUT/folder).rglob('*.npz'):
            with np.load(path) as z:
                for key in z.files:assert np.isfinite(z[key]).all();arrays+=1
    assert all(json.loads((OUT/f'evaluation/{a}/complete.json').read_text())['trials']==54 for a in ('control','corrective'))
    files={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'}
    result=dict(status='PASS',new_formal_trials=117,collection_trials=27,posthoc_development_trials=39,fresh_supplemental_trials=13,
        partial_confirmation_trials=json.loads((OUT/'evaluation/fitted/stopped.json').read_text())['completed_trials'],finite_trace_arrays=arrays,sources=source_records,
        files=files,production_sha256=sha(Path('D:/microduck_rl/microdinosaur_p2.onnx')))
    assert result['production_sha256']=='0804114efd1e457ec2c297d709c0464fd0c2bfe251cbb36db7de74543857b0e3'
    (OUT/'artifact_manifest.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print('PASS',len(files),'artifacts,',len(sources),'source files,',arrays,'finite trace arrays')


if __name__=='__main__':main()
