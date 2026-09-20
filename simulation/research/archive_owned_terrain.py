"""Freeze final research sources and validate artifacts against recorded inputs."""
import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'20260914_imu_owned_terrain'
RL=Path('D:/microduck_rl')
RUN=RL/'logs/rsl_rl/microdinosaur_imu_owned_terrain/20260914_resume_512x100'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    provenance=read(RUN/'run_provenance.json');source_files={Path(p) for p in provenance['source_hashes']}
    for p,digest in provenance['source_hashes'].items():assert sha(Path(p))==digest,p
    all_names=None;arrays_checked=0
    for label in ('frozen_yaw','trained_yaw'):
        records=read(OUT/f'evaluation/{label}/matrix.json');assert len(records)==45
        names={r['name'] for r in records};assert len(names)==45
        if all_names is None:all_names=names
        else:assert names==all_names
        expected=provenance['onnx_sha256'] if label=='trained_yaw' else '96c995ebd175cd5d8f04da38ebfd80aaed017249005c831ad4b8be9828a0aaf2'
        for r in records:
            assert r['policy_sha256']==expected and r['status']=='COMPLETE'
            assert r==read(OUT/f"evaluation/{label}/{r['name']}.json")
            for name,digest in r['source_hashes'].items():
                path=ROOT/name;assert sha(path)==digest,(label,name);source_files.add(path)
            with np.load(OUT/f"evaluation/{label}/{r['name']}.npz") as z:
                for name in z.files:
                    assert np.isfinite(z[name]).all(),(label,r['name'],name)
                    arrays_checked+=1
    for name in ('audit_lane_live.py','audit_owned_failure.py','audit_owned_training.py','archive_owned_terrain.py',
        'probe_head_ownership.py','probe_yaw_ownership.py','summarize_head_ownership.py',
        'summarize_owned_terrain.py','render_owned_terrain.py','join_owned_terrain_video.py','test_imu_owned_terrain.py'):
        source_files.add(ROOT/name)
    dest=OUT/'source_snapshot';dest.mkdir(exist_ok=False);manifest=[]
    for path in sorted(source_files):
        if path.is_relative_to(RL):relative=Path('rl')/path.relative_to(RL)
        elif path.is_relative_to(ROOT):relative=Path('research')/path.relative_to(ROOT)
        else:raise AssertionError(path)
        target=dest/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        digest=sha(path);assert sha(target)==digest
        manifest.append(dict(original=str(path),snapshot=str(target.relative_to(OUT)),sha256=digest))
    old=RL/'logs/rsl_rl/microdinosaur_transition_refine/20260914_train_512x201/source_snapshot/2_mdp.py'
    current=RL/'src/mjlab_microduck/tasks/mdp.py'
    assert old.exists()
    diff=''.join(difflib.unified_diff(old.read_text(encoding='utf-8').splitlines(keepends=True),
        current.read_text(encoding='utf-8').splitlines(keepends=True),fromfile='previous_run/mdp.py',tofile='current_run/mdp.py'))
    (OUT/'mdp_changes.diff').write_text(diff,encoding='utf-8')
    for name in ('candidate.onnx','run_provenance.json'):
        target=OUT/'final_model'/name;target.parent.mkdir(exist_ok=True);shutil.copy2(RUN/name,target)
    assert sha(OUT/'final_model/candidate.onnx')==provenance['onnx_sha256']
    assert sha(RL/'microdinosaur_p2.onnx')=='0804114efd1e457ec2c297d709c0464fd0c2bfe251cbb36db7de74543857b0e3'
    # Validate current report links; historical notes deliberately retain history.
    for report in (OUT/'RESULTS.md',OUT/'CONTROL_FINDINGS.md'):
        for target in re.findall(r'\]\(([^)]+)\)',report.read_text(encoding='utf-8')):
            if not target.startswith(('http://','https://','#')):
                assert (report.parent/target).exists(),(report,target)
    result=dict(status='PASS',matched_trials=90,finite_trace_arrays_checked=arrays_checked,
        source_files=manifest,report_links_valid=True,production_unchanged=True,
        artifacts={str(p.relative_to(OUT)):dict(bytes=p.stat().st_size,sha256=sha(p))
            for p in sorted(OUT.rglob('*')) if p.is_file() and p.suffix!='.log' and p.name!='artifact_manifest.json'})
    (OUT/'artifact_manifest.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:(len(v) if isinstance(v,(dict,list)) else v) for k,v in result.items()},indent=2))


if __name__=='__main__':main()
