"""Archive completed feet-first experiment without changing any earlier study."""
import ast
import json
from pathlib import Path
import shutil
from evaluate_policy import sha
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'20260914_foot_flight'
RUN=Path('D:/microduck_rl/logs/rsl_rl/microdinosaur_foot_flight/20260914_train_512x401')


def main():
    provenance=json.loads((RUN/'run_provenance.json').read_text());assert provenance['status']=='COMPLETE'
    assert json.loads((OUT/'training_audit.json').read_text())['passed']
    records={label:json.loads((OUT/'evaluation'/label/'matrix.json').read_text()) for label in ['previous','candidate']}
    assert all(len(rows)==48 for rows in records.values())
    for filename in ['run_provenance.json','candidate.onnx','model_17050.pt']:
        target=OUT/filename
        if target.exists():assert sha(target)==sha(RUN/filename)
        else:shutil.copy2(RUN/filename,target)
    snapshot=OUT/'training_source_snapshot'
    if not snapshot.exists():shutil.copytree(RUN/'source_snapshot',snapshot)
    todo=[ROOT/name for name in ['foot_flight_cfg.py','train_terrain_skill.py','test_foot_flight.py','audit_foot_flight_gpu.py',
        'audit_foot_training.py','probe_tuck_jump.py','probe_policy_tuck.py','probe_synchronized_jump.py','probe_air_tuck_ik.py',
        'evaluate_foot_flight.py','summarize_foot_flight.py','render_foot_flight_pair.py','diagnose_foot_flight.py','freeze_foot_flight.py']]
    visited=set();snapshot=OUT/'source_snapshot';snapshot.mkdir(exist_ok=True)
    while todo:
        path=todo.pop()
        if path in visited:continue
        visited.add(path);shutil.copy2(path,snapshot/path.name)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            modules=[node.module] if isinstance(node,ast.ImportFrom) else ([n.name for n in node.names] if isinstance(node,ast.Import) else [])
            for module in modules:
                if module:
                    local=ROOT/(module.split('.')[0]+'.py')
                    if local.exists() and local not in visited:todo.append(local)
    sources={
        'production':Path('D:/microduck_rl/microdinosaur_p2.onnx'),
        'source_checkpoint':Path(provenance['source_checkpoint']),
        'previous_onnx':ROOT/'20260914_jump_refine/candidate.onnx',
        'v7':ROOT/'20260913_handoff/v7_reference.onnx',
        'plant':ROOT/'20260914_run_jump/plant/nominal.mjb'}
    dependencies={key:dict(path=str(path),sha256=sha(path)) for key,path in sources.items()}
    assert dependencies['plant']['sha256']=='02b4bde647eee347105916e49a9d1992689c2bed0894a2d0dc84c28501d164a2'
    assert dependencies['production']['sha256']=='0804114efd1e457ec2c297d709c0464fd0c2bfe251cbb36db7de74543857b0e3'
    script_rows=[r for folder in ['script_probes','script_timed','hybrid_probes','synchronized_probes','air_ik_probes','air_ik_target_probes']
        for r in json.loads((OUT/folder/'matrix.json').read_text())]
    manifest=dict(status='COMPLETE',promoted=False,formal_planned=96,formal_executed=sum(r['status']=='COMPLETE' for rows in records.values() for r in rows),
        scripted_probe_count=len(script_rows),scripted_feet_goal_count=sum(r.get('metrics',{}).get('feet_goal',False) for r in script_rows),
        interim_cpu_probes=4,dependencies=dependencies,files={str(path.relative_to(OUT)):sha(path) for path in sorted(OUT.rglob('*'))
            if path.is_file() and path.name not in ['manifest.json','freeze.log'] and '__pycache__' not in path.parts})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['files','dependencies']},indent=2))


if __name__=='__main__':main()
