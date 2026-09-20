"""Archive completed evidence, exact training sources and local evaluation dependencies."""
import ast
import json
from pathlib import Path
import shutil
from evaluate_policy import sha

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'20260914_jump_refine'
RUN=Path('D:/microduck_rl/logs/rsl_rl/microdinosaur_jump_refine/20260914_train_512x401')


def main():
    provenance=json.loads((RUN/'run_provenance.json').read_text());assert provenance['status']=='COMPLETE'
    audit=json.loads((OUT/'training_audit.json').read_text());assert audit['passed']
    records={label:json.loads((OUT/'evaluation'/label/'matrix.json').read_text()) for label in ['source','candidate']}
    assert all(len(rows)==48 for rows in records.values())
    assert {r['name'] for r in records['source']}=={r['name'] for r in records['candidate']}
    assert [(r['name'],r.get('error')) for r in records['source'] if r['status']=='FAILED']==[
        (r['name'],r.get('error')) for r in records['candidate'] if r['status']=='FAILED']
    for filename in ['run_provenance.json','candidate.onnx','model_17050.pt']:
        source=RUN/filename;target=OUT/filename
        if target.exists():assert sha(target)==sha(source)
        else:shutil.copy2(source,target)
    target=OUT/'training_source_snapshot'
    if not target.exists():shutil.copytree(RUN/'source_snapshot',target)
    todo=[ROOT/name for name in ['train_terrain_skill.py','jump_refine_cfg.py','test_jump_refine.py',
        'audit_jump_refine_gpu.py','audit_jump_refine_training.py','evaluate_jump_refine.py',
        'probe_jump_refine_gpu.py','summarize_jump_refine.py','render_jump_refine_pair.py','freeze_jump_refine.py']]
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
    dependencies={
        'production':Path('D:/microduck_rl/microdinosaur_p2.onnx'),
        'source_checkpoint':Path(provenance['source_checkpoint']),
        'source_onnx':Path(records['source'][0]['policy']),
        'v7_return':ROOT/'20260913_handoff/v7_reference.onnx',
        'plant':ROOT/'20260914_run_jump/plant/nominal.mjb',
        'xml':Path('D:/microduck_rl/src/mjlab_microduck/robot/microdinosaur_v07/robot_microdinosaur_v07.xml')}
    hashes={key:dict(path=str(path),sha256=sha(path)) for key,path in dependencies.items()}
    assert hashes['production']['sha256']=='0804114efd1e457ec2c297d709c0464fd0c2bfe251cbb36db7de74543857b0e3'
    assert hashes['plant']['sha256']=='02b4bde647eee347105916e49a9d1992689c2bed0894a2d0dc84c28501d164a2'
    assert hashes['source_checkpoint']['sha256']==provenance['source_sha256']
    assert hashes['xml']['sha256']==provenance['xml_sha256']
    manifest=dict(status='COMPLETE',promoted=False,formal_evaluation_planned=96,
        formal_evaluation_completed=sum(r['status']=='COMPLETE' for rows in records.values() for r in rows),
        interim_cpu_probes=8,replay_captures_are_exact_repeats_not_new_independent_seeds=True,
        dependencies=hashes,artifacts={str(path.relative_to(OUT)):sha(path) for path in sorted(OUT.rglob('*'))
            if path.is_file() and path.name not in ['manifest.json','freeze.log'] and '__pycache__' not in path.parts})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['artifacts','dependencies']},indent=2))


if __name__=='__main__':main()
