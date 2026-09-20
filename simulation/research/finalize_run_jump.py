"""Freeze research sources, verify completed evidence and write an artifact manifest."""
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'20260914_run_jump'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    counts={};records=[]
    for label,expected in [('source',36),('source_stress',16),('jump',16),
                           ('jump_return',16),('jump_return_fresh',9),('run',34)]:
        rows=json.loads((OUT/'evaluation'/label/'matrix.json').read_text())
        assert len(rows)==expected,(label,len(rows))
        for r in rows:
            if r['status']=='COMPLETE':
                assert (OUT/'evaluation'/label/(r['name']+'.npz')).is_file()
                if label=='jump_return':assert r['prefix_max_error']<1e-9
                if label.startswith('jump_return'):
                    h=r['handoff'];assert h['history_before_sha256']==h['history_after_sha256']
            else:
                assert r['status']=='FAILED' and 'Calibration angular motion is excessive' in r['error']
        counts[label]=dict(planned=len(rows),completed=sum(r['status']=='COMPLETE' for r in rows))
        records.extend(rows)
    assert 'Ran 7 tests' in (OUT/'tests.log').read_text() and '\nOK' in (OUT/'tests.log').read_text()
    pre=json.loads((OUT/'physics_precheck.json').read_text())
    assert pre['ready'] and sha(OUT/'plant/nominal.mjb')==pre['plant_sha256']
    production=Path('D:/microduck_rl/microdinosaur_p2.onnx')
    assert sha(production)==pre['production_sha_before']
    trained=json.loads((OUT/'training_audit.json').read_text());external={}
    for skill,audit in trained.items():
        run=Path(audit['run']);provenance=run/'run_provenance.json'
        p=json.loads(provenance.read_text());assert p['status']=='COMPLETE'
        for key,value in p['source_hashes'].items():
            assert sha(key)==value,('training dependency drift',key)
            external[key]=value
        assert sha(p['source_checkpoint'])==p['source_sha256']
        assert sha(p['final_checkpoint'])==audit['checkpoint_sha256']
        assert sha(p['onnx'])==audit['onnx_sha256']
        for path in [provenance,Path(p['source_checkpoint']),Path(p['final_checkpoint']),Path(p['onnx']),
                     *sorted((run/'params').glob('*')),*sorted((run/'source_snapshot').glob('*'))]:
            if path.is_file():external[str(path)]=sha(path)
    external[str(production)]=sha(production)
    return_policy=ROOT/'20260913_handoff/v7_reference.onnx'
    external[str(return_policy)]=sha(return_policy)

    names=['run_jump_cfg','prepare_run_jump','test_run_jump','train_terrain_skill',
           'evaluate_run_jump','evaluate_jump_return','summarize_run_jump','render_run_jump_saved',
           'finalize_run_jump']
    pending=[ROOT/(n+'.py') for n in names];sources=set()
    while pending:
        path=pending.pop()
        if path in sources:continue
        code=path.read_text(encoding='utf-8-sig');tree=ast.parse(code,filename=str(path))
        compile(tree,str(path),'exec');sources.add(path)
        for node in ast.walk(tree):
            imports=[]
            if isinstance(node,ast.Import):imports=[n.name for n in node.names]
            elif isinstance(node,ast.ImportFrom) and node.module:imports=[node.module]
            for name in imports:
                local=ROOT/(name.split('.')[0]+'.py')
                if local.is_file() and local not in sources:pending.append(local)
    snap=OUT/'source_snapshot';snap.mkdir(exist_ok=True)
    for path in sorted(sources):
        shutil.copy2(path,snap/path.name);external[str(path)]=sha(path)
    for path in [ROOT.parent/'RESEARCH_NOTES.md',ROOT.parent/'README.md']:
        external[str(path)]=sha(path)

    versions={}
    for name in ['torch','mujoco','mujoco-warp','mjlab','rsl-rl-lib','onnxruntime','numpy','imageio','imageio-ffmpeg']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]='not found under this distribution name'
    manifest=dict(status='COMPLETE_RESEARCH_CANDIDATES_NOT_PROMOTED',python=sys.version,
        interpreter=sys.executable,versions=versions,counts=counts,
        planned_confirmations=len(records),completed_motion_records=sum(r['status']=='COMPLETE' for r in records),
        training_transitions=sum(a['transitions'] for a in trained.values()),production_unchanged=True,
        training_dependencies_unchanged_since_launch=True,local_source_modules_snapshotted=len(sources),
        runtime_snapshot_note='Current local Python import closure plus immutable training snapshots. Installed dependencies are versioned, not vendored.',
        external_sha256=external,
        artifact_sha256={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*'))
                        if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({k:manifest[k] for k in ['status','counts','planned_confirmations','completed_motion_records',
             'training_transitions','production_unchanged','training_dependencies_unchanged_since_launch',
             'local_source_modules_snapshotted','versions']},indent=2))


if __name__=='__main__':main()
