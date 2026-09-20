"""Freeze this bounded study after verifying evidence, without promoting skills."""
from pathlib import Path
import ast
import hashlib
import json
import re
import runpy
import shutil
import time

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'20260914_contact_motion'
OLD=ROOT/'20260914_squat_specialist'
RL=Path('D:/microduck_rl')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,data):p.write_text(json.dumps(data,indent=2),encoding='utf-8')
def verify(mapping,base=None):
    for name,expected in mapping.items():
        path=base/name if base else Path(name)
        assert sha(path)==expected,('Changed frozen file',str(path))

# Prior evidence and deliverables stay immutable.
old=read(OLD/'manifest.json')
verify(old['files'],OLD);verify(old['external_files'])
protected=read(OLD/'protected_files.json')
verify(protected)
write(OUT/'protected_files.json',protected)
write(OUT/'prior_study_integrity.json',dict(files=len(old['files']),
    external_files=len(old['external_files']),manifest_sha256=sha(OLD/'manifest.json'),passed=True))

# Tests target true support and geometry, and the previously discovered dead zone.
ns=runpy.run_path(str(RL/'tests/test_contact_motion.py'))
passed=[]
for name,fn in ns.items():
    if name.startswith('test_') and callable(fn):fn();passed.append(name)
assert len(passed)==3
write(OUT/'reward_gate_tests.json',dict(passed=passed,test_source_sha256=sha(RL/'tests/test_contact_motion.py')))

runs=[
 ('run','20260914_normal005_train_512x301','contact_run_train.log',301),
 ('run','20260914_sole_lift_train_512x201','contact_run_sole_train.log',201),
 ('getup','20260914_cadfiltered_train_512x401','contact_getup_train.log',162),
 ('getup','20260914_dense_train_512x201','contact_getup_dense_train.log',201),
]
audit=[];external=dict(old['external_files'])
for skill,name,logname,count in runs:
    path=RL/'logs/rsl_rl'/('microdinosaur_contact_'+skill)/name
    p=read(path/'run_provenance.json')
    for i,(source,expected) in enumerate(p['sources'].items()):
        assert sha(path/'source_snapshot'/f'{i}_{Path(source).name}')==expected,source
    assert sha(Path(p['source_checkpoint']))==p['source_sha256']
    text=(ROOT/logname).read_text(encoding='utf-8',errors='replace')
    iterations=[int(x) for x in re.findall(r'Learning iteration (\d+)/',text)]
    assert len(iterations)==count,(name,len(iterations),count)
    assert iterations==list(range(iterations[0],iterations[0]+count))
    penalties={}
    for key in ('dof_pos_limits','action_rate_l2','body_ang_vel','foot_slip'):
        values=[float(x) for x in re.findall(r'Episode_Reward/'+key+r':\s*([-\d.e+]+)',text)]
        if values:
            assert max(values)<=1e-12,(name,key,max(values))
            penalties[key]=dict(samples=len(values),min=min(values),max=max(values))
    assert p['envs']==512 and p['learning_rate']==1e-4
    assert p['physics_dt']==.00125 and p['policy_dt']==.02
    final=Path(p.get('final_checkpoint',p.get('resume_checkpoint')))
    if 'final_sha256' in p:assert sha(final)==p['final_sha256']
    if 'onnx_sha256' in p:assert sha(path/'candidate.onnx')==p['onnx_sha256']
    audit.append(dict(run=str(path),status=p['status'],logged_updates=count,
        first_iteration=iterations[0],last_iteration=iterations[-1],transitions=count*512*24,
        retained_checkpoint=str(final),penalty_signs=penalties,source_snapshots_verified=len(p['sources'])))

# Preserve all bounded formal and smoke runs, including stopped diagnostics.
for skill in ('run','getup'):
    for path in sorted((RL/'logs/rsl_rl'/('microdinosaur_contact_'+skill)).glob('20260914_*')):
        if not (path/'run_provenance.json').exists():continue
        dest=OUT/'training_runs'/skill/path.name
        dest.mkdir(parents=True,exist_ok=True)
        for f in path.rglob('*'):
            if not f.is_file():continue
            if f.suffix=='.pt':
                external[str(f)]=sha(f)
            else:
                rel=f.relative_to(path);target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(f,target)
        p=read(path/'run_provenance.json')
        external[p['source_checkpoint']]=p['source_sha256']
        for i,(source,expected) in enumerate(p['sources'].items()):
            assert sha(path/'source_snapshot'/f'{i}_{Path(source).name}')==expected,(path,source)
total=sum(r['transitions'] for r in audit)
assert total==10629120
write(OUT/'training_audit.json',dict(formal_runs=audit,total_logged_transitions=total,
    diagnostic_smokes_excluded=True,stopped_branch_updates_not_in_resume_checkpoint=11))
for name in ('run_export_parity.json','run_sole_export_parity.json','getup_export_parity.json'):
    parity=read(OUT/name)
    assert parity['samples']==117 and parity['max_absolute_error']<1e-5
    assert parity['official_normalizer_verified']

# Legacy evaluations stored the compiled model hash but not the runtime override.
write(OUT/'evaluation_contracts.json',dict(
    legacy_default='Before finalization, evaluate_contact_motion.py used torsion .015 m unless --baseline-friction was passed.',
    matrices={
        'run_final':dict(torsion_m=.015,plant='normal_contact_plant',command_ms=10),
        'run_sole_final':dict(torsion_m=.015,plant='normal_contact_plant',command_ms=10),
        'run_sole_original_torsion':dict(torsion_m=.010,plant='normal_contact_plant',command_ms=10),
        'getup_dense_final':dict(torsion_m=.015,plant='normal_contact_plant',command_ms=10),
        'getup_preferred_contact':dict(torsion_m=.010,plant='preferred_plant',command_ms=10)},
    voltage=12.,position_feedback_ms=20,velocity_feedback_ms=20,physics_dt=.00125,policy_dt=.02,
    squat='Each scan plan.json and case contains the original plant SHA and explicit normal/friction override.',
    current_evaluator='Default now follows selected_contact.json; explicit --torsional-friction remains available; writes full hardware_case.',
    getup='Physical push occurs only before policy switch. Sensor and action history preserved. Final 2 s must be >=95% stable.',
    geometry='CAD transforms depend on identical body/mesh/joint arrays, not runtime friction.',
    release='No production promotion or hardware deployment'))
gates=read(OUT/'final_geometry_gates.json')
assert gates['preferred_squat_cad']['passed'] and gates['run_sole_final_cad']['passed']
assert not gates['getup_dense_final_cad']['passed']
write(OUT/'skill_release.json',dict(
    squat=dict(status='BOUNDED_SHALLOW_SQUAT_SIMULATION_ACCEPTED',depth_mm=25,passed=11,total=11,
        semantics='Reference plus .015*tanh(raw ONNX) bounded leg residual; head IMU outer control',
        complete_fold=False,physical_release=False),
    run=dict(status='FAST_WALK_ONLY_FAILED_RUNNING_AND_STOP_GATES',passed=0,total=9,production_promoted=False),
    getup=dict(status='INVALID_SELF_INTERSECTION_AND_FAILED_STABLE_RECOVERY',passed=0,total=12,
        next_required='Implement and validate head/neck/battery and other observed self contacts before further skill qualification',
        production_promoted=False)))

# Images were inspected during the final review; their metadata was measured.
videos=read(OUT/'video_audit.json')
for v in videos:v['visual_review']=True
write(OUT/'video_audit.json',videos)

logs=OUT/'logs';logs.mkdir(exist_ok=True)
for f in ROOT.glob('contact*.log'):shutil.copy2(f,logs/f.name)

# Capture local source dependencies without modifying historical snapshots.
names=['contact_motion_scan','prepare_contact_motion','prepare_normal_contact','select_preferred_contact',
    'contact_motion_cfg','build_contact_motion_bank','build_getup_curriculum','build_safe_getup_bank',
    'filter_recovery_bank','train_contact_motion','evaluate_contact_motion','export_contact_checkpoint',
    'check_contact_export','prepare_contact_cad','scan_recovery_head','audit_contact_motion',
    'check_contact_turning','render_contact_squat','contact_motion_report','hardware_sim',
    'try_squat_reference','evaluate_run_jump','check_squat_cad','extract_squat_pairs',
    'measure_squat_intersections','audit_contact_results','finalize_contact_motion']
todo=[ROOT/(n+'.py') for n in names];seen=set()
while todo:
    f=todo.pop()
    if f in seen or not f.exists():continue
    seen.add(f);tree=ast.parse(f.read_text(encoding='utf-8-sig'),filename=str(f))
    for node in ast.walk(tree):
        if isinstance(node,ast.ImportFrom) and node.module:
            todo.append(ROOT/(node.module.split('.')[0]+'.py'))
        elif isinstance(node,ast.Import):
            todo.extend(ROOT/(alias.name.split('.')[0]+'.py') for alias in node.names)
snap=OUT/'final_source_snapshot';snap.mkdir(exist_ok=True)
sources={}
for f in sorted(seen):
    shutil.copy2(f,snap/f.name);sources[str(f)]=sha(f)
for f in (RL/'src/mjlab_microduck/tasks/mdp.py',RL/'tests/test_contact_motion.py',ROOT.parent/'RESEARCH_NOTES.md'):
    shutil.copy2(f,snap/f.name);sources[str(f)]=sha(f)
write(OUT/'final_sources.json',sources)
verify(protected);verify(old['files'],OLD);verify(old['external_files'])
files={f.relative_to(OUT).as_posix():sha(f) for f in sorted(OUT.rglob('*')) if f.is_file() and f.name!='manifest.json'}
manifest=dict(status='COMPLETE_BOUNDED_SIMULATION_STUDY',created_unix=time.time(),
    production_promoted=False,all_requested_skills_mastered=False,files=files,external_files=external)
write(OUT/'manifest.json',manifest)
verify(files,OUT);verify(external)
print(json.dumps(dict(status=manifest['status'],files=len(files),external_files=len(external),
    formal_transitions=total,tests=passed,protected_unchanged=True,prior_study_files=len(old['files']))))
