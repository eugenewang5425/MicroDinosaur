"""Freeze diagnostic evidence and verify old artifacts were not rewritten."""
from pathlib import Path
import sys,json,shutil,hashlib
import numpy as np
import imageio.v2 as imageio
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'20260914_compact_fold'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks={
    'cad':(ROOT.parent/'design_source/current/MicroDinosaur_v1.blender','0a4f86aefea24e0d5260c837849a16164ec83cc2d364f58e6c03947de95ce286'),
    'production':(Path('D:/microduck_rl/microdinosaur_p2.onnx'),'0804114efd1e457ec2c297d709c0464fd0c2bfe251cbb36db7de74543857b0e3'),
    'v7':(ROOT/'20260913_handoff/v7_reference.onnx','53120a401d02f124b35361d791eb74748a7601550c701c511d62cbad9cddff96'),
    'old_plant':(ROOT/'20260914_run_jump/plant/nominal.mjb','02b4bde647eee347105916e49a9d1992689c2bed0894a2d0dc84c28501d164a2')}
dependencies={}
for name,(path,expected) in checks.items():
    actual=sha(path);assert actual==expected,(name,actual)
    dependencies[name]=dict(path=str(path),sha256=actual)
verified={}
for study in ('20260914_foot_flight','20260914_jump_dynamics_review'):
    folder=ROOT/study;manifest=json.loads((folder/'manifest.json').read_text())
    for name,expected in manifest['files'].items():
        digest=expected if isinstance(expected,str) else expected['sha256']
        assert sha(folder/name)==digest,(study,name)
    verified[study]=len(manifest['files'])
from compact_contact_cfg import build_config
try:build_config()
except RuntimeError as exc:
    assert 'interference gate failed' in str(exc)
else:raise AssertionError('Diagnostic plant must not silently train')
records=json.loads((OUT/'floor_only_nominal/summary.json').read_text())
assert len(records)==3 and all(r['passed'] for r in records)
for path in (OUT/'floor_only_nominal').glob('*.npz'):
    a=np.load(path)
    assert a['trace'].shape==(10400,12)
    assert a['torque'].shape==(10400,19)
    assert a['qpos'].shape[0]==325
    assert all(np.isfinite(a[k]).all() for k in ['trace','torque','joint_velocity','qpos'])
reader=imageio.get_reader(str(OUT/'compact_fold_trial.mp4'))
media=reader.get_meta_data();count=reader.count_frames();reader.close()
assert count==325 and media['fps']==25
assert not (OUT/'jaw_hulls.json').exists(),'An unexpected unvalidated mouth candidate appeared'
sources=['extract_compact_contacts.py','extract_compact_pairs.py','cad_contact_geometry.py',
    'decompose_compact_contacts.py','compact_contact_cfg.py','try_compact_fold.py',
    'check_compact_cad.py','prepare_compact_cad.py','measure_compact_intersections.py',
    'scan_compact_joints.py','render_compact_fold.py','render_compact_interference.py',
    'hardware_sim.py','evaluate_policy.py','head_attitude_sim.py','imu_owned_head.py',
    'run_heading_stable_start.py','heading_sim.py','finalize_compact_fold.py']
snapshot=OUT/'source_snapshot';snapshot.mkdir(exist_ok=True)
for name in sources:shutil.copy2(ROOT/name,snapshot/name)
status=dict(status='COMPLETE_DIAGNOSTIC',overall_feasibility_passed=False,
    dynamic_trials=3,dynamic_gate_passed=3,cad_geometry_gate_passed=False,
    new_training_updates=0,running_training_started=False,recovery_training_started=False,
    new_policy_promoted=False,body_ground_contacts_fixed_in_diagnostic_plant=True,
    mouth_self_contact_fixed=False,mouth_decomposition_complete=False,
    mouth_decomposition_note='Initial triangle-soup input rejected; two interrupted attempts; indexed attempt exited 1 after lower jaw completed. No full hull output.',
    body_floor_geometry_flat_plane_only=True,verified_prior_archives=verified,
    dependencies=dependencies,video=dict(frames=count,fps=25,seconds=count/25),
    training_default_rejected=True,next_required='Resolve CAD assembly/rigid-body assignment and effective self contacts before skill training.')
(OUT/'status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
files={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()
       and 'tools' not in p.relative_to(OUT).parts and '__pycache__' not in p.parts
       and p.name!='manifest.json'}
(OUT/'manifest.json').write_text(json.dumps(dict(**status,files=files),indent=2),encoding='utf-8')
print(json.dumps(dict(files=len(files),old_archives_verified=verified,
    dynamics_passed=3,geometry_passed=False,video_frames=count,new_training_updates=0)))
