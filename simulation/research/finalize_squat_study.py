"""Freeze measured outcomes and the specialist's distinct execution contract."""
from pathlib import Path
import json,re,shutil,time
import numpy as np
import imageio.v2 as imageio
from evaluate_policy import sha

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'20260914_squat_specialist'
RUNS=Path('D:/microduck_rl/logs/rsl_rl/microdinosaur_squat_specialist')
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
geometry=read(OUT/'solid_intersections.json')
base={tuple(v['parts']):v['intersection_mm3'] for v in geometry[0]['pairs']}
new=[dict(frame=f['frame'],parts=v['parts'],increase_mm3=v['intersection_mm3']-base[tuple(v['parts'])])
     for f in geometry[1:] for v in f['pairs']]
maximum=max(new,key=lambda r:r['increase_mm3'])
assert len(geometry)==81 and maximum['increase_mm3']<.001
gate=dict(status='BOUNDED_LEARNED_SAMPLES_PASS',frames=81,pairs_per_frame=len(base),
    maximum_new_intersection=maximum,numerical_volume_tolerance_mm3=.001,
    baseline_vendor_internal_overlap_mm3=max(base.values()),full_joint_range_released=False,
    upper_lower_jaw_dynamic_self_contact_complete=False,learned_trajectory_audit_pending=False)
(OUT/'geometry_gate.json').write_text(json.dumps(gate,indent=2))

names=['evaluation_ppo201','evaluation_imitation','evaluation_zero','evaluation_residual','confirmation_residual']
comparison={}
for name in names:
    rows=read(OUT/name/'summary.json');executed=[r for r in rows if r['status']=='COMPLETE']
    comparison[name]=dict(planned=len(rows),executed=len(executed),passed=sum(r['passed'] for r in rows),
        calibration_rejections=sum(r['status']=='CALIBRATION_REJECTED' for r in rows),
        actual_depth_mm=[r['actual_depth_mm'] for r in executed],
        mean_depth_error_mm=float(np.mean([abs(r['actual_depth_mm']-25) for r in executed])),
        mean_hold_tilt_max_deg=float(np.mean([r['hold_tilt_max_deg'] for r in executed])),
        mean_hold_contact_speed_p99_mm_s=float(np.mean([r['hold_loaded_contact_tangent_speed_p99_mm_s'] for r in executed])))
assert comparison['evaluation_residual']['passed']==7 and comparison['confirmation_residual']['passed']==3
(OUT/'comparison.json').write_text(json.dumps(comparison,indent=2))

stages=[]
for folder,log,count,envs in [
    ('20260914_smoke_64x5','smoke.log',5,64),
    ('20260914_train_512x201','train.log',201,512),
    ('20260914_residual_smoke_64x5','residual_smoke.log',5,64),
    ('20260914_residual_train_512x101','residual_train.log',101,512)]:
    run=RUNS/folder;record=read(run/'run_provenance.json');assert record['status']=='COMPLETE'
    for i,(source,expected) in enumerate(record['source_hashes'].items()):
        assert sha(run/'source_snapshot'/f'{i}_{Path(source).name}')==expected
    text=(OUT/log).read_text(errors='replace')
    iterations=re.findall(r'Learning iteration (\d+)/',text);assert len(iterations)==count
    for term in ('body_ang_vel','dof_pos_limits','action_rate_l2','foot_slip','translation'):
        values=[float(v) for v in re.findall(r'Episode_Reward/'+term+r':\s*([-+0-9.eE]+)',text)]
        assert values and max(values)<=0
    stages.append(dict(run=str(run),updates=count,envs=envs,transitions=count*envs*24,
        penalty_signs_valid=True,source_snapshots_verified=True,official_export=True))
(OUT/'training_audit.json').write_text(json.dumps(dict(stages=stages,formal_rl_transitions=(201+101)*512*24,
    supervised_gradient_steps=1500,physical_demonstration_frames=5850,
    unit_checks='Two direct runpy assertion tests passed; pytest not installed',
    normalizer_export=read(OUT/'residual_export_audit.json')),indent=2))

final=read(RUNS/'20260914_residual_train_512x101/run_provenance.json')
contract=read(OUT/'plant/contract.json')
contract.update(kind='Independent squat reference plus bounded residual',actor_dim=81,action_dim=19,
    onnx=final['onnx'],onnx_sha256=final['onnx_sha256'],plant_sha256=sha(OUT/'plant/nominal.mjb'),
    outputs='Residual values, not the production position-action contract',
    commands='All locomotion/gesture commands zero; body_pose Z (command[9], actor[72]) in [-0.025,0] m; jaw command +0.04 rad',
    trajectory='Caller supplies a 3s smooth ramp down, 3s hold, 3s up, 3s stable check; never jump directly to the target',
    target='Calibrated standing applied targets + (-command_z/0.025)*25mm reference leg delta + 0.015*tanh(raw) on each leg joint',
    head='Existing 20ms head IMU loop; yaw encoder baseline; head gyro not added to actor',
    angle_envelope='Ankle target +/-51deg; measured task envelope <55deg; jaw target +0.04rad; global physical limits unchanged',
    switch='After verified stationary calibration, preserve applied targets, sensor and command delay histories; set residual last_action to zero',
    deployment_status='Simulation research only; production policy unchanged; real friction and S288 motoring curve not identified')
(OUT/'residual_skill_contract.json').write_text(json.dumps(contract,indent=2),encoding='utf-8')

protected={
    ROOT.parent/'design_source/current/MicroDinosaur_v1.blender':'0a4f86aefea24e0d5260c837849a16164ec83cc2d364f58e6c03947de95ce286',
    Path('D:/microduck_rl/microdinosaur_p2.onnx'):'0804114efd1e457ec2c297d709c0464fd0c2bfe251cbb36db7de74543857b0e3',
    Path('D:/microduck_rl/src/mjlab_microduck/robot/microdinosaur_v07/robot_microdinosaur_v07.xml'):'e33fa22db6eb35c9f6d9d076267ae46e4bd7916cc83a63cf2b82ee63ed2e721c'}
for path,expected in protected.items():assert sha(path)==expected,(path,sha(path))
(OUT/'protected_files.json').write_text(json.dumps({str(p):v for p,v in protected.items()},indent=2))
reader=imageio.get_reader(OUT/'squat_comparison.mp4');meta=reader.get_meta_data();frames=reader.count_frames();reader.close()
assert frames==325 and meta['fps']==25
(OUT/'video_audit.json').write_text(json.dumps(dict(frames=frames,fps=25,seconds=13,
    size=meta['size'],sha256=sha(OUT/'squat_comparison.mp4'),hold_frame_visually_reviewed=True),indent=2))
print(json.dumps(dict(status='COMPLETE',comparison=comparison,geometry=gate,formal_transitions=(201+101)*512*24),indent=2))
