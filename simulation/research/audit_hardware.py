"""Verify the selected CAD/BOM, exported policy contract and completed run."""
import hashlib
import json
from pathlib import Path
import subprocess

import mujoco
import numpy as np
import torch
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from mjlab_microduck.s288_protocol import (
    GEAR_RATIO, OUTPUT_ENCODER_STEP, ROTOR_POSITION_OUTPUT_STEP, VELOCITY_STEP,
    KP_STEP, KD_STEP, JY61P_GYRO_STEP, round_scalar, wire_time_seconds,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'research/20260913_hardware_physics'
RUN = Path('D:/microduck_rl/logs/rsl_rl/microdinosaur_v07_calibration/20260913_s288_v7recipe_4096x101')
SOURCE = Path('D:/microduck_rl/logs/rsl_rl/velocity_microdinosaur/2026-09-13_13-46-57_velocity_microdinosaur')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def differences(a, b, path=''):
    result=[]
    if isinstance(a,dict) and isinstance(b,dict):
        for k in sorted(set(a)|set(b)):
            result.extend(differences(a.get(k),b.get(k),f'{path}.{k}'))
    elif a!=b:
        result.append({'path':path,'v7':a,'candidate':b})
    return result


def main():
    cad=ROOT/'design_source'
    ledger=json.loads((cad/'current/mass_estimate.json').read_text())
    rows=ledger['rows']
    motor_rows=[r for r in rows if r['name'].startswith('S288_') and r['name'].endswith('_CASE')]
    assert len(rows)==672 and len(motor_rows)==19
    assert np.isclose(sum(r['mass_estimate_g'] for r in motor_rows),370.5)
    model_path=ROOT/'research/20260913_handoff/native_v07/nominal.mjb'
    model=mujoco.MjModel.from_binary_path('nominal.mjb',assets={'nominal.mjb':model_path.read_bytes()})
    assert abs(model.body_mass.sum()*1000-ledger['mass_estimate_g'])<.001
    active_geoms=[{'name':mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i),
        'contype':int(model.geom_contype[i]),'conaffinity':int(model.geom_conaffinity[i]),
        'condim':int(model.geom_condim[i]),'friction':model.geom_friction[i].tolist()}
        for i in range(model.ngeom) if model.geom_contype[i] or model.geom_conaffinity[i]]
    buses=json.loads((cad/'current/s288_joint_bus_map.json').read_text())
    bus_counts={bus:sum(r['bus']==bus for r in buses) for bus in ('A','B')}
    assert bus_counts=={'A':10,'B':9}
    provenance=json.loads((RUN/'run_provenance.json').read_text())
    assert provenance['status']=='COMPLETE' and provenance['finite_output_check']
    source=torch.load(SOURCE/'model_15500.pt',map_location='cpu',weights_only=False)
    final=torch.load(RUN/'model_15600.pt',map_location='cpu',weights_only=False)
    counters=[d['infos']['env_state']['common_step_counter'] for d in (source,final)]
    assert counters[1]-counters[0]==101*24
    ea=EventAccumulator(str(RUN),size_guidance={'scalars':0});ea.Reload()
    scalars={tag:ea.Scalars(tag) for tag in ea.Tags()['scalars']}
    assert all(np.isfinite(e.value) for entries in scalars.values() for e in entries)
    parameter_diff=[]
    reward_match=False
    for filename in ('env.yaml','agent.yaml'):
        a=yaml.load((SOURCE/'params'/filename).read_text(),Loader=yaml.BaseLoader)
        b=yaml.load((RUN/'params'/filename).read_text(),Loader=yaml.BaseLoader)
        parameter_diff.extend(differences(a,b,filename))
        if filename=='env.yaml':
            reward_match=a['rewards']==b['rewards']
            # BaseLoader compares saved parameters; callable implementations
            # are deliberately audited separately in source snapshots.
            assert reward_match
    (OUT/'training_parameter_diff.json').write_text(json.dumps(parameter_diff,indent=2),encoding='utf-8')
    sources={
        'S288_product':'https://www.unitree.com/cn/mobile/DigitalServo/',
        'S288_protocol':'https://github.com/unitreerobotics/digital_servo/blob/main/specs/protocol.md',
        'S288_example':'https://github.com/unitreerobotics/digital_servo/blob/main/python/servo_demo.py',
        'JY61P_standard_reference':'https://www.witmotion.cn/proztmz/37.html',
        'MuJoCo_accelerometer':'https://mujoco.readthedocs.io/en/stable/XMLreference.html#sensor-accelerometer',
    }
    selected_rows=[r for r in rows if any(k in r['name'].upper() for k in ('BATTERY','PI_ZERO','F411','POLOLU','JY61P'))]
    record={
        'cad_commit':subprocess.check_output(['git','-C',str(cad),'rev-parse','HEAD'],text=True).strip(),
        'cad_git_status':subprocess.check_output(['git','-C',str(cad),'status','--porcelain'],text=True).strip(),
        'cad_sha256':sha(cad/'current/MicroDinosaur_v1.blender'),
        'ledger_sha256':sha(cad/'current/mass_estimate.json'),
        'plant_sha256':sha(model_path), 'ledger_rows':len(rows),
        'cad_mass_g':ledger['mass_estimate_g'],'plant_mass_g':float(model.body_mass.sum()*1000),
        'cad_com_mm':ledger['com_mm'],'s288_count':len(motor_rows),'s288_mass_g':370.5,
        'selected_component_ledger_rows':selected_rows,
        'bus_counts':bus_counts,'wire_only_cycle_ms':{b:wire_time_seconds(n)*1000 for b,n in bus_counts.items()},
        'active_contact_geoms':active_geoms,
        'physics':{'dt':float(model.opt.timestep),'iterations':int(model.opt.iterations),
            'integrator':str(mujoco.mjtIntegrator(int(model.opt.integrator))),
            'solver':str(mujoco.mjtSolver(int(model.opt.solver)))},
        'protocol':{'gear_ratio':GEAR_RATIO,'encoder_step_rad':OUTPUT_ENCODER_STEP,
            'command_step_rad':ROTOR_POSITION_OUTPUT_STEP,'velocity_step_rad_s':VELOCITY_STEP,
            'kp_step':KP_STEP,'kd_step':KD_STEP,'rounded_nominal_kp':round_scalar(7,KP_STEP),
            'rounded_nominal_kd':round_scalar(.8,KD_STEP),'gyro_step_deg_s':float(np.rad2deg(JY61P_GYRO_STEP))},
        'training':{'run':str(RUN),'complete':True,'counter_before_after':counters,
            'updates':101,'num_envs':4096,'new_environment_transitions':101*24*4096,
            'all_tensorboard_scalars_finite':True,'scalar_tags':len(scalars),
            'saved_reward_parameters_match_v7':reward_match,
            'saved_parameter_difference_count':len(parameter_diff),
            'source_implementations_identical_to_v7':False,
            'friction_randomization_in_candidate':'Inherited BAM-only event; ineffective for S288',
            'post_candidate_fix':'Current s288_profile uses joint_friction; separately validated, not in candidate',
            'head_gyro_actor_input':False,'reward_history_HOME_and_arm_mirror_fixes_retained':True,
            'final_checkpoint_sha256':sha(RUN/'model_15600.pt'),'onnx_sha256':sha(RUN/'candidate.onnx'),
            'source_snapshot_matches_current':{Path(r['path']).name:sha(r['path'])==r['sha256'] for r in provenance['source_files']}},
        'sources':sources,
        'measured_mass_inertia_motor_curve_electrical_thermal':False,
    }
    assert record['cad_git_status']==''
    (OUT/'hardware_audit.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    sidecar={
        'onnx_sha256':record['training']['onnx_sha256'],
        'plant_sha256':record['plant_sha256'],'actor_input':81,'output':19,
        'normalizer_in_onnx':True,'external_filter_and_hardware_profile_in_onnx':False,
        'host_action_filter':{'alpha_new':.9,'max_delta_rad_per_20ms':.6,'state_reset':'absolute HOME'},
        'profile':'s288-protocol','implementation':str(ROOT/'research/hardware_sim.py'),
        'protocol':record['protocol'],
        'training_delay_ms':{'motor':[5,15],'position':[20,40],'velocity':[20,20]},
        'evaluation_default_delay_ms':{'motor':10,'position':20,'velocity':20},
        'motor_curve_in_training':False,'thermal_or_electrical_model_in_training':False,
        'friction_randomization_in_training':'BAM-only no-op for S288; fixed after this checkpoint',
        'hardware_deployment_validated':False,
    }
    (RUN/'candidate_contract.json').write_text(json.dumps(sidecar,indent=2),encoding='utf-8')
    (OUT/'candidate_contract.json').write_text(json.dumps(sidecar,indent=2),encoding='utf-8')
    print(json.dumps({k:record[k] for k in ('cad_mass_g','plant_mass_g','active_contact_geoms','physics','protocol','training')},indent=2))


if __name__=='__main__':main()
