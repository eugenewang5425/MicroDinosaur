"""Frozen previous policy: compare policy-offset head control and IMU ownership."""
import json
from pathlib import Path
import numpy as np
from dataclasses import replace
from imu_owned_head import configure_owned
from terrain_skill_eval import TerrainSkillExperiment
from hardware_sim import HardwareCase
from evaluate_policy import sha

ROOT=Path(__file__).parent;OUT=ROOT/'20260914_imu_owned_terrain'
PREVIOUS=Path('D:/microduck_rl/logs/rsl_rl/microdinosaur_transition_refine/20260914_train_512x201/candidate.onnx')
CHECKPOINT=PREVIOUS.parent/'model_16150.pt'


def main():
    dest=OUT/'ownership_probe';dest.mkdir(parents=True,exist_ok=False)
    cases=[('flat','stand','none',10),('flat','straight','none',10),('flat','stand','crouch40',10),
        ('steps_10','straight','none',10)]
    records=[]
    jobs=[(*case,mode) for case in cases for mode in ('policy','owned')]
    jobs += [('flat',scenario,'none',delay,'owned') for delay in (5,15) for scenario in ('stand','straight')]
    jobs += [('flat',scenario,'none',10,'owned') for scenario in ('left_then_hold','right_then_hold')]
    plan=dict(policy=str(PREVIOUS),policy_sha256=sha(PREVIOUS),jobs=jobs,
        selection_rule='Reject IMU ownership if nominal flat falls, speed loss >5%, or camera direction worsens. Inspect positive-delay/step/turn probes; do not tune gains after results.')
    (dest/'plan.json').write_text(json.dumps(plan,indent=2))
    for terrain,scenario,posture,delay,mode in jobs:
        e=TerrainSkillExperiment(PREVIOUS,terrain=terrain,posture=posture,
            hardware_case=HardwareCase(physics_dt=.00125,command_ms=delay))
        if mode=='owned':
            configure_owned(e)
            # Preserve the original three-axis probe's recorded configuration.
            # All owned baselines are overwritten by measured joints each step.
            e.head_config=replace(e.head_config,nominal_target_filter_tau_s=0.)
            e.head_controller.config=e.head_config
        m,t=e.run(scenario,'imu',1,12.,True)
        name=f'{mode}_{terrain}_{scenario}_{posture}_lag{delay}'
        r=dict(name=name,mode=mode,terrain=terrain,scenario=scenario,posture=posture,motor_delay_ms=delay,
            seed=1,metrics=m,policy_sha256=sha(PREVIOUS))
        (dest/(name+'.json')).write_text(json.dumps(r,indent=2));records.append(r)
        np.savez_compressed(dest/(name+'.npz'),trace=t,gaze=np.asarray(e.sim.gaze_trace),
            head_controls=np.asarray(e.head_rows),head_physics=np.asarray(e.head_physics),commands=np.asarray(e.command_rows))
        print(json.dumps(dict(name=name,fell=m['fell'],vx=m['body_vx_mean_m_s'],
            yaw=m['camera_heading_error_rms_deg'],pitch=m['camera_pitch_rms_deg'],rate=m['camera_yaw_rate_error_rms_deg_s'])),flush=True)
    (dest/'matrix.json').write_text(json.dumps(records,indent=2))


if __name__=='__main__':main()
