"""Follow-up to full head ownership: isolate yaw, retain learned pitch/roll baseline."""
import json
import numpy as np
from imu_owned_head import configure_owned
from terrain_skill_eval import TerrainSkillExperiment
from hardware_sim import HardwareCase
from probe_head_ownership import OUT,PREVIOUS
from evaluate_policy import sha


def main():
    dest=OUT/'yaw_probe';dest.mkdir(exist_ok=False)
    cases=[('flat','stand','none',10),('flat','straight','none',10),('flat','stand','crouch40',10),('steps_10','straight','none',10)]
    cases += [('flat',scenario,'none',delay) for delay in (5,15) for scenario in ('stand','straight')]
    cases += [('flat',scenario,'none',10) for scenario in ('left_then_hold','right_then_hold')]
    (dest/'plan.json').write_text(json.dumps(dict(cases=cases,reason='Full ownership improves yaw but nominal walking pitch RMS rises 1.46 to 3.74 degrees. Isolate yaw ownership before training.',
        gates='No flat falls; retain >=95% nominal speed; improve nominal yaw by >=30%; nominal pitch RMS <= previous*1.1; no gain tuning.'),indent=2))
    records=[]
    for terrain,scenario,posture,delay in cases:
        e=configure_owned(TerrainSkillExperiment(PREVIOUS,terrain=terrain,posture=posture,
            hardware_case=HardwareCase(physics_dt=.00125,command_ms=delay)),(1,))
        m,t=e.run(scenario,'imu',1,12.,True);name=f'yaw_{terrain}_{scenario}_{posture}_lag{delay}'
        r=dict(name=name,mode='yaw_owned',terrain=terrain,scenario=scenario,posture=posture,motor_delay_ms=delay,
            seed=1,metrics=m,policy_sha256=sha(PREVIOUS));records.append(r)
        (dest/(name+'.json')).write_text(json.dumps(r,indent=2))
        np.savez_compressed(dest/(name+'.npz'),trace=t,gaze=np.asarray(e.sim.gaze_trace),
            head_controls=np.asarray(e.head_rows),head_physics=np.asarray(e.head_physics),commands=np.asarray(e.command_rows))
        print(json.dumps(dict(name=name,fell=m['fell'],vx=m['body_vx_mean_m_s'],yaw=m['camera_heading_error_rms_deg'],
            pitch=m['camera_pitch_rms_deg'],rate=m['camera_yaw_rate_error_rms_deg_s'])),flush=True)
    (dest/'matrix.json').write_text(json.dumps(records,indent=2))


if __name__=='__main__':main()
