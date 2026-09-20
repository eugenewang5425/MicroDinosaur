"""Compile and check the actual specialist plant before launching PPO."""
from collections import deque
import json
import time
import numpy as np
import mujoco
from mjlab.scene import Scene
from run_jump_cfg import ROOT,OUT,build_config,BANK,PROXIES,XML
from evaluate_policy import sha
from mjlab_microduck.s288_protocol import ROTOR_POSITION_OUTPUT_STEP


def main():
    dest=OUT/'plant';dest.mkdir(parents=True,exist_ok=False)
    _,cfg=build_config('jump',1)
    scene=Scene(cfg.env.scene,device='cpu')
    ref=mujoco.MjModel.from_binary_path('nominal.mjb',assets={'nominal.mjb':(ROOT/'20260913_handoff/native_v07/nominal.mjb').read_bytes()})
    for key in ('timestep','integrator','solver','iterations','ls_iterations','cone','impratio','tolerance','ls_tolerance'):
        setattr(scene.spec.option,key,getattr(ref.opt,key))
    scene.spec.option.timestep=.00125;scene.spec.option.iterations=30;scene.spec.option.ls_iterations=50
    m=scene.compile();d=mujoco.MjData(m)
    contract=json.loads((ROOT/'20260913_handoff/native_v07/contract.json').read_text())
    contract.update(physics_dt=.00125,nominal_kp=7,nominal_kd=.8,skill='run_jump',
        floor_proxy_source_sha256=sha(PROXIES),note='Conservative v07 visual AABB floor proxies, not complete self collision')
    names=contract['action_names'];ids=[m.joint('robot/'+n).id for n in names]
    qa=m.jnt_qposadr[ids];va=m.jnt_dofadr[ids]
    aids=[int(np.where(m.actuator_trnid[:,0]==j)[0][0]) for j in ids]
    mass=float(sum(m.body_mass[i] for i in range(m.nbody) if m.body(i).name.startswith('robot/')))
    assert abs(mass-1.0982148)<1e-7 and m.nu==19
    assert np.allclose(m.actuator_forcerange[aids],[-.6,.6])
    buffer=np.empty(mujoco.mj_sizeModel(m),np.uint8);mujoco.mj_saveModel(m,buffer=buffer)
    (dest/'nominal.mjb').write_bytes(buffer.tobytes())
    (dest/'contract.json').write_text(json.dumps(contract,indent=2),encoding='utf-8')
    bank=json.loads(BANK.read_text())['records'];rows=[]
    floor=m.geom('terrain').id;body=m.body('robot/trunk_base').id
    for seed in range(3):
        b=bank[seed%len(bank)]
        mujoco.mj_resetData(m,d);d.qpos[:7]=b['root_qpos'];d.qvel[:6]=b['root_qvel']
        d.qpos[qa]=b['joint_pos'];d.qvel[va]=b['joint_vel'];d.ctrl[aids]=b['applied']
        mujoco.mj_forward(m,d)
        seen=set();peak=0.
        for _ in range(2400):
            mujoco.mj_step(m,d);peak=max(peak,float(abs(d.actuator_force[aids]).max()))
            for c in d.contact:
                if floor in (c.geom1,c.geom2) and c.dist<=0:
                    name=m.geom(c.geom2 if c.geom1==floor else c.geom1).name
                    if name.endswith('_floor_proxy'):seen.add(name)
        tilt=float(np.degrees(np.arccos(np.clip(d.xmat[body].reshape(3,3)[2,2],-1,1))))
        rows.append(dict(seed=seed,hold_s=3.,tilt_deg=tilt,root_height_m=float(d.qpos[2]),
                         nonfoot_contacts=sorted(seen),peak_torque_nm=peak,finite=bool(np.isfinite(d.qpos).all())))
    report=dict(robot_mass_kg=mass,actuator_count=m.nu,xml_sha256=sha(XML),plant_sha256=sha(dest/'nominal.mjb'),
                standing=rows,ready=all(r['finite'] and r['tilt_deg']<15 and not r['nonfoot_contacts'] for r in rows),
                old_jump_probe='Three prior 2cm crouch extensions had zero flight; no proof all learned hops are impossible',
                no_hardware_validation=True,production_sha_before=sha('D:/microduck_rl/microdinosaur_p2.onnx'))
    (OUT/'physics_precheck.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    assert report['ready']


if __name__=='__main__':main()
