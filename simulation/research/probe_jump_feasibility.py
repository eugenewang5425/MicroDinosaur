"""Bounded servo extension probes, not a learned jump or hardware guarantee."""
from collections import deque
import json
from pathlib import Path
import mujoco
import numpy as np
from mjlab_microduck.s288_protocol import ROTOR_POSITION_OUTPUT_STEP

ROOT = Path(__file__).parent
OUT = ROOT/'20260914_terrain_skills'


def main():
    model = mujoco.MjModel.from_binary_path('contacts.mjb', assets={'contacts.mjb':
        (OUT/'planned_motion_contacts/candidate_ground_contacts.mjb').read_bytes()})
    contract = json.loads((ROOT/'20260913_handoff/native_v07/contract.json').read_text())
    poses = json.loads((OUT/'crouch_feasibility.json').read_text())['results']
    crouch = np.array(poses[2]['target_joint_angles']); home = np.array(contract['action_offset'][0])
    names = contract['action_names']; jids = [model.joint('robot/'+n).id for n in names]
    jadr = model.jnt_qposadr[jids]; vadr = model.jnt_dofadr[jids]
    aids = [int(np.where(model.actuator_trnid[:, 0] == j)[0][0]) for j in jids]
    terrain = model.body('terrain').id
    results = []
    for duration in (.08, .15, .30):
        data = mujoco.MjData(model); data.qpos[2] = .097182; data.qpos[3:7] = [1, 0, 0, 0]
        data.qpos[jadr] = crouch; data.ctrl[aids] = crouch
        mujoco.mj_forward(model, data)
        queue = deque([crouch.copy() for _ in range(8)])
        transmitted = crouch.copy(); applied = crouch.copy(); records = []
        longest_flight = flight = 0.; min_contact_height = np.inf
        for step in range(4800):
            time = step*.00125
            if step%16 == 0:
                blend = np.clip((time-3.)/duration, 0, 1)
                target = crouch+(home-crouch)*blend
                # Original external EMA and slew, plus explicit finite servo
                # target speed for this scripted feasibility probe.
                smoothed = applied+.9*(target-applied)
                applied += np.clip(smoothed-applied, -.16, .16)
                transmitted = np.round(applied/ROTOR_POSITION_OUTPUT_STEP)*ROTOR_POSITION_OUTPUT_STEP
            queue.append(transmitted.copy()); data.ctrl[aids] = queue.popleft()
            mujoco.mj_step(model, data)
            if time < 3.:
                continue
            total_force = 0.; force = np.zeros(6)
            for k, contact in enumerate(data.contact):
                if terrain not in (model.geom_bodyid[contact.geom1], model.geom_bodyid[contact.geom2]):
                    continue
                mujoco.mj_contactForce(model, data, k, force); total_force += max(0., force[0])
            flight = flight+.00125 if total_force < .05 else 0.
            longest_flight = max(longest_flight, flight)
            records.append([time-3., data.qpos[2], data.qvel[2], total_force,
                float(np.max(abs(data.actuator_force[aids]))), float(np.max(abs(data.qvel[vadr])))])
        a = np.array(records)
        rise = float(np.max(a[:, 1])-a[0, 1])
        results.append(dict(extension_time_s=duration, maximum_all_contact_flight_s=longest_flight,
            body_height_rise_mm=rise*1000, max_upward_velocity_m_s=float(np.max(a[:, 2])),
            peak_servo_torque_nm=float(np.max(a[:, 4])), peak_joint_speed_rad_s=float(np.max(a[:, 5])),
            final_root_height_m=float(a[-1, 1]), demonstrated_hop=bool(longest_flight>=.02 and np.max(a[:, 2])>.15)))
        np.savez_compressed(OUT/f'jump_probe_{int(duration*1000)}ms.npz', trace=a)
    report = dict(results=results, trained=False, hardware_verified=False,
        command_delay_ms=10, physics_dt_s=.00125, controller_dt_s=.02, nominal_kp=7, nominal_kd=.8,
        interpretation='A positive result is only feasibility in the provisional torque-limited model; a negative result does not prove every RL jump impossible')
    (OUT/'jump_feasibility.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
