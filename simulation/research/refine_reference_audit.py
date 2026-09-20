"""Frozen-policy startup probes and a sensor-calibrated reset bank.

Changing a reward reference cannot change a frozen actor. The reference
comparison is therefore a diagnostic score, not a claimed policy ablation.
"""
import json
from pathlib import Path
import numpy as np
from terrain_skill_eval import TerrainSkillExperiment
from imu_heading import matrix, from_rpy, wrap
from evaluate_policy import sha
from head_attitude import rotation_vector

ROOT = Path(__file__).parent
OUT = ROOT/'20260914_transition_refine'
V7 = ROOT/'20260913_handoff/v7_reference.onnx'
DEEP = Path('D:/microduck_rl/logs/rsl_rl/microdinosaur_deep_crouch/20260914_train_512x301/candidate.onnx')


def single_packet_restart(e):
    """Reproduce the old GPU sensor initializer, with the plant untouched."""
    source, values = e.stream.latest
    accel = values[3].copy()
    e.head_est.bias[:] = 0.
    e.head_est.accel_filtered = accel
    x, y, z = accel
    e.head_est.q = from_rpy(np.arctan2(y, z), np.arctan2(-x, np.hypot(y, z)))
    e.head_est.last_time = source; e.head_est.yaw = e.head_est.previous_yaw = 0.
    x, y, z = values[1]
    body = matrix(from_rpy(np.arctan2(y, z), np.arctan2(-x, np.hypot(y, z))))
    # Read the same delayed encoder history that the next actor observation uses.
    s = e.sim
    measured = s.obs_history[max(0, len(s.obs_history)-s.obs_lag)][6:25]+s.home
    relative, _ = e.kinematics.forward(measured[e.chain_indices])
    expected = body@relative; independent = matrix(e.head_est.q)
    e.head_alignment = wrap(np.arctan2(expected[1,0],expected[0,0])-np.arctan2(independent[1,0],independent[0,0]))
    e.alignment_rotation = matrix(from_rpy(0., 0., e.head_alignment))


class StartupProbe(TerrainSkillExperiment):
    def __init__(self, startup, posture, scenario):
        super().__init__(DEEP, posture=posture)
        self.startup = startup; self.restart_done = False; self.reference_rows = []
        original = self.transform_target
        def transform(target, obs):
            if self.active:
                t = self.sim.data.time-6.
                trigger = 0. if startup == 'single_still' else 3.
                if startup != 'calibrated' and not self.restart_done and t+1e-9 >= trigger:
                    single_packet_restart(self); self.restart_done = True
                head = self.sim.data.site_xmat[self.sim.model.site('robot/head_imu').id].reshape(3,3)
                body = self.sim.data.site_xmat[self.sim.model.site('robot/imu').id].reshape(3,3)
                body_yaw = np.arctan2(body[1,0],body[0,0])
                desired = matrix(from_rpy(0,0,self.controller.reference+self.world_heading_zero))
                old = matrix(from_rpy(0,0,body_yaw))
                self.reference_rows.append([t,*rotation_vector(head.T@desired),*rotation_vector(head.T@old),
                    wrap(body_yaw-self.world_heading_zero-self.controller.reference)])
            return original(target, obs)
        self.sim.transform_target = transform

    def reset(self, seed=0):
        result = super().reset(seed)
        self.restart_done = False; self.reference_rows = []
        body = self.sim.data.site_xmat[self.sim.model.site('robot/imu').id].reshape(3,3)
        self.world_heading_zero = np.arctan2(body[1,0],body[0,0])
        return result


def build_bank():
    records = []
    for seed in (1,2,3):
        e = TerrainSkillExperiment(V7); calibration = e.reset(seed); s = e.sim
        body = s.data.site_xmat[s.model.site('robot/imu').id].reshape(3,3)
        source, sensor = e.stream.latest
        records.append(dict(seed=seed, root_qpos=s.data.qpos[:7].tolist(), root_qvel=s.data.qvel[:6].tolist(),
            joint_pos=s.data.qpos[s.jadr].tolist(), joint_vel=s.data.qvel[s.vadr].tolist(),
            raw_action=s.raw.tolist(), applied=s.applied.tolist(),
            head_rotation=matrix(e.head_est.q).tolist(), head_accel=e.head_est.accel_filtered.tolist(),
            head_bias=e.head_est.bias.tolist(), alignment=e.alignment_rotation.tolist(),
            packet=np.r_[sensor[2],sensor[3],sensor[1]].tolist(),
            heading_world_zero=float(np.arctan2(body[1,0],body[0,0])), calibration=calibration))
    result = dict(source_policy=str(V7), source_sha256=sha(V7), calibration_seconds=6.,
        sample_count=50, motor_delay_ms=10, position_delay_ms=20, physics_dt=.00125,
        action_names=s.names, records=records,
        contract='Sample calibrated nominal standing snapshots; domain randomization acts after reset. This is not a claim of per-randomized-plant online calibration.')
    (OUT/'calibrated_reset_bank.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


def main():
    OUT.mkdir(exist_ok=False)
    build_bank()
    dest = OUT/'reference_audit'; dest.mkdir()
    records = []
    for scenario, posture in (('stand','none'),('stand','crouch40'),('straight','none')):
        for startup in ('calibrated','single_still','single_moving'):
            for seed in (1,2,3):
                e = StartupProbe(startup, posture, scenario)
                metrics, trace = e.run(scenario,'imu',seed,12.,True)
                refs = np.asarray(e.reference_rows)
                name = f'{posture}_{scenario}_{startup}_s{seed}'
                record = dict(name=name, startup=startup, scenario=scenario, posture=posture,seed=seed,
                    metrics=metrics, policy_sha256=sha(DEEP),
                    reference_yaw_disagreement_rms_deg=float(np.rad2deg(np.sqrt(np.mean(refs[:,7]**2)))),
                    navigation_attitude_error_rms_deg=np.rad2deg(np.sqrt(np.mean(refs[:,1:4]**2,axis=0))).tolist(),
                    body_heading_attitude_error_rms_deg=np.rad2deg(np.sqrt(np.mean(refs[:,4:7]**2,axis=0))).tolist())
                records.append(record)
                (dest/(name+'.json')).write_text(json.dumps(record,indent=2),encoding='utf-8')
                np.savez_compressed(dest/(name+'.npz'),reference=refs,trace=trace,gaze=np.asarray(e.sim.gaze_trace))
                print(json.dumps(dict(name=name, fell=metrics['fell'], pitch=metrics['camera_pitch_rms_deg'],
                    yaw=metrics['camera_heading_error_rms_deg'],vx=metrics['body_vx_mean_m_s'])),flush=True)
    (dest/'matrix.json').write_text(json.dumps(records,indent=2),encoding='utf-8')


if __name__ == '__main__': main()
