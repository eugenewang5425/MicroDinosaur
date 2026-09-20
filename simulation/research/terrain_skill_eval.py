"""CPU terrain/posture evaluation with both current IMU control loops enabled."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import mujoco
import numpy as np
from mjlab.scene import Scene
from terrain_skill_cfg import build_config
from head_attitude_sim import HeadExperiment
from head_attitude import HeadConfig, HeadController
from hardware_sim import HardwareCase
from heading_sim import WARMUP_SECONDS
from evaluate_policy import sha

ROOT = Path(__file__).parent
OUT = ROOT/'20260914_terrain_skills'


def ground_height(kind, x, y=0.):
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    if kind.startswith('downsteps_'):
        h=float(kind.split('_')[-1])/1000
        return -np.clip(np.floor((x-.3)/.18)+1,0,3)*h
    if kind.startswith('slope'):
        angle = float(kind.split('_')[-1])
        return np.maximum(x-.3, 0.)*np.tan(np.deg2rad(angle))
    if kind.startswith('steps'):
        h = float(kind.split('_')[-1])/1000
        return np.clip(np.floor((x-.3)/.18)+1, 0, 3)*h
    if kind in ('uneven_8mm', 'roughgrid_8mm'):
        ix = np.floor((x-.3)/.25).astype(int)
        iy = np.floor((y+3.)/.3).astype(int)
        # Preserve v1 for its archived records. Multiplying ix by 7 before
        # modulo 7 accidentally made v1 constant along each forward lane.
        h = .002+.001*((ix*(5 if kind == 'roughgrid_8mm' else 7)+iy*3)%7)
        return np.where((ix>=0)&(ix<16)&(iy>=0)&(iy<20), h, 0.)
    return np.zeros_like(x)


def freeze_plant(kind):
    dest = OUT/'plants'/kind
    if (dest/'nominal.mjb').exists():
        return dest
    _, cfg = build_config('crouch', 1)
    scene = Scene(cfg.env.scene, device='cpu')
    reference = mujoco.MjModel.from_binary_path('nominal.mjb', assets={'nominal.mjb': (ROOT/'20260913_handoff/native_v07/nominal.mjb').read_bytes()})
    for name in ('timestep', 'integrator', 'solver', 'iterations', 'ls_iterations', 'cone',
                 'impratio', 'tolerance', 'ls_tolerance', 'noslip_iterations'):
        setattr(scene.spec.option, name, getattr(reference.opt, name))
    scene.spec.option.timestep = .00125
    scene.spec.option.iterations = 30
    scene.spec.option.ls_iterations = 50
    terrain = scene.spec.body('terrain')
    def box(name, pos, size, quat=(1., 0., 0., 0.)):
        terrain.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size, quat=quat,
            contype=1, conaffinity=1, group=0, rgba=[.32, .39, .36, 1.], friction=[1., .005, .0001])
    if kind.startswith('slope'):
        scene.spec.geom('terrain').pos[2] = -.5
        box('start_platform', [-4.85, 0., -.04], [5.15, 3., .04])
        angle = np.deg2rad(float(kind.split('_')[-1]))
        horizontal = 8.
        normal = np.array([-np.sin(angle), 0., np.cos(angle)])
        center = np.array([.3+horizontal/2, 0., horizontal*np.tan(angle)/2])-.04*normal
        box('ramp', center, [horizontal/(2*np.cos(angle)), 3., .04], [np.cos(angle/2), 0., -np.sin(angle/2), 0.])
    elif kind.startswith('downsteps_'):
        height=float(kind.split('_')[-1])/1000
        scene.spec.geom('terrain').pos[2]=-3*height
        # Flat calibrated start remains z=0; descend to the lower base plane.
        for i,(left,right,top) in enumerate([(-10.,.3,0.),(.3,.48,-height),(.48,.66,-2*height)]):
            bottom=-3*height-.08
            box(f'downstep_{i}',[(left+right)/2,0.,(top+bottom)/2],[(right-left)/2,3.,(top-bottom)/2])
    elif kind.startswith('steps'):
        height = float(kind.split('_')[-1])/1000
        for i in range(3):
            left = .3+.18*i; right = left+.18 if i<2 else 8.
            h = (i+1)*height
            box(f'step_{i}', [(left+right)/2, 0., h/2], [(right-left)/2, 3., h/2])
    elif kind in ('uneven_8mm', 'roughgrid_8mm'):
        for i in range(16):
            for j in range(20):
                x = .3+(i+.5)*.25; y = -3+(j+.5)*.3; h = float(ground_height(kind, x, y))
                box(f'cell_{i}_{j}', [x, y, h/2], [.125, .15, h/2])
    model = scene.compile()
    assert model.nu == reference.nu == 19
    robot_ids = [i for i in range(model.nbody) if model.body(i).name.startswith('robot/')]
    np.testing.assert_allclose(model.body_mass[robot_ids], reference.body_mass[robot_ids], atol=1e-10)
    np.testing.assert_allclose(model.actuator_gainprm, reference.actuator_gainprm, atol=.04)
    dest.mkdir(parents=True)
    buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=buffer); (dest/'nominal.mjb').write_bytes(buffer.tobytes())
    contract = json.loads((ROOT/'20260913_handoff/native_v07/contract.json').read_text())
    contract.update(terrain=kind, physics_dt=.00125, scene_solver_iterations=30,
        terrain_start_x=.3, note='Same current v07 robot; deterministic held-out terrain geometry')
    (dest/'contract.json').write_text(json.dumps(contract, indent=2), encoding='utf-8')
    # Verify the analytic scorer against ray intersections in the actual model.
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    mask = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8); tests = []
    for x in (.15, .4, .65, 1.1, 2.3):
        for y in (-.25, .25):
            origin = np.array([x, y, 2.]); geom = np.array([-1], dtype=np.int32)
            distance = mujoco.mj_ray(model, data, origin, np.array([0., 0., -1.]), mask, 1, -1, geom)
            actual = 2-distance; expected = float(ground_height(kind, x, y))
            assert abs(actual-expected)<1e-6, (kind, x, y, actual, expected)
            tests.append([x, y, actual, expected])
    (dest/'ground_ray_audit.json').write_text(json.dumps(tests, indent=2), encoding='utf-8')
    return dest


class TerrainSkillExperiment(HeadExperiment):
    def __init__(self, policy, terrain='flat', posture='none', head_enabled=True, walking_speed=None, hardware_case=None):
        self.terrain_kind = terrain; self.posture = posture; self.height_reference = 0.
        self.walking_speed = walking_speed
        super().__init__(freeze_plant(terrain), policy, hardware_case or HardwareCase(physics_dt=.00125),
            head_enabled=head_enabled, head_config=HeadConfig(max_measurement_age_s=.04))
        def fall(samples):
            local = samples[:, 2]-ground_height(terrain, samples[:, 0], samples[:, 1])
            return bool(np.any(local < .055) or np.any(samples[:, 6] > 60))
        self.sim.fall_test = fall
        original = self.sim.step
        def step(command, *args, **kwargs):
            command = command.copy()
            if self.active:
                if self.walking_speed is not None and abs(command[0])>.001:
                    command[0]=self.walking_speed
                t = self.sim.data.time-WARMUP_SECONDS
                desired = {'none': 0., 'crouch10': -.01, 'crouch20': -.02, 'crouch30': -.03, 'crouch40': -.04}[self.posture]
                if not 2 <= t < 7:
                    desired = 0.
                self.height_reference += float(np.clip(desired-self.height_reference, -.0004, .0004))
                command[9] = self.height_reference
                self.command_rows.append([t, self.height_reference])
            original(command, *args, **kwargs)
        self.sim.step = step
        before_physics = self.sim.before_physics_step
        def force_probe(target):
            result = before_physics(target)
            if self.terrain_kind == 'push_2n' and self.active:
                t = self.sim.data.time-WARMUP_SECONDS
                if 3. <= t < 3.1:
                    self.sim.data.xfrc_applied[self.sim.body, 1] = 2.
            return result
        self.sim.before_physics_step = force_probe

    def reset(self, *args, **kwargs):
        self.height_reference = 0.; self.command_rows = []
        return super().reset(*args, **kwargs)

    def run(self, *args, **kwargs):
        m, a = super().run(*args, **kwargs)
        full = np.asarray(self.sim.gaze_trace)[-round(m['seconds']/self.sim.model.opt.timestep):]
        local = full[:, 3]-ground_height(self.terrain_kind, full[:, 1], full[:, 2])
        t = full[:, 0]-WARMUP_SECONDS
        command = np.asarray(self.command_rows)
        desired = .117182+np.interp(t, command[:, 0], command[:, 1])
        plateau = (t>=4)&(t<6.5); returned = t>=9.
        m.update(terrain=self.terrain_kind, posture=self.posture,
            ground_clearance_min_m=float(np.min(local)),
            terrain_exposure_fraction=float(np.mean(full[:, 1]>.35)),
            forward_displacement_m=float(full[-1, 1]-full[0, 1]),
            height_error_rms_mm=float(np.sqrt(np.mean((local-desired)**2))*1000),
            height_plateau_mean_m=float(np.mean(local[plateau])),
            height_returned_mean_m=float(np.mean(local[returned])) if np.any(returned) else None,
            body_tilt_max_deg=float(np.max(full[:, 14])),
            foot_contact_fraction=full[:, 9:11].mean(0).tolist())
        if self.walking_speed is not None: m['walking_command_m_s']=self.walking_speed
        return m, a


def main():
    p = argparse.ArgumentParser(); p.add_argument('--policy', required=True, type=Path)
    p.add_argument('--label', required=True); p.add_argument('--suite', choices=('terrain', 'crouch', 'crouch_slow', 'stress'), required=True)
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args(); dest = OUT/'evaluation'/args.label/args.suite; dest.mkdir(parents=True, exist_ok=args.resume)
    cases = [('flat', 'none', 'straight'), ('slope_3', 'none', 'straight'), ('slope_-3', 'none', 'straight'),
             ('slope_5', 'none', 'straight'), ('steps_5', 'none', 'straight'), ('steps_10', 'none', 'straight'),
             ('roughgrid_8mm', 'none', 'straight')] if args.suite == 'terrain' else [
             ('flat', 'none', 'stand'), ('flat', 'crouch10', 'stand'), ('flat', 'crouch20', 'stand'),
             ('flat', 'crouch10', 'straight'), ('flat', 'crouch20', 'straight')]
    if args.suite == 'stress':
        cases = [('slope_8', 'none', 'straight'), ('steps_15', 'none', 'straight'), ('push_2n', 'none', 'straight')]
    if args.suite == 'crouch_slow':
        cases = [('flat','crouch10','straight'),('flat','crouch20','straight')]
    records = []
    for terrain, posture, scenario in cases:
        for seed in range(1, args.seeds+1):
            record_path = dest/f'{terrain}__{posture}__{scenario}__s{seed}.json'
            if record_path.exists():
                old = json.loads(record_path.read_text())
                assert args.resume and old['policy_sha256'] == sha(args.policy)
                records.append(old); continue
            e = TerrainSkillExperiment(args.policy, terrain, posture,
                walking_speed=.2 if args.suite=='crouch_slow' else None)
            try:
                m, trace = e.run(scenario, 'imu', seed, 12., True)
                record = dict(status='COMPLETE', metrics=m, policy_sha256=sha(args.policy))
                np.savez_compressed(dest/f'{terrain}__{posture}__{scenario}__s{seed}.npz', trace=trace,
                    gaze=np.array(e.sim.gaze_trace), commands=np.array(e.command_rows))
            except ValueError as exc:
                record = dict(status='FAILED', error=str(exc), terrain=terrain, posture=posture, seed=seed)
            records.append(record)
            (dest/f'{terrain}__{posture}__{scenario}__s{seed}.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
            print(json.dumps(dict(terrain=terrain, posture=posture, seed=seed, status=record['status'],
                fell=record.get('metrics', {}).get('fell'), vx=record.get('metrics', {}).get('body_vx_mean_m_s'))), flush=True)
    (dest/'matrix.json').write_text(json.dumps(records, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
