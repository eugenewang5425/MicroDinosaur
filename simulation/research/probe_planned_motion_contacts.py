"""Current v07 ground-contact proxy audit before prone/recovery skill training.

Collision proxies are fit to current visual mesh bounds. Only floor contact is
added; this is explicitly not a complete self-collision/assembly certificate.
"""
import json
from pathlib import Path
import mujoco
import numpy as np
from mjlab.scene import Scene
from terrain_skill_cfg import build_config
from imu_heading import matrix, from_rpy
from evaluate_policy import sha

ROOT = Path(__file__).parent
OUT = ROOT/'20260914_terrain_skills/planned_motion_contacts'


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    _, cfg = build_config('crouch', 1)
    scene = Scene(cfg.env.scene, device='cpu')
    original = scene.compile()
    robot_ids = [i for i in range(original.nbody) if original.body(i).name.startswith('robot/')]
    proxies = []
    for body_id in robot_ids:
        name = original.body(body_id).name
        if name.endswith(('ankle_left', 'ankle_right')):
            continue
        points = []
        for gid in range(original.ngeom):
            if original.geom_bodyid[gid] != body_id or original.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH or original.geom_contype[gid] != 0:
                continue
            mesh = original.geom_dataid[gid]; start = original.mesh_vertadr[mesh]; count = original.mesh_vertnum[mesh]
            v = original.mesh_vert[start:start+count]
            points.append(v@matrix(original.geom_quat[gid]).T+original.geom_pos[gid])
        if not points:
            continue
        p = np.vstack(points); lo = p.min(0); hi = p.max(0)
        center = (lo+hi)/2; half = np.maximum((hi-lo)/2, .001)
        scene.spec.body(name).add_geom(name=name+'_floor_proxy', type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=center, size=half, contype=4, conaffinity=8, group=3, rgba=[.2, .6, .5, .25])
        proxies.append(dict(body=name, center=center.tolist(), half_size=half.tolist(), source='Current v07 visual mesh bounds'))
    for g in scene.spec.geoms:
        if g.name.endswith('_floor_proxy'):
            continue
        if g.name == 'terrain':
            g.contype = 8; g.conaffinity = 2|4
        elif g.contype or g.conaffinity:
            g.contype = 2; g.conaffinity = 2|8
    scene.spec.option.timestep = .00125; scene.spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    scene.spec.option.iterations = 30; scene.spec.option.ls_iterations = 50
    model = scene.compile()
    np.testing.assert_allclose(original.body_mass[robot_ids], model.body_mass[robot_ids], atol=1e-10)
    contract = json.loads((ROOT/'20260913_handoff/native_v07/contract.json').read_text())
    home = np.array(contract['action_offset'][0]); names = contract['action_names']
    jids = [model.joint('robot/'+n).id for n in names]
    jadr = model.jnt_qposadr[jids]
    aids = [int(np.where(model.actuator_trnid[:, 0] == j)[0][0]) for j in jids]
    data = mujoco.MjData(model); cases = []
    for label, quat in [('stand', from_rpy(0, 0)), ('prone', from_rpy(0, np.pi/2)),
                        ('supine', from_rpy(0, -np.pi/2)), ('left_side', from_rpy(np.pi/2, 0)),
                        ('right_side', from_rpy(-np.pi/2, 0))]:
        mujoco.mj_resetData(model, data); data.qpos[2] = .117182; data.qpos[3:7] = quat
        data.qpos[jadr] = home; data.ctrl[aids] = home
        mujoco.mj_forward(model, data)
        if label != 'stand':
            lowest = np.inf
            for gid in range(model.ngeom):
                if model.geom_bodyid[gid] in robot_ids and model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX:
                    r = data.geom_xmat[gid].reshape(3, 3)
                    lowest = min(lowest, data.geom_xpos[gid, 2]-abs(r[2])@model.geom_size[gid])
            data.qpos[2] += .005-lowest; mujoco.mj_forward(model, data)
        contacts = set(); max_torque = 0.; min_root = np.inf
        for _ in range(2400):
            mujoco.mj_step(model, data)
            max_torque = max(max_torque, float(abs(data.actuator_force[aids]).max()))
            min_root = min(min_root, float(data.qpos[2]))
            for contact in data.contact:
                if contact.dist <= .001:
                    for gid in (contact.geom1, contact.geom2):
                        name = model.geom(gid).name
                        if name.endswith('_floor_proxy'):
                            contacts.add(name)
        mujoco.mj_forward(model, data)
        r = data.xmat[model.body('robot/trunk_base').id].reshape(3, 3)
        cases.append(dict(pose=label, final_root_height_m=float(data.qpos[2]), minimum_root_height_m=min_root,
            final_body_tilt_deg=float(np.rad2deg(np.arccos(np.clip(r[2, 2], -1, 1)))),
            finite=bool(np.isfinite(data.qpos).all()), peak_torque_nm=max_torque, contacted_proxies=sorted(contacts)))
    buf = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8); mujoco.mj_saveModel(model, buffer=buf)
    (OUT/'candidate_ground_contacts.mjb').write_bytes(buf.tobytes())
    report = dict(current_xml_sha256=sha('D:/microduck_rl/src/mjlab_microduck/robot/microdinosaur_v07/robot_microdinosaur_v07.xml'),
        proxy_count=len(proxies), robot_mass_kg=float(model.body_mass[robot_ids].sum()), proxies=proxies, cases=cases,
        full_self_collision_verified=False, motions_trained=False,
        next_gate='Check conservative box fit and collision pairs before training prone/jump/get-up; define head-loop disable/reacquire during recovery')
    (OUT/'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('proxy_count', 'robot_mass_kg', 'cases')}, indent=2))


if __name__ == '__main__': main()
