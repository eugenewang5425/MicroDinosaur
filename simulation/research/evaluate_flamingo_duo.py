"""Flamingo 双姿态 × 双支撑侧 验收 —— 留 margin 的绑定指标。

为什么要按组测: side/pose 在出生时随机, 跟着随机跑就无法保证每组都有样本,
也就给不出"哪一组不达标"的判据。这里把 [side, pose] 四个组合**各自固定**跑一遍。

判据(默认 6s = 300 步 @ 50Hz，由实际控制周期计算):
  1) 不摔倒    tilt_max < 70°            (fell_over 终止线)
  2) 不趴下    躯干最低高 > 60mm          (趴下=失败, 即使没触 70° 线)
  3) 质心在支撑脚上方  CoM-支撑脚水平距离均值 < 40mm   (奖励项 std=30mm)
  4) 支撑脚踩地 支撑脚离地均值 < 20mm 且接触占比 >= 80%
  5) 摆动脚到位 摆动脚高度均值与目标(前伸 50mm / 后折 150mm)偏差 < 40mm

保持率 = 五条全过的 episode 比例。输出每组的 margin(离限值还差多少), 供跨轮 diff。
"""
from pathlib import Path
import sys
import json
import shutil
import argparse
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from flamingo_cfg import build_config, SWING_Z_FORWARD, SWING_Z_FOLDED  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from evaluate_policy import sha  # noqa: E402

DEFAULT_SECONDS = 6.0
CRIT = dict(tilt_max_deg=70.0, root_z_min_mm=60.0, com_dist_mm=40.0,
            stance_z_mm=20.0, stance_contact_frac=.80, swing_z_dev_mm=40.0)
GROUPS = [('右脚支撑/前伸', 1.0, 1.0), ('右脚支撑/后折', 1.0, -1.0),
          ('左脚支撑/前伸', -1.0, 1.0), ('左脚支撑/后折', -1.0, -1.0)]


def ascii_copy(path: Path) -> Path:
    """ONNX Runtime 在中文路径上会失败, 先拷到纯 ASCII 临时路径。"""
    tmp = Path(tempfile.gettempdir()) / f'fl_duo_{path.stem[:24]}.onnx'
    shutil.copy2(path, tmp)
    return tmp


def to_obs_tensor(obs, num_envs):
    o = obs['actor'] if isinstance(obs, dict) else obs
    if not isinstance(o, torch.Tensor):
        o = torch.cat([torch.as_tensor(x, dtype=torch.float32).reshape(num_envs, -1)
                       for x in (o.values() if hasattr(o, 'values') else [o])], dim=-1)
    if o.dim() > 2:
        o = o.flatten(1)
    return o


def run_group(session, onnx_path, side, pose, envs, seed, push=0.0, seconds=DEFAULT_SECONDS):
    import onnxruntime as ort
    task, cfg = build_config(envs=envs, seed=seed, fix_side=side, fix_pose=pose,
                             push=push)
    env = ManagerBasedRlEnv(cfg.env, device='cuda')
    policy_dt = float(cfg.env.sim.mujoco.timestep * cfg.env.decimation)
    steps = round(seconds / policy_dt)
    if steps < 1 or abs(steps * policy_dt - seconds) > 1e-8:
        raise ValueError('Evaluation duration must be a positive multiple of the policy period')
    obs, _ = env.reset()
    robot = env.scene['robot']
    sites = [n.split('/')[-1] for n in robot.site_names]
    root_bid = env.scene['robot'].indexing.root_body_id
    stance_foot = 'right_foot' if side > 0 else 'left_foot'
    swing_foot = 'left_foot' if side > 0 else 'right_foot'
    z_target = SWING_Z_FORWARD if pose > 0 else SWING_Z_FOLDED
    si_st, si_sw = sites.index(stance_foot), sites.index(swing_foot)
    sensor = env.scene.sensors.get('feet_ground_contact')
    st_slot = 1 if side > 0 else 0

    per = [dict(swing_z=[], stance_z=[], com_d=[], tilt=[], root_z=[], contact=[],
                dead_step=None) for _ in range(envs)]
    for step in range(steps):
        with torch.no_grad():
            onx = to_obs_tensor(obs, envs).detach().cpu().numpy().astype(np.float32)
            act = np.zeros((onx.shape[0], 19), np.float32)
            for b in range(onx.shape[0]):
                act[b] = session.run(None, {'obs': onx[b:b + 1]})[0][0]
        obs, _, term, trunc, _ = env.step(torch.as_tensor(act, dtype=torch.float32))
        sp = robot.data.site_pos_w
        com = torch.as_tensor(env.sim.data.subtree_com[:, root_bid, :2])
        # found 是接触点数(0/1/2, 脚跟+脚尖), 不是布尔 —— 实测均值曾给出 137%,
        # 必须 >0 离散化后才能当"接触占比"用
        found = ((sensor.data.found[:, st_slot] > 0).float() if sensor is not None
                 else torch.ones(envs))
        for i in range(envs):
            p = per[i]
            if p['dead_step'] is not None:
                continue
            if bool(term[i]) or (bool(trunc[i]) and step < steps - 1):
                p['dead_step'] = step
                continue  # env.step may already have reset: never score the new episode.
            if bool(trunc[i]):
                continue  # Last-step timeout is completion, but its reset state is not data.
            p['swing_z'].append(float(sp[i, si_sw, 2]))
            p['stance_z'].append(float(sp[i, si_st, 2]))
            p['com_d'].append(float(torch.linalg.norm(com[i] - sp[i, si_st, :2])))
            p['tilt'].append(float(np.degrees(np.arccos(np.clip(
                -float(robot.data.projected_gravity_b[i][2]), -1.0, 1.0)))))
            p['root_z'].append(float(env.sim.data.qpos[i, 2]))
            p['contact'].append(float(found[i]))
    env.close()

    rows = []
    for i, p in enumerate(per):
        if not p['swing_z']:
            rows.append(dict(tilt_max_deg=180., root_z_min_mm=0., com_dist_mean_mm=1e6,
                stance_z_mean_mm=1e6, stance_contact_frac=0., swing_z_mean_mm=0.,
                swing_z_dev_mm=z_target*1000, swing_z_min_mm=0., dead_step=p['dead_step'],
                recorded_samples=0, requested_seconds=seconds, policy_dt=policy_dt))
            continue
        sw = float(np.mean(p['swing_z'])) * 1000
        rows.append(dict(
            tilt_max_deg=float(np.max(p['tilt'])),
            root_z_min_mm=float(np.min(p['root_z'])) * 1000,
            com_dist_mean_mm=float(np.mean(p['com_d'])) * 1000,
            stance_z_mean_mm=float(np.mean(p['stance_z'])) * 1000,
            stance_contact_frac=float(np.mean(p['contact'])),
            swing_z_mean_mm=sw,
            swing_z_dev_mm=abs(sw - z_target * 1000),
            swing_z_min_mm=float(np.min(p['swing_z'])) * 1000,
            dead_step=p['dead_step'],
            recorded_samples=len(p['swing_z']), requested_seconds=seconds, policy_dt=policy_dt,
        ))
    return rows


def summarize(rows):
    n = len(rows)
    ok = np.ones(n, bool)
    ok &= np.array([r['dead_step'] is None for r in rows], dtype=bool)
    m = {}
    c = CRIT
    a = np.array([r['tilt_max_deg'] for r in rows]); ok &= a < c['tilt_max_deg']
    m['tilt'] = f'{a.max():.1f}° (线 {c["tilt_max_deg"]:.0f}°, 余 {c["tilt_max_deg"]-a.max():.1f}°)'
    a = np.array([r['root_z_min_mm'] for r in rows]); ok &= a > c['root_z_min_mm']
    m['root_z'] = f'{a.min():.0f}mm (线 {c["root_z_min_mm"]:.0f}mm, 余 {a.min()-c["root_z_min_mm"]:.0f}mm)'
    a = np.array([r['com_dist_mean_mm'] for r in rows]); ok &= a < c['com_dist_mm']
    m['com_dist'] = f'{a.mean():.1f}mm (线 {c["com_dist_mm"]:.0f}mm, 余 {c["com_dist_mm"]-a.max():.1f}mm)'
    a = np.array([r['stance_z_mean_mm'] for r in rows]); ok &= a < c['stance_z_mm']
    m['stance_z'] = f'{a.mean():.1f}mm (线 {c["stance_z_mm"]:.0f}mm)'
    a = np.array([r['stance_contact_frac'] for r in rows]); ok &= a >= c['stance_contact_frac']
    m['stance_contact'] = f'{a.mean()*100:.0f}% (线 {c["stance_contact_frac"]*100:.0f}%)'
    a = np.array([r['swing_z_dev_mm'] for r in rows]); ok &= a < c['swing_z_dev_mm']
    m['swing_z_dev'] = f'{a.mean():.1f}mm (线 {c["swing_z_dev_mm"]:.0f}mm)'
    dead = sum(1 for r in rows if r['dead_step'] is not None)
    return int(ok.sum()), n, m, dead


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--envs', type=int, default=32, help='每组的并行环境数')
    ap.add_argument('--seed', type=int, default=9001)
    ap.add_argument('--out', type=Path, default=ROOT / '20260919_dualpose' / 'eval')
    ap.add_argument('--only', nargs='+', choices=[g[0] for g in GROUPS],
                    help='只验收指定组(缺省全四组)')
    ap.add_argument('--push', type=float, default=0.0,
                    help='验收时的推力幅度(m/s), 须与训练一致才测得出抗扰')
    ap.add_argument('--seconds', type=float, default=DEFAULT_SECONDS,
                    help='Actual rollout duration in seconds; default 6 seconds')
    a = ap.parse_args()

    import onnxruntime as ort
    onnx_path = ascii_copy(a.checkpoint)
    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    print(f'[验收] {a.label}  checkpoint={a.checkpoint.name} sha={sha(a.checkpoint)[:12]}',
          flush=True)
    print(f'  每组 {a.envs} 个环境 × {a.seconds:g}s, 判据 {CRIT}', flush=True)

    groups = [g for g in GROUPS if a.only is None or g[0] in a.only]
    report = {}
    for name, side, pose in groups:
        rows = run_group(session, onnx_path, side, pose, a.envs, a.seed,
                         push=a.push, seconds=a.seconds)
        passed, n, m, dead = summarize(rows)
        report[name] = dict(side=side, pose=pose, passed=passed, n=n,
                            frac=passed / n, dead=dead, margins=m, rows=rows)
        print(f'  {name}: 保持率 {passed}/{n} = {passed/n*100:5.1f}%  (终止 {dead})',
              flush=True)
        for k, v in m.items():
            print(f'      {k:15s} {v}', flush=True)

    total_p = sum(v['passed'] for v in report.values())
    total_n = sum(v['n'] for v in report.values())
    print(f'\n[总] 保持率 {total_p}/{total_n} = {total_p/total_n*100:.1f}%', flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f'{a.label}.json').write_text(json.dumps(
        dict(checkpoint=str(a.checkpoint.resolve()), sha=sha(a.checkpoint), crit=CRIT,
             seconds=a.seconds, seed=a.seed, envs_per_group=a.envs, push_velocity_m_s=a.push,
             policy_dt=rows[0]['policy_dt'], policy_steps=round(a.seconds/rows[0]['policy_dt']),
             evaluator_sha256=sha(Path(__file__)), terminated_trials_cannot_pass=True,
             initial_condition='Spawn directly in selected single-leg pose; not stand-to-lift transition',
             limitations='Site height and aggregate contact criteria; not whole-sole clearance or hardware validation',
             total=[total_p, total_n], groups=report), indent=2, ensure_ascii=False),
        encoding='utf-8')
    print(f'[write] {a.out / (a.label + ".json")}', flush=True)


if __name__ == '__main__':
    main()
