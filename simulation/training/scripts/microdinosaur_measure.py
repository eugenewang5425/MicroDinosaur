"""Quantify the three user-reported gait complaints on a policy, headless.

Reports, for one checkpoint's ONNX:
  A. zero-command standing: yaw drift over 5 s, residual motion (jitter, steps)
  B. turning: achieved yaw rate vs the commanded +/-0.9 rad/s
  C. large stride: forward speed and LEFT/RIGHT foot swing + stance asymmetry
     at action scale 1.00 vs 1.15 (the "right heel catches" regime)

Usage (from the repo root):
  uv run scripts/microdinosaur_measure.py --onnx microdinosaur_p2.onnx
  uv run scripts/microdinosaur_measure.py --ckpt logs/.../model_15500.pt
"""

import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import microdinosaur_play as play  # noqa: E402

FEET = ("ankle_left", "ankle_right")


def yaw_of(sim):
    w, x, y, z = sim.data.qpos[3:7]
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def ang_vel(sim):
    return np.array(sim.data.sensordata[sim.imu_adr : sim.imu_adr + 3], dtype=float)


def foot_z(sim):
    return {n: float(sim.data.xpos[play.mujoco.mj_name2id(
        sim.model, play.mujoco.mjtObj.mjOBJ_BODY, n)][2]) for n in FEET}


def zero_cmd():
    return np.zeros(18, dtype=np.float32)


FOOT_BODIES = {"ankle_left": "left_foot_collision", "ankle_right": "right_foot_collision"}


def foot_contact_slip(sim):
    """Per-foot body speed while its collision geom is in contact.

    Grinding turning shows here as a LARGE contact-phase speed with no lift;
    lift-and-place turning shows a small contact-phase speed with a large lift
    (see phase_turn).  Judging turning by the achieved yaw rate alone is not
    enough -- grinding can score well on that.
    """
    out = {n: None for n in FEET}
    m = sim.model
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        for gg in (c.geom1, c.geom2):
            gname = play.mujoco.mj_id2name(m, play.mujoco.mjtObj.mjOBJ_GEOM, gg) or ""
            for body, geom in FOOT_BODIES.items():
                if gname != geom:
                    continue
                b = play.mujoco.mj_name2id(m, play.mujoco.mjtObj.mjOBJ_BODY, body)
                v = np.zeros(6)
                play.mujoco.mj_objectVelocity(
                    m, sim.data, play.mujoco.mjtObj.mjOBJ_BODY, b, v, 0)
                out[body] = float(np.linalg.norm(v[3:5]))
    return out


def phase_stand(sim, settle_s=2.0, measure_s=5.0, perturb_deg=0.0, rng=None):
    """Stand still and report drift / residual motion.

    With perturb_deg > 0 the initial state is nudged (yaw, joint angles, joint
    velocities) so repeated trials measure whether standing is ROBUST or merely
    lucky at the nominal reset.  A single unperturbed trial is not enough: the
    same script gave +0.79 deg, -33.7 deg and +71.4 deg for three different
    policies, which is either a real qualitative difference or chaos -- trials
    with perturbations separate the two.
    """
    cmd = zero_cmd()
    if perturb_deg:
        d, m = sim.data, sim.model
        yaw = np.radians(rng.uniform(-perturb_deg, perturb_deg))
        d.qpos[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        for a in sim.jadr:
            d.qpos[a] += np.radians(rng.uniform(-perturb_deg, perturb_deg))
        for a in sim.dofadr:
            d.qvel[a] = rng.uniform(-0.2, 0.2)
        play.mujoco.mj_forward(m, d)
    for _ in range(int(settle_s * 50)):
        sim.step(cmd, 1.0)
    y0 = yaw_of(sim)
    wz, lvxy, dact = [], [], []
    prev_act = None
    zs = {n: [] for n in FEET}
    for _ in range(int(measure_s * 50)):
        sim.step(cmd, 1.0)
        wz.append(ang_vel(sim)[2])
        lvxy.append(float(np.linalg.norm(sim.data.qvel[:2])))
        a = np.asarray(sim.last_action, dtype=float)
        if prev_act is not None:
            dact.append(float(np.abs(a - prev_act).mean()))
        prev_act = a
        for n, z in foot_z(sim).items():
            zs[n].append(z)
    dyaw = np.degrees(yaw_of(sim) - y0)
    steps = {n: int(np.sum(np.diff(np.sign(np.diff(np.array(zs[n])))) < 0)) for n in FEET}
    lift = {n: float((max(zs[n]) - min(zs[n])) * 1000.0) for n in FEET}
    return {
        "yaw_drift_deg_5s": dyaw,
        "yaw_rate_deg_s": float(np.degrees(np.mean(wz))),
        "lin_vel_m_s": float(np.mean(lvxy)),
        "action_jerk": float(np.mean(dact)),
        "steps": steps,
        "foot_lift_mm": lift,
    }


def phase_turn(sim, wz_cmd, settle_s=1.5, measure_s=4.0):
    cmd = zero_cmd()
    cmd[2] = wz_cmd
    for _ in range(int(settle_s * 50)):
        sim.step(cmd, 1.0)
    w = []
    zs = {n: [] for n in FEET}
    slip = {n: [] for n in FEET}
    for _ in range(int(measure_s * 50)):
        sim.step(cmd, 1.0)
        w.append(ang_vel(sim)[2])
        for n, z in foot_z(sim).items():
            zs[n].append(z)
        for n, v in foot_contact_slip(sim).items():
            if v is not None:
                slip[n].append(v)
    out = {
        "cmd": wz_cmd,
        "achieved": float(np.mean(w)),
        "ratio": float(np.mean(w) / wz_cmd) if wz_cmd else float("nan"),
        "lift": {n: float((max(zs[n]) - min(zs[n])) * 1000.0) for n in FEET},
        "slip": {n: float(np.mean(slip[n]) * 1000.0) if slip[n] else 0.0 for n in FEET},
    }
    return out


def phase_forward(sim, scale, vx=0.55, settle_s=2.0, measure_s=6.0):
    cmd = zero_cmd()
    cmd[0] = vx
    for _ in range(int(settle_s * 50)):
        sim.step(cmd, scale)
    vx_s, zs = [], {n: [] for n in FEET}
    # Per-step COMMAND demand while walking (rad) -- sizes the slew limit.
    prev_cmd, slew = None, []
    # View stability: the head_camera site is trunk-mounted, so both the head
    # bodies' and the trunk's WORLD angular rate matter (user: "走路的时候头
    # 尽量稳定……保持视角稳定").
    gaze_ang = []
    trunk_ang = []
    head_dev = []
    q_head0 = None
    head_ids = [play.mujoco.mj_name2id(sim.model, play.mujoco.mjtObj.mjOBJ_BODY, n)
                for n in ("neck", "neck_pitch", "yaw_roll_motion")]
    head_body = play.mujoco.mj_name2id(sim.model, play.mujoco.mjtObj.mjOBJ_BODY, "jaw_soft")
    trunk_id = sim.trunk
    for _ in range(int(measure_s * 50)):
        sim.step(cmd, scale)
        vx_s.append(float(sim.data.qvel[0]))
        for n, z in foot_z(sim).items():
            zs[n].append(z)
        cur = np.asarray(sim.last_action, dtype=float) * scale
        if prev_cmd is not None:
            slew.append(float(np.abs(cur - prev_cmd).max()))
        prev_cmd = cur
        v = np.zeros(6)
        for b in head_ids:
            play.mujoco.mj_objectVelocity(sim.model, sim.data,
                                          play.mujoco.mjtObj.mjOBJ_BODY, b, v, 0)
            gaze_ang.append(float(np.linalg.norm(v[:3])))
        play.mujoco.mj_objectVelocity(sim.model, sim.data,
                                      play.mujoco.mjtObj.mjOBJ_BODY, trunk_id, v, 0)
        trunk_ang.append(float(np.linalg.norm(v[:3])))
        # view stability: how far the HEAD's world attitude wanders from its
        # attitude when the walk started (deg) -- this is what a head-mounted
        # camera actually sees, and it is the quantity the head_world_gaze
        # reward targets.
        qh = np.array(sim.data.xquat[head_body], dtype=float)
        if q_head0 is None:
            q_head0 = qh
        dot = abs(float(np.dot(qh, q_head0)))
        head_dev.append(np.degrees(2.0 * np.arccos(min(1.0, dot))))
    slew_stats = {}
    if slew:
        a = np.array(slew)
        slew_stats = {"mean": float(a.mean()), "p95": float(np.percentile(a, 95)),
                      "p99": float(np.percentile(a, 99)), "max": float(a.max()),
                      "over008": float((a > 0.08).mean()),
                      "over015": float((a > 0.15).mean())}
    out = {"scale": scale, "vx": float(np.mean(vx_s)), "vx_std": float(np.std(vx_s)),
           "slew": slew_stats,
           "head_ang_rms": float(np.sqrt(np.mean(np.square(gaze_ang)))),
           "head_att_dev_rms": float(np.sqrt(np.mean(np.square(head_dev)))),
           "trunk_ang_rms": float(np.sqrt(np.mean(np.square(trunk_ang)))),
           "feet": {}}
    for n in FEET:
        z = np.array(zs[n]) * 1000.0
        out["feet"][n] = {"min": float(z.min()), "max": float(z.max()),
                          "lift": float(z.max() - z.min()), "mean": float(z.mean())}
    L, R = out["feet"]["ankle_left"], out["feet"]["ankle_right"]
    out["lift_ratio_L_over_R"] = L["lift"] / R["lift"] if R["lift"] else float("nan")
    out["stance_diff_mm"] = L["min"] - R["min"]
    return out


def phase_tail_up(sim, tail_pitch, scale=1.0, vx=0.35, settle_s=2.0, measure_s=5.0):
    """Walk with the tail commanded up: does it raise, and does it stay up?

    2026-09-13 user: "让尾巴完全竖起来…实现尾巴可以竖着走".  The joint limit
    and the command range were widened to +1.75 rad, so the question is whether
    the POLICY follows the command while walking and stays upright doing it.
    """
    cmd = zero_cmd()
    cmd[0] = vx
    cmd[16] = tail_pitch
    jid = sim.jids[play.JOINT_ORDER.index("tail_pitch")]
    for _ in range(int(settle_s * 50)):
        sim.step(cmd, scale)
    ang, vxs, tilt = [], [], []
    zmin = 1.0
    for _ in range(int(measure_s * 50)):
        sim.step(cmd, scale)
        ang.append(float(sim.data.qpos[sim.model.jnt_qposadr[jid]]))
        vxs.append(float(sim.data.qvel[0]))
        R = sim.data.xmat[sim.trunk].reshape(3, 3)
        tilt.append(float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))))
        zmin = min(zmin, float(sim.data.xpos[sim.trunk][2]))
    return {
        "cmd": tail_pitch,
        "achieved": float(np.mean(ang)),
        "achieved_deg": float(np.degrees(np.mean(ang))),
        "vx": float(np.mean(vxs)),
        "tilt": float(np.mean(tilt)),
        "fell": bool(zmin < 0.05),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--model", default="fine", choices=list(play.DEFAULT_XML))
    ap.add_argument("--kp", type=float, default=7.0)
    ap.add_argument("--kv", type=float, default=0.4)
    ap.add_argument("--iters-solver", type=int, default=20)
    ap.add_argument("--scale", type=float, default=1.15,
                    help="large-stride action scale for phase C")
    ap.add_argument("--trials", type=int, default=5,
                    help="repeats of the standing phase with perturbed initial states")
    ap.add_argument("--perturb-deg", type=float, default=1.0,
                    help="initial-state perturbation amplitude for the standing trials")
    ap.add_argument("--lp-alpha", type=float, default=0.9,
                    help="action low-pass EMA (must match training)")
    ap.add_argument("--max-delta", type=float, default=0.6,
                    help="per-step action change limit (must match training)")
    ap.add_argument("--foot-contact", default="condim6", choices=("condim6", "condim4", "legacy"),
                    help="foot contact model: condim6 (training) or legacy condim=3")
    args = ap.parse_args()

    onnx = args.onnx
    if onnx is None:
        if not args.ckpt:
            raise SystemExit("need --onnx or --ckpt")
        onnx = "_measure_tmp.onnx"
        print(f"[measure] exporting {args.ckpt} ...", flush=True)
        subprocess.run(["uv", "run", "scripts/export.py",
                        "Mjlab-Velocity-Flat-MicroDinosaur",
                        "--checkpoint-file", args.ckpt, "--onnx-file", onnx], check=False)

    sim = play.Sim(play.DEFAULT_XML[args.model], onnx, kp=args.kp, kv=args.kv,
                   iters=args.iters_solver, foot_contact=args.foot_contact,
                   lp_alpha=args.lp_alpha,
                   max_delta=(None if args.max_delta <= 0 else args.max_delta))

    print(f"\n[measure] model={args.model}  kp={args.kp} kv={args.kv}  onnx={onnx}")

    sim.reset()
    n = max(1, args.trials)
    rng = np.random.default_rng(12345)  # same perturbations for every policy
    trials = []
    for k in range(n):
        if k:
            sim.reset()
        trials.append(phase_stand(sim, perturb_deg=args.perturb_deg if k or n == 1 else 0.0,
                                  rng=rng))
    def ms(key, scale=1.0):
        v = np.array([t[key] for t in trials], dtype=float) * scale
        return (v.mean(), v.std(), v.min(), v.max())
    d_m, d_s, d_lo, d_hi = ms("yaw_drift_deg_5s")
    print(f"\nA. 零指令站立({n} 次试验, 初始扰动 ±{args.perturb_deg:g}°, 静置 2 s 后测 5 s)")
    print(f"   偏航漂移        {d_m:+.2f} ± {d_s:.2f} deg / 5 s   (范围 {d_lo:+.1f} ~ {d_hi:+.1f})")
    for key, label, sc in (("yaw_rate_deg_s", "平均偏航角速度", 1.0),
                           ("lin_vel_m_s", "水平速度", 1000.0),
                           ("action_jerk", "动作抖动", 1.0)):
        m, s, lo, hi = ms(key, sc)
        print(f"   {label}    {m:+.3f} ± {s:.3f}"
              + (" mm/s" if key == "lin_vel_m_s" else (" deg/s" if key == "yaw_rate_deg_s" else "")))
    st_L = np.array([t["steps"]["ankle_left"] for t in trials])
    st_R = np.array([t["steps"]["ankle_right"] for t in trials])
    print(f"   踏步次数 L/R    {st_L.mean():.0f} / {st_R.mean():.0f}")
    lf_L = np.array([t["foot_lift_mm"]["ankle_left"] for t in trials])
    lf_R = np.array([t["foot_lift_mm"]["ankle_right"] for t in trials])
    print(f"   站立时抬脚 L/R  {lf_L.mean():.1f} / {lf_R.mean():.1f} mm"
          f"   (<3 mm = 真站着; ~20 mm = 在原地踏步)")
    print("   (同一次测量里 漂移大且多次试验范围宽 = 站立不稳定; 全在 0 附近 = 站得住)")

    print("\nB. 原地转向(指令 ±0.9 rad/s) —— 要求抬脚转,不许磨地拧")
    print(f"   {'指令':>5s}{'实测':>10s}{'达成度':>9s}{'抬脚L/R':>15s}{'接触期滑移L/R':>18s}")
    for wz_cmd in (0.9, -0.9):
        sim.reset()
        b = phase_turn(sim, wz_cmd)
        print(f"   {b['cmd']:+.2f}{b['achieved']:>+10.3f}{b['ratio']*100:>+8.0f}%"
              f"{b['lift']['ankle_left']:>8.1f}/{b['lift']['ankle_right']:<6.1f}mm"
              f"{b['slip']['ankle_left']:>10.1f}/{b['slip']['ankle_right']:<7.1f}mm/s")

    print(f"\nC. 前进 0.55 m/s(动作幅度 1.00 与 {args.scale:.2f} 对比)")
    for sc in (1.0, args.scale):
        sim.reset()
        c = phase_forward(sim, sc)
        L, R = c["feet"]["ankle_left"], c["feet"]["ankle_right"]
        print(f"   scale {sc:.2f}: 前进 {c['vx']:+.3f} m/s (σ {c['vx_std']:.3f})"
              f"  抬脚 L {L['lift']:.1f} / R {R['lift']:.1f} mm"
              f"  (比值 {c['lift_ratio_L_over_R']:.2f})"
              f"  最低点差 L-R {c['stance_diff_mm']:+.2f} mm")
        sl = c["slew"]
        if sl:
            print(f"              每步动作变化需求: 均值 {sl['mean']:.3f}  P95 {sl['p95']:.3f}"
                  f"  P99 {sl['p99']:.3f}  最大 {sl['max']:.3f} rad"
                  f"  (>0.08 占 {sl['over008']*100:.0f}%, >0.15 占 {sl['over015']*100:.0f}%)")
        print(f"              视角稳定: 头部世界姿态偏离 RMS {c['head_att_dev_rms']:.1f} deg"
              f"  头部角速度 RSB {c['head_ang_rms']*57.3:.1f} deg/s"
              f"  躯干 {c['trunk_ang_rms']*57.3:.1f} deg/s  (越小越稳)")
    print("\nD. 尾巴竖着走(前进 0.35 m/s + 尾巴抬起指令)")
    print(f"   {'指令rad':>8s}{'指令deg':>9s}{'尾部实际deg':>12s}{'跟上比例':>10s}{'前进m/s':>10s}{'机身倾角':>10s}{'是否摔':>7s}")
    for tp in (0.4, 0.9, 1.4, 1.57, 1.75):
        sim.reset()
        e = phase_tail_up(sim, tp)
        ratio = e["achieved"] / tp if tp else float("nan")
        print(f"   {e['cmd']:>8.2f}{np.degrees(e['cmd']):>9.0f}{e['achieved_deg']:>12.1f}"
              f"{ratio*100:>9.0f}%{e['vx']:>10.3f}{e['tilt']:>9.1f}°"
              f"{('摔' if e['fell'] else '否'):>7s}")

    print("\n判读: 偏航漂移越小越好; 转向达成度 100% = 完全跟上; "
          "抬脚比值与最低点差接近 1.00 / 0.00 mm = 左右协同好")

    if args.onnx is None and os.path.exists(onnx):
        os.remove(onnx)


if __name__ == "__main__":
    main()
