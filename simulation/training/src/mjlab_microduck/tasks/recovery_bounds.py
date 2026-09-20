"""Single source of truth for the get-up operating envelope.

Training (`mdp` / `contact_motion_cfg.RecoveryAction`) and the CPU acceptance
evaluator (`evaluate_contact_motion.py`) have to clamp identical bounds, and they
did not: the evaluator enforced ankle +/-51 deg and jaw +0.04 but no neck bound at
all, while the CAD clearance gate requires neck_pitch to stay under ~38.5 deg. The
released candidate drove neck_pitch to 56.8 deg and pushed the head-roll case
6839 mm^3 into the battery envelope -- a violation the evaluator could not see
because it never applied the bound training did.

Keeping the numbers in one importable module is what stops that class of drift.
Numpy-only on purpose: the evaluator is a plain CPython/NumPy path with no torch.
"""
import numpy as np

ANKLE_LIMIT_DEG = 51.
JAW_TARGET_RAD = .04
NECK_PITCH_MIN_DEG = -90.
NECK_PITCH_MAX_DEG = 35.
"""Target clamp. CAD breach measured at 40.75 deg nominal / 38.50 deg at the
head_roll extremes, so the clamp sits a full 3.5 deg below the worst case."""
NECK_BATTERY_TERMINATE_DEG = 38.
"""Measured-pose envelope. Higher than the clamp so servo overshoot is not fatal,
still below the 38.5 deg worst-case CAD breach."""


def apply_numpy(target, names):
    """Apply the envelope to a joint-target array shaped (..., len(names))."""
    out = np.array(target, dtype=float, copy=True)
    limit = np.deg2rad(ANKLE_LIMIT_DEG)
    for name in ('left_ankle', 'right_ankle'):
        index = names.index(name)
        out[..., index] = np.clip(out[..., index], -limit, limit)
    out[..., names.index('jaw_hinge')] = JAW_TARGET_RAD
    neck = names.index('neck_pitch')
    out[..., neck] = np.clip(out[..., neck], np.deg2rad(NECK_PITCH_MIN_DEG),
                             np.deg2rad(NECK_PITCH_MAX_DEG))
    return out


STAND_BAND_DEG = (8., 8., 8., 8., 6., 8., 8., 8., 8., None, 8., 8., 8., 8., 6., 8., 8., None, None)
"""站定后的操作包络半宽(度),按 contract 的 action_names 顺序;None = 不限。

2026-09-16 二次收紧(12-20 -> 8, 踝 6)。依据:实测站姿偏差恰好等于包络半宽
(左/前例 15.39/15.40° 对着 15° 的判据,而包络给的是 12-20°),判据要过就必须比它更紧。
前一轮(35/25/30/40 -> 15/10/15/20)的依据:实测 iter33200 的"站姿偏差"
在两例里恰好是 35.4° = 上一版给膝的半宽 35° —— **包络半宽就是站姿偏差的下限**。
既然验收要求站姿 <=15°、脚底 <=8°(踝半宽决定)、朝向 <=30°,
包络必须比判据更紧,否则判据物理上不可能通过(那是参数设定问题,不是策略问题)。
踝收到 10° 是因为 v7 HOME 踝 +26° 时脚底才平贴 0.1°,而上一版 25° 半宽仍允许
~25° 的脚底倾角。"""
"""站定后的操作包络半宽(度),按 contract 的 action_names 顺序;None = 不限。

只在 phi > BAND_GATE 之后生效,翻滚起身阶段完全不限(深屈膝仍可用)。
存在的理由:实测策略在 ±0.07rad 夹取下把膝/髋/颈/尾饱和在有效边界上
(膝 −86° = jnt_range[0]+0.07rad),而物理挡块在 ±90° 从未被碰到;
卡死姿态与 v7 正确站姿根高几乎相同(116 vs 114mm),十二条判据分辨不出。
把站起后的可夹取范围改成 HOME±半宽,是从动作空间上关掉那条路。"""

BAND_GATE = .6
BAND_RAMP = .15


def apply_band_numpy(target, names, home, phi):
    """把站定后的目标夹到 HOME±STAND_BAND_DEG,按 phi 平滑过渡到该包络。

    `home` 由调用方给出(训练侧取 contract.action_offset,评估侧取 sim.home),
    避免这里再去读文件而和调用方漂开。半宽按 STAND_BAND_DEG 与 names 同序对齐。
    """
    out = np.array(target, dtype=float, copy=True)
    weight = float(np.clip((float(phi) - BAND_GATE) / BAND_RAMP, 0., 1.))
    if weight <= 0.:
        return out
    home = np.asarray(home, dtype=float).reshape(-1)
    for i, _ in enumerate(names):
        half_deg = STAND_BAND_DEG[i] if i < len(STAND_BAND_DEG) else None
        if half_deg is None:
            continue
        half = np.deg2rad(half_deg)
        clipped = np.clip(out[..., i], home[i] - half, home[i] + half)
        out[..., i] = out[..., i] * (1. - weight) + clipped * weight
    return out


def phi_numpy(rotation, root_z, env_z=0.):
    """与 mdp.recovery_potential 同式的 phi,供 CPU 评估侧使用。"""
    g_z = float((rotation.T @ np.array([0., 0., -1.]))[2])
    upright = min(1., max(0., (1. - g_z) / 2.))
    height = min(1., max(0., ((root_z - env_z) - .025) / (.1135 - .025)))
    return .7 * upright + .3 * height


# ---------------------------------------------------------------------------
# 坐撑折腿示范(单一真源;训练侧 contact_motion_cfg.SitFoldResidualAction 与
# 评估侧 evaluate_contact_motion 的 back 案例都从这里取,防止两侧漂移)。
#
# 依据(2026-09-17 实测,try_sit_reference.py):从坐撑态(back 方位倒地)出发,
# "折腿 2s → 平滑伸腿 3s → HOME 保持"4/4 seed 全过判据 —— 倾角 0.06°、根高
# 114.9mm、双脚各 5.39N、非足 0.00N、站姿 1.63-3.12°、偏航 3.5e-05°、末 2s 漂移
# 0.0mm。膝折方向是正的(knee_range_audit:+90° 给出最紧的 31.59° 进位角)。
# 同一折腿从 left/right/front 出发无效(侧躺折腿没有可蹬的东西),所以示范
# 只对坐撑起点生效,按落地签名分类。
# ---------------------------------------------------------------------------

SITFOLD_FOLD_DEG = {'left_hip_pitch': 86., 'right_hip_pitch': -86.,
                    'left_knee': 86., 'right_knee': -86.}
SITFOLD_FOLD_S = 2.
"""第一段时长:g0(倒地真实关节角)→FOLD 的 smoothstep。腿到位的瞬间正是伸腿
开始的时候 —— 这是一段连续的卷腿-弹起动作(实测两个教训:跳到 FOLD 干等 1.4s,
身体塌到 78mm;伸腿改成阶跃也一样起不来 —— 都丢了验证脚本的连续性)。"""
SITFOLD_EXTEND_S = 3.
"""第二段时长:FOLD→HOME 的 smoothstep。"""
SITFOLD_RESIDUAL_AFTER_S = SITFOLD_FOLD_S + SITFOLD_EXTEND_S
"""残差生效起点(t>5s,站定之后)。依据(2026-09-17 第⑱轮):折腿期给残差,
36549 的旧蜷缩指令经 tanh 饱和后 ±0.86° 就足以把动作推进"坐起"盆地(零残差
对照 100% 通过);而脚本本身缺的只有站定后的主动平衡(抗扰临界 0.6-1.0N)。
所以折腿期纯脚本,策略的残差只在站定后学平衡。"""
SITFOLD_RESIDUAL_RAD = .015
"""有界残差半宽(rad),与蹲技能 SquatResidualAction 同量级 —— 策略只负责
脚本没有的那块:主动平衡(实测脚本抗扰临界仅 0.6-1.0N)。"""
SITFOLD_SIT_ROOT_Z = .12
"""坐撑起点签名:根高下限(实测 back 起点 137mm;front 36mm 进不了)。"""
SITFOLD_SIT_TILT_DEG = 65.
"""坐撑起点签名:倾角上限(实测 back 44.7°;left/right 110° 进不了)。"""


def sitfold_reference(t, g0, names, home):
    """坐撑折腿参考轨迹(与 try_sit_reference.py 的三段式逐式对齐)。

    t(秒,标量或 len N) ;g0(N,len) episode 开始时的真实关节角;
    返回 (N,len)。g0→FOLD smoothstep 2s → FOLD→HOME smoothstep 3s → HOME。
    时序与符号是验证过的脚本值,不要在这里调参 —— 要改就先重跑
    try_sit_reference.py 拿到 4/4 再改。
    """
    home = np.asarray(home, dtype=float).reshape(-1)
    fold = home.copy()
    names = list(names)
    for n, d in SITFOLD_FOLD_DEG.items():
        fold[names.index(n)] = np.deg2rad(d)
    t = np.atleast_1d(np.asarray(t, dtype=float))
    g0 = np.atleast_2d(np.asarray(g0, dtype=float))
    s1 = np.clip(t / SITFOLD_FOLD_S, 0., 1.)[:, None]
    s1 = s1 * s1 * (3. - 2. * s1)
    s2 = np.clip((t - SITFOLD_FOLD_S) / SITFOLD_EXTEND_S, 0., 1.)[:, None]
    s2 = s2 * s2 * (3. - 2. * s2)
    return (g0 * (1. - s1) + fold * s1) * (1. - s2) + home * s2
