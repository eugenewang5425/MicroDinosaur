"""命令式姿态栈(零训练):起身/站立同源的深蹲 + 折叠蹲 + 尾/脖/臂/颌偏移。

全部通道都是指令,没有任何新训练:
  body_z    深蹲幅度 0..-25mm(蹲专项残差网络出平衡,obs[72] 通道)
  fold      折叠蹲比例 0..1(起身模组验证过的 ±86° 紧凑折腿插值)
  tail_pitch / tail_yaw / arm_l / arm_r / neck_pitch / jaw   直接偏移(rad)
头部俯仰/偏航由 IMU 稳定环托管(yaw_follows_trunk),不在命令栈内。
"""
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from skill_console import SquatStepper  # noqa: E402
from mjlab_microduck.tasks import recovery_bounds  # noqa: E402

CHANNELS = {
    'body_z':     (-0.025, 0.0,  '深蹲幅度 (m)'),
    'fold':       (0.0, 1.0,     '折叠蹲比例'),
    'tail_pitch': (-0.45, 0.45,  '尾俯仰 (rad)'),
    'tail_yaw':   (-0.55, 0.55,  '尾偏航 (rad)'),
    'arm_l':      (-0.65, 0.65,  '左臂 (rad)'),
    'arm_r':      (-0.65, 0.65,  '右臂 (rad)'),
    'neck_pitch': (-0.18, 0.28,  '脖子俯仰 (rad)'),
    'jaw':        (0.0, 0.25,    '嘴 (rad)'),
}


class PoseStack:
    def __init__(self, seed=941):
        self.st = SquatStepper(HERE / '20260914_squat_specialist' / 'plant',
                               HERE / '20260914_squat_specialist' / 'imitation/candidate.onnx')
        self.st.reset(seed)
        s = self.st.sim
        self.s = s
        self.names = s.names
        self.home = np.asarray(s.home, dtype=float)
        self.legs = self.st.legs
        ix = {n: self.names.index(n) for n in
              ('tail_pitch', 'tail_yaw', 'arm_l', 'arm_r', 'neck_pitch', 'jaw_hinge')}
        self.ix = ix
        fold = self.home.copy()
        for n, deg in recovery_bounds.SITFOLD_FOLD_DEG.items():
            fold[self.names.index(n)] = np.deg2rad(deg)
        self.fold_pose = fold
        self.limits = s.model.jnt_range[s.jids]
        self.state = {k: 0. for k in CHANNELS}
        self.state['jaw'] = .04
        s.transform_target = self._trajectory

    def _trajectory(self, target, obs):
        st, ix, s = self.state, self.ix, self.s
        fraction = np.clip(-float(obs[72]) / .025, 0, 1)
        goal = self.st.initial.copy()
        goal[self.legs] += fraction * (self.st.ref25 - self.home)[self.legs]
        goal[self.legs] += st['fold'] * (self.fold_pose - self.home)[self.legs]
        raw = (target - s.home) / s.scale
        goal[self.legs] += .015 * np.tanh(raw[self.legs])
        goal[ix['tail_pitch']] += st['tail_pitch']
        goal[ix['tail_yaw']] += st['tail_yaw']
        goal[ix['arm_l']] += st['arm_l']
        goal[ix['arm_r']] += st['arm_r']
        goal[ix['neck_pitch']] += st['neck_pitch']
        goal[ix['jaw_hinge']] = st['jaw']
        goal[self.legs] = np.clip(goal[self.legs],
                                  self.limits[self.legs, 0] + .07,
                                  self.limits[self.legs, 1] - .07)
        return self.st.exp.transform_target(goal, obs)

    def apply(self, cmd):
        for k in self.state:
            if k in cmd:
                lo, hi, _ = CHANNELS[k]
                self.state[k] = float(np.clip(cmd[k], lo, hi))

    def step(self):
        self.st.step({'body_z': self.state['body_z']})

    def readout(self):
        """(root_z mm, 倾角 deg, 双脚法向 N, 非足 N) — 检查窗口读数用。"""
        import mujoco
        d = self.s.data
        m = self.s.model
        rot = d.xmat[self.s.body].reshape(3, 3)
        tilt = float(np.degrees(np.arccos(np.clip(rot[2, 2], -1, 1))))
        floor = m.geom('terrain').id
        w = np.zeros(6)
        feet = 0.
        nonfoot = 0.
        for i, c in enumerate(d.contact):
            if floor not in (c.geom1, c.geom2):
                continue
            other = c.geom2 if c.geom1 == floor else c.geom1
            mujoco.mj_contactForce(m, d, i, w)
            load = max(0., w[0])
            if other in self.s.foot_geoms:
                feet += load
            else:
                nonfoot += load
        return dict(root_z_mm=float(d.qpos[2] * 1000), tilt_deg=tilt,
                    foot_n=float(feet), nonfoot_n=float(nonfoot))
