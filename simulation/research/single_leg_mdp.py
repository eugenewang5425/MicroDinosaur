"""单脚站立任务的奖励项(用户 2026-09-18 指定)。

任务:按指令抬起指定的一只脚到目标高度,另一只脚支撑,身体维持平衡。
**指令通道**:复用 twist 命令里空闲的 vy 维(`+1` 抬左脚、`-1` 抬右脚;
单脚站立不需要位移,vx 与 wz 恒 0)。这样左右脚都能训而**不动 81 维观测契约**。
所有奖励项都从同一个 `twist[:,1]` 读指令 —— 指令与奖励同源,不会左右搞反。

平衡手段(用户指定):头部与尾巴同时朝抬起的脚的方向转动当配重。
不硬编码"头要转多少度"(那是手段不是目的,且符号标定易错):先只给平衡与
抬脚的目标奖励,让策略在"必须站住"的压力下自己发现头尾是最快的配重
(两者合计约占体重 20%,是上半身唯一能快速横移的质量)。验收时单独看头尾摆幅。

抬脚奖励形状 = 线性(脚高度/目标,截 [0,1]),**不是高斯**:高斯在"脚不离地"时
仍有 exp(-1)=0.37 的免费分,策略会学出"两脚站死不动"(实测两轮都是这个结局)。
"""
import torch


def _lift_side(env):
    """从 twist 指令读"抬哪只脚":True = 抬左脚(+1),False = 抬右脚(-1)。"""
    cmd = env.command_manager.get_command('twist')
    return cmd[:, 1] > 0


def single_leg_lift(env, target_height=.025, sensor_name='foot_height_scan',
                    **_ignored):
    """抬起脚高度 / 目标高度,截到 [0,1]。不动 = 0 分,单调递增,梯度恒定。"""
    h = env.scene[sensor_name].data.heights[:, :2]          # [B, F] 左,右
    lift_left = _lift_side(env)
    h_lift = torch.where(lift_left, h[:, 0], h[:, 1])
    return torch.clamp(h_lift / max(target_height, 1e-6), 0., 1.)


def support_foot_grounded(env, sensor_name='feet_ground_contact'):
    """支撑脚(非指令侧)必须踩地:着地给 1。"""
    found = env.scene[sensor_name].data.found               # [B, F, K]
    flat = found.reshape(env.num_envs, 2, -1)
    lift_left = _lift_side(env)
    sup = torch.where(lift_left, flat[:, 1, :].gt(0).any(-1),
                      flat[:, 0, :].gt(0).any(-1))
    return sup.float()


def lifted_foot_airborne(env, sensor_name='feet_ground_contact'):
    """抬起脚离地给 1(防止两脚都站着冒充单脚站立)。"""
    found = env.scene[sensor_name].data.found
    flat = found.reshape(env.num_envs, 2, -1)
    lift_left = _lift_side(env)
    air = torch.where(lift_left, flat[:, 0, :].gt(0).any(-1),
                      flat[:, 1, :].gt(0).any(-1))
    return air.logical_not().float()


def trunk_still(env, omega_scale=.6):
    """只压**角速度**(不许打转),**不压线速度** —— 允许为平衡而挪步。

    依据(2026-09-18 实测):脚底接触带仅 6.1mm 宽(弧形脚底),单脚站立时质心
    侧向容差 ±3mm。原版 trunk_still 同时压线速度,等于要求"原地不动",与物理
    条件冲突:策略只能二选一(要么不抬脚,要么抬了就挪)。实测抬左脚时它向左
    挪 2.02m —— 那是它在用移动维持平衡,是这台机构的合理动态平衡方式。
    所以放开平移,只禁止打转(打转与平衡无关,且会破坏朝向)。
    """
    robot = env.scene['robot']
    w = robot.data.root_link_ang_vel_b.square().sum(-1)
    return torch.exp(-w / omega_scale ** 2)


def stay_near_origin(env, radius=.8):
    """软约束:别跑远。允许挪步,但跑出 radius 就扣分(防止"一路走掉")。

    返回 0(在半径内)~1(超出很多),由负权重罚。
    """
    robot = env.scene['robot']
    pos = robot.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
    dist = torch.norm(pos, dim=1)
    return torch.clamp((dist - radius * .5) / (radius * .5), 0., 1.)


def lift_side_unload(env, sensor_name='feet_ground_contact', margin=.3):
    """奖励**重心转移到支撑腿**(抬脚侧卸力),这是抬脚的前置步骤。

    为什么加(2026-09-18):训练日志证明策略"从未让脚离地过"(抬脚奖励单步值
    全程≈0.0008),不是"抬了站不住"。直接要求"脚抬 10mm"跳过了中间那步 ——
    抬一条腿前必须先把重心移到另一条腿上,这一步比抬脚容易得多,却一直没被
    奖励过。形状用**相对承重差**:支撑脚受力占比越高分越高,抬脚侧离地时给满分;
    不低于 0(不允许用"两脚都轻踩"来骗分,那会触发抬高奖励的另一路)。
    """
    f = env.scene[sensor_name].data.force            # [B, F, 3]
    mag = torch.norm(f, dim=-1)                      # [B, F] 左,右
    total = mag.sum(dim=1, keepdim=True).clamp(min=1e-6)
    share = mag / total                              # 每只脚的承重占比
    lift_left = _lift_side(env)
    lift_share = torch.where(lift_left, share[:, 0], share[:, 1])
    # 抬脚侧占比 0.5 -> 0 分;降到 margin 以下 -> 满分
    return torch.clamp((.5 - lift_share) / max(margin, 1e-6), 0., 1.)


class single_leg_hold_duration:
    """奖励"连续保持抬脚"的时长 —— 震荡式短抬拿不到分。

    为什么加(2026-09-18 实测):`single_leg_lift` 是**逐步**给分,于是
    "抬起-落下"交替也能拿一半分(实测保持率 49~51%,而目标是 90%+ 的持续保持)。
    这个项对**连续**保持累积计分、一旦落下就清零,所以短促反复无效,
    只有真正把脚持续举在目标高度才有分。上限 3 秒(超过不再加,避免无限抬着)。
    """

    def __init__(self, cfg, env):
        self.dt = env.step_dt
        self.target = float(cfg.params.get('target_height', .025))
        self.floor_ratio = float(cfg.params.get('floor_ratio', .7))
        self.cap = float(cfg.params.get('cap_s', 3.))
        self.hold = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.hold.zero_()
        else:
            self.hold[env_ids] = 0.

    def __call__(self, env, **params):
        h = env.scene['foot_height_scan'].data.heights[:, :2]
        lift_left = _lift_side(env)
        h_lift = torch.where(lift_left, h[:, 0], h[:, 1])
        ok = h_lift >= self.target * self.floor_ratio
        self.hold = torch.where(ok, self.hold + self.dt,
                                torch.zeros_like(self.hold))
        return (self.hold / max(self.cap, 1e-6)).clamp(0., 1.)
