"""Independent motion specialists on the corrected current CAD plant."""
from pathlib import Path
import json
from functools import partial
from dataclasses import dataclass
import numpy as np
import torch
from mjlab.managers import EventTermCfg,RewardTermCfg,TerminationTermCfg
from mjlab.sensor import ContactSensorCfg,ContactMatch
from mjlab_microduck.tasks import mdp
from run_jump_cfg import build_config as base_config
from squat_plant import robot_spec,XML
from compact_contact_cfg import finalize_contacts
from squat_skill_cfg import SquatActionCfg,SquatAction
from jump_refine_cfg import limit_failure

OUT=Path(__file__).parent/'20260914_contact_motion'

# HOME 附近的半宽(度),按 action_names 顺序;None = 不限。只在"已经站起"之后收紧。
# 依据:策略站的是关节限位自锁姿态(九个关节顶在硬限位 3-4° 内、力矩仅 0.006-0.2N·m,
# 而臂/尾顶到 0.6 饱和)——免控,所以奖励改不动它。v7 专家站姿 114.0mm/膝0°/髋-27°/
# 颈+19°/尾+5° 与卡死姿态 116mm/膝-86° 在高度上几乎相同,判据分辨不出。
from mjlab_microduck.tasks.recovery_bounds import (STAND_BAND_DEG,BAND_GATE,BAND_RAMP,
    SITFOLD_FOLD_DEG,SITFOLD_FOLD_S,SITFOLD_EXTEND_S,SITFOLD_RESIDUAL_RAD,
    SITFOLD_RESIDUAL_AFTER_S,SITFOLD_SIT_ROOT_Z,SITFOLD_SIT_TILT_DEG)

class RecoveryAction(SquatAction):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.controller.limits=self.controller.limits.clone()
        # 操作包络必须在验收界**之内**留余量。实测踩过:这里设 [3°,29°] 而验收要求 head_pitch<=29°,
        # 操作界与验收界完全相等 => 实测顶到 29.04~29.07°, 每个 checkpoint 都贴边过不了。
        # 现改为 [3°, 26.5°], 留 2.5° 伺服超调余量。
        self.controller.limits[1]=torch.tensor([np.deg2rad(5)-.035,np.deg2rad(26.5)],device=env.device)
        limits=env.sim.mj_model.jnt_range[[env.sim.mj_model.joint('robot/'+n).id for n in self._target_names]]
        self.recovery_limits=torch.tensor(limits,device=env.device,dtype=torch.float32)
        self.neck=self._target_names.index('neck_pitch')
        contract=json.loads((OUT/'normal_contact_plant'/'contract.json').read_text())
        home=torch.tensor(contract['action_offset'][0],device=env.device,dtype=torch.float32)
        band=torch.tensor([0. if d is None else np.deg2rad(d) for d in STAND_BAND_DEG],
                          device=env.device,dtype=torch.float32)
        assert len(STAND_BAND_DEG)==len(self._target_names)
        self.home=home;self.band=band
        self.band_gate=float(cfg.band_gate);self.band_ramp=float(cfg.band_ramp)
    def process_actions(self,actions):
        target=self._offset+self._scale*actions
        target=target.clamp(self.recovery_limits[:,0]+.07,self.recovery_limits[:,1]-.07)
        # CAD clearance: past ~40.8 deg of neck_pitch the head-roll case enters the
        # battery envelope by up to 6839 mm^3. Both bodies hang off trunk_base, so
        # MuJoCo filters their contact by construction and no collision proxy can
        # express it; the joint envelope is the enforceable form. The CPU evaluator
        # applies the same bound, so this is an operating limit, not a training trick.
        target[:,self.neck]=target[:,self.neck].clamp(-np.deg2rad(90),
            np.deg2rad(mdp.NECK_BATTERY_CLEARANCE_DEG))
        # 站定后的操作包络(与踝±51/neck≤35 同一惯例:操作包络,不动物理限位)。
        # 实测:九个关节被顶在硬限位 3-4° 内、力矩 0.006-0.2 N·m,双臂/尾顶到 0.6 饱和
        # => 靠约束求解器自锁的免控姿态,奖励改不动力学。把 HOME 附近的包络**只在站起来
        # 之后**收紧,让"卡死"从动作空间消失;起身翻滚阶段(phi<=gate)完全不限,深屈膝可用。
        phi=mdp.recovery_potential(self.env)
        w=((phi-self.band_gate)/self.band_ramp).clamp(0,1)[:,None]
        half=self.band[None,:]
        banded=torch.clamp(target,self.home[None,:]-half,self.home[None,:]+half)
        target=target*(1-w)+banded*w
        super().process_actions((target-self._offset)/self._scale)
        self._raw_actions[:]=actions

@dataclass(kw_only=True)
class RecoveryActionCfg(SquatActionCfg):
    band_gate: float = .6
    band_ramp: float = .15
    def build(self,env):return RecoveryAction(self,env)


class SitFoldResidualAction(RecoveryAction):
    """坐撑示范 + 有界残差(与蹲技能 SquatResidualAction 同构)。

    用户 2026-09-17 指示:"坐着直接收腿就能站起来,教他做"。脚本是已验证的紧凑
    折腿参考(recovery_bounds.sitfold_reference;try_sit_reference 4/4 seed 全过:
    倾角 0.06°、双脚各 5.39N、非足 0、偏航 3.5e-05°),策略输出只以
    ±SITFOLD_RESIDUAL_RAD 的有界残差进入,负责脚本没有的那块 —— 主动平衡
    (实测脚本抗扰临界仅 0.6-1.0N,踝 0.6N·m 本可给约 3 倍 CoP 行程)。

    生效范围:episode 开始时按落地签名逐环境分类(根高>12cm 且倾角<65°;实测
    back 起点 137mm/44.7° 唯一入选;front 36mm、left/right 倾角 110° 都进不了),
    分类一旦定死不在中途重判(站起后签名会变)。非示范环境保持 RecoveryAction
    原行为(含站定包络),完全不被打断。

    为什么示范环境绕过站定包络:包络在 phi>0.6 收紧,而坐撑起点 phi≈0.90
    (倾角 44.7°→upright 0.856、根高 137mm→height 1.0),从第一步就全额生效,
    会把 ±86° 的折腿从动作空间里夹没。示范环境不需要包络防蜷缩:脚本在
    t>HOLD+EXTEND 后把目标钉在 HOME±0.86° 上,蜷缩终态在结构上够不到。
    """

    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.legs=[i for i,n in enumerate(self._target_names) if n.startswith(('left_','right_'))]
        fold=self.home.clone()
        for n,d in SITFOLD_FOLD_DEG.items():
            fold[self._target_names.index(n)]=np.deg2rad(d)
        self.fold=fold
        self.script_mask=None
        self.g0=None

    def process_actions(self,actions):
        target=self._offset+self._scale*actions
        lo=self.recovery_limits[:,0]+.07;hi=self.recovery_limits[:,1]-.07
        target=target.clamp(lo,hi)
        target[:,self.neck]=target[:,self.neck].clamp(-np.deg2rad(90),
            np.deg2rad(mdp.NECK_BATTERY_CLEARANCE_DEG))
        fresh=self.env.episode_length_buf<=1
        # 首个调用可能发生在任何 env 判 fresh 之前(mjlab 的 episode_length_buf 初值
        # 不保证 <=1),此时对全体环境做一次完整分类,不能让 mask 停在 None。
        if self.script_mask is None or fresh.any():
            robot=self.env.scene['robot']
            tilt=torch.acos((-robot.data.projected_gravity_b[:,2]).clamp(-1.,1.))
            z=robot.data.root_link_pos_w[:,2]-self.env.scene.env_origins[:,2]
            m=(z>SITFOLD_SIT_ROOT_Z)&(tilt<np.deg2rad(SITFOLD_SIT_TILT_DEG))
            if self.script_mask is None or self.script_mask.shape!=m.shape:
                self.script_mask=m.clone()
            else:
                self.script_mask[fresh]=m[fresh]
            self.env.sitfold_script_mask=self.script_mask
            # 三段式参考的起点:复位时的真实关节角(与 try_sit_reference 的 g0 同源)
            g0=(robot.data.joint_pos+robot.data.encoder_bias)[:,self._target_ids]
            if self.g0 is None or self.g0.shape!=g0.shape:
                self.g0=g0.detach().clone()
            else:
                self.g0[fresh]=g0[fresh]
        if self.script_mask.any():
            t=self.env.episode_length_buf*self.env.step_dt
            # 三段式(与 recovery_bounds.sitfold_reference 同式):g0→FOLD smoothstep
            # 2s → FOLD→HOME smoothstep 3s → HOME。连续性是起身动量的一部分,实测
            # "跳到 FOLD 干等"或"阶跃伸腿"身体都起不来。
            u1=(t/SITFOLD_FOLD_S).clamp(0.,1.);s1=u1*u1*(3.-2.*u1)
            u2=((t-SITFOLD_FOLD_S)/SITFOLD_EXTEND_S).clamp(0.,1.);s2=u2*u2*(3.-2.*u2)
            ref=(self.g0*(1.-s1[:,None])+self.fold*s1[:,None])*(1.-s2[:,None])+self.home[None,:]*s2[:,None]
            # 残差只在站定后生效(t>5s):折腿期纯脚本。依据见 recovery_bounds
            # SITFOLD_RESIDUAL_AFTER_S——旧策略的饱和残差在折腿期会把动作推进坐起盆地。
            ref[:,self.legs]+=(SITFOLD_RESIDUAL_RAD*torch.tanh(actions[:,self.legs])
                *(t>SITFOLD_RESIDUAL_AFTER_S).float()[:,None])
            ref=ref.clamp(lo[None,:],hi[None,:])
            ref[:,self.neck]=ref[:,self.neck].clamp(-np.deg2rad(90),
                np.deg2rad(mdp.NECK_BATTERY_CLEARANCE_DEG))
            target=torch.where(self.script_mask[:,None],ref,target)
        else:
            phi=mdp.recovery_potential(self.env)
            w=((phi-self.band_gate)/(self.band_ramp)).clamp(0,1)[:,None]
            band=torch.deg2rad(torch.tensor(self.band_deg,device=self.env.device)).clamp(min=-3.1)
            banded=torch.minimum(torch.maximum(target,self.home[None,:]-band[None,:]),
                                 self.home[None,:]+band[None,:])
            target=target*(1-w)+banded*w
        super().process_actions((target-self._offset)/self._scale)
        self._raw_actions[:]=actions


@dataclass(kw_only=True)
class SitFoldResidualActionCfg(RecoveryActionCfg):
    def build(self,env):return SitFoldResidualAction(self,env)

def surface_contacts(spec,profile,timeconst=None):
    finalize_contacts(spec)
    for geom in spec.geoms:
        if geom.name.endswith(('left_foot_collision','right_foot_collision')):
            geom.friction=profile;geom.priority=1;geom.condim=6
        if timeconst and geom.name.endswith(('_foot_collision','_floor_proxy')):
            geom.solref=[timeconst,1.]
            if geom.name.endswith('_floor_proxy'):
                geom.priority=1;geom.friction=[1.,.005,.0001]

def build_config(skill='run',envs=64,seed=914,flight_refine=False,recovery_dense=False,
                 ground_tax_after_steps=None,body_clearance=False,quiet_stand=False,coach=False,
                 yaw_hold=False,sitfold_residual=False):
    selected=json.loads((OUT/'selected_contact.json').read_text())
    task,cfg=base_config('run',envs,seed)
    cfg.env.scene.entities['robot'].spec_fn=robot_spec
    cfg.env.scene.spec_fn=partial(surface_contacts,profile=selected['friction'],timeconst=selected.get('normal_contact_timeconst_s'))
    args=vars(cfg.env.actions['joint_pos']).copy()
    # 起身:头部偏航跟随躯干(用户设计意图),不锁世界坐标。走路/蹲/跑不受影响。
    if skill=='getup':args['yaw_follows_trunk']=True
    args['bank_path']=str(OUT/((('getup' if skill=='getup' else 'standing')+'_reset_bank')+selected.get('bank_suffix','')+'.json'))
    if skill=='getup' and 'getup_bank' in selected:args['bank_path']=str(OUT/selected['getup_bank'])
    cfg.env.actions['joint_pos']=(RecoveryActionCfg if skill=='getup' else SquatActionCfg)(**args)
    if skill=='getup' and sitfold_residual:
        cfg.env.actions['joint_pos']=SitFoldResidualActionCfg(**args)
    cfg.env.events['reset_motion_motor']=EventTermCfg(func=mdp.reset_squat_motor_envelope,mode='reset')
    cfg.env.commands['jaw_pose'].ranges=((.04,.04),)
    cfg.env.curriculum={}
    cfg.env.terminations['joint_limit_failure']=TerminationTermCfg(func=limit_failure,time_out=False)
    cfg.env.terminations['ankle_envelope']=TerminationTermCfg(func=mdp.independent_squat_envelope,time_out=False)
    if skill=='run':
        cfg.env.rewards['airborne_running']=RewardTermCfg(func=mdp.motion_clear_running,weight=2.)
        cfg.env.rewards['motion_stop']=RewardTermCfg(func=mdp.motion_stable_stand,weight=4.,params={'stop_only':True})
        cfg.env.terminations['fell_over'].params['limit_angle']=np.deg2rad(30)
        if flight_refine:
            cfg.env.scene.sensors=(*cfg.env.scene.sensors,ContactSensorCfg(name='whole_ground_force',
                primary=ContactMatch(mode='subtree',pattern='trunk_base',entity='robot'),
                secondary=ContactMatch(mode='body',pattern='terrain'),fields=('found','force'),reduce='netforce',num_slots=1))
            cfg.env.rewards['running_sole_lift']=RewardTermCfg(func=mdp.motion_running_sole_lift,weight=8.)
            cfg.env.rewards['airborne_running'].weight=4.
            cfg.env.rewards['track_linear_velocity'].weight=5.
            cfg.env.rewards.pop('motion_stop')
    else:
        cfg.env.terminations['head_pitch_envelope']=TerminationTermCfg(func=mdp.recovery_head_envelope,time_out=False)
        cfg.env.terminations['neck_battery_envelope']=TerminationTermCfg(func=mdp.recovery_neck_battery_envelope,time_out=False)
        # A stalled episode is a failure, not a survivable resting policy: without
        # this the released candidate holds a collapsed pose for the whole episode.
        cfg.env.terminations['recovery_no_progress']=TerminationTermCfg(
            func=mdp.recovery_no_progress,time_out=False)
        # Force-reading sensor for the non-foot floor proxies. The stock
        # `nonfoot_ground` sensor exposes only `found`, and a body that taps the floor
        # for 1-2 ms at a time cannot be graded from a boolean.
        if body_clearance:
            cfg.env.scene.sensors=(*cfg.env.scene.sensors,ContactSensorCfg(name='getup_body_force',
                primary=ContactMatch(mode='geom',pattern='.*_floor_proxy',entity='robot'),
                secondary=ContactMatch(mode='body',pattern='terrain'),
                fields=('found','force'),reduce='netforce',num_slots=1))
        # A fallen body may touch the floor and start below walking clearance.
        # Joint and numerical limits remain. No within-episode pose writes.
        for key in ('nonfoot_contact','fell_over','local_clearance'):
            cfg.env.terminations.pop(key,None)
        cfg.env.commands['twist']=mdp.IndependentSquatCommandCfg(width=3,resampling_time_range=(100.,100.))
        # Width 3 command produces zeros: get-up mode is the selected ONNX.
        cfg.env.commands['body_pose'].skill='run'
        cfg.env.episode_length_s=12.
        keep=('dof_pos_limits','action_rate_l2','arm_pose_tracking','tail_pose_tracking')
        cfg.env.rewards={k:v for k,v in cfg.env.rewards.items() if k in keep}
        # Get-up needs large fast limb motion; the walking-scale rate penalty only
        # discourages the exploration this task depends on.
        cfg.env.rewards['action_rate_l2'].weight=-.005
        for key in ('arm_pose_tracking','tail_pose_tracking'):cfg.env.rewards[key].weight=.02
        # Potential-based progress: the only term with gradient across the full
        # fallen-to-standing rotation. Absolute posture is kept small because it is
        # already ~0.14 at the 100 deg starts and is farmable by lying still.
        cfg.env.rewards['recovery_progress']=RewardTermCfg(func=mdp.motion_recovery_progress,weight=20.,
            params={'tolerance':.02,'stuck_seconds':3.})
        cfg.env.rewards['recovery_posture']=RewardTermCfg(func=mdp.motion_recovery_posture_dense if recovery_dense else mdp.motion_recovery_posture,weight=.5)
        cfg.env.rewards['recovery_stand']=RewardTermCfg(func=mdp.motion_stable_stand,weight=8.)
        # Body-on-floor tax, ramped in. Measured block at iteration 17000: three of
        # four fall directions park in a propped pose (tilt 73 deg, body still on the
        # floor for 100% of the final 2 s) that banks most of the potential without
        # ever loading a foot. Off for the first 2M steps so the get-up motion is
        # learned before the constraint that rules out the propped shortcut arrives.
        if ground_tax_after_steps is not None:
            cfg.env.rewards['recovery_body_ground_tax']=RewardTermCfg(
                func=mdp.motion_recovery_body_ground_tax,weight=-4.,
                params={'delay_steps':ground_tax_after_steps,'ramp_steps':2_000_000,'upright_gate':.6})
        # Body-clearance shaping, two halves that must be read together.
        #
        # (a) Dense cost on non-foot floor force above the acceptance deadband. The gate
        #     is literally "body support < 0.2 N", so the cost is the gate itself. It
        #     preserves the right ordering -- clean stand (+8.5/s) > tail-down stand
        #     (>=-14/s) > lying (>=-32/s) -- without letting the unavoidable floor
        #     contact of the get-up phase dominate the return, which is why the weight
        #     sits at 3 rather than the 8 first tried.
        # (b) A gated bonus for holding the tail up. Measured: the standing pose parks
        #     tail_pitch at -47.3 deg against its -51.6 deg limit, using the tail as a
        #     third contact point; -10 deg would give 109 mm of clearance. Tail lift is
        #     the single directly observable hinge on the sub-gate that stayed at
        #     0.84-0.91 through three rounds of penalty-only shaping. A bonus cannot
        #     make standing itself unattractive the way an aggressive penalty can.
        if body_clearance:
            cfg.env.rewards['recovery_clear_body']=RewardTermCfg(
                func=mdp.motion_recovery_body_force_cost,weight=3.,
                params={'deadband_n':.2,'upright_gate':.6})
            # weight 2 and gate 0.9 are both deliberate. A first attempt at weight 6 /
            # gate 0.6 lifted the tail but collapsed the standing reward from 3.5 to
            # 0.02: the gate at 0.6 admits poses that are up but not standing, so the
            # easy +6/s tail bonus outbid the hard +8/s standing jackpot. At 0.9 the
            # bonus is only reachable from a real stand, and at 2.0 the dominant prize
            # stays the standing jackpot -- which requires stillness, so balance is
            # protected. Measured target: tail contact duty cycle must fall from
            # 15-19% to under 5% of the final 2 s.
            # 站定后的残余运动。iter23100 八例的绑定判据全是角速(4-19%);iter21100 起
            # 抬尾那一轮已有 2/8 例翻到 平速/角速 91%。`motion_stable_stand` 里
            # exp(-|v|^2/.0025-|w|^2/.25) 被整条合取乘掉了,在 91% 处≈0、没有梯度,
            # 所以单列一项、只保留"已经站起来"这一个门,再对阻尼平滑付酬。
            cfg.env.rewards['recovery_quiet_stand']=RewardTermCfg(
                func=mdp.motion_recovery_quiet_stand,weight=4.,
                params={'upright_gate':.6,'omega_scale':.3,'speed_scale':.05})
        # 示范奖励(教练)。用户看视频指出"根本没有回到正确的站立姿态":四个方位
        # 站起来的是一个深度蜷缩姿态,与 v7 HOME 站姿逐关节差 20-106°(髋俯仰
        # -86° vs -26°、膝 -86° vs 0°、踝顶满 ±51° 限位、neck -87° vs +20°、
        # head_yaw +105°、tail -47°、双臂张开)。十二项验收判据量不出姿势本身
        # ——蜷缩躯干的根高恰好落进 103.5-125.5mm 窗。参考 = contract 的 HOME
        # (v7 站姿,策略观测里本来就有 joint_pos,无特权输入);pose 分给靠拢
        # 梯度(仅任务后半段),hold 分只在"正确站姿+真站稳"时发,结构与已验证的
        # squat_tracking 同型。
        if coach:
            coach_home=json.loads((OUT/selected.get('plant_directory','normal_contact_plant')/'contract.json').read_text())['action_offset'][0]
            # 工资核算(2026-09-16):权 2 时"站对姿势的全部潜在收益"≈0.4/回合,
            # 而蜷缩站姿已稳拿 recovery_stand≈3.3/s —— 学新姿势不划算,iter23000
            # 实测 19 关节平均偏差纹丝不动(43°→42°)。改权 12 + 2.5s 开课(起身
            # 只需 ~1s):站对的回报≈114/回合,压倒既得收益;当前姿态下也开始有
            # 实质分(≈0.34/回合),不再淹没在 advantage 噪声里。hold 分不动,
            # 它是"修姿势不许赔站稳"的保险。
            # 第⑩轮参数:pose 全程计分(pose_free,起身期也给"往 HOME 靠"的梯度)
            # + hold×姿势门(hold_gate=.3):站得再稳,姿势分不到 0.3 就没有 hold。
            # 上一版(iter24600 实测)85% 教练分来自 hold,蜷缩站立白拿保底工资,
            # 伸腿净亏 → 86° 腿行程一步不动。
            cfg.env.rewards['recovery_coach']=RewardTermCfg(
                func=mdp.motion_recovery_coach,weight=12.,
                params={'home_pose':coach_home,'pose_std_rad':.25,'hold_std_rad':.35,
                        'coach_after_s':2.5,'hold_weight':6.,
                        'hold_gate':.25,'pose_free':True,'pose_soft':.15})
            # 朝向保持:摁住躯干偏航,一次修掉 朝向判据 + head_yaw 偏差 + "头朝后"
            if yaw_hold:
                # 配平(铁律 5:奖励顺序必须是 干净站立 > 躺平)。
                # 上一版把权重压在 −3 的理由是:当时 coach 只有 1.9/s,−8 会让该项达
                # −8.58/s,与站起来的全部收益(stand 8 + coach 1.9 + quiet 2 ≈ 12/s)相当,
                # 躺平不动更划算。**那条约束现在失效了**:第⑯轮实测 coach 已经涨到
                # +29.05/s(静立 3.07、站立 2.46),站起来的全部收益 ≈ 34.96/s。
                # 按当前实测重新配平:权重 −10 / 死区 20° 时,
                #   零漂移站立 = +34.96/s(最优);
                #   114° 漂移站立 = 34.96 − 10×(1.99−0.35) = +18.6/s ≫ 躺平 0/s。
                # 顺序保持(站 > 躺,且干净站 > 歪着站 16/s),同时给偏航一个真梯度。
                # 依据:实测偏航罚原先只占总量 8.5%,策略心甘情愿付钱换站起来;
                # 偏航 100% 丢在起身的 1.3~1.9s 里(站定后 6s 只动 ±0.6°)。
                cfg.env.rewards['recovery_yaw_hold']=RewardTermCfg(
                    func=mdp.motion_recovery_yaw_hold,weight=-10.,params={'deadband_deg':20.})
            cfg.env.rewards['recovery_tail_lift']=RewardTermCfg(
                func=mdp.motion_recovery_tail_lift,weight=2.,
                params={'upright_gate':.9,'target_deg':0.,'floor_deg':-50.})
            # 把锁加在**大的那一项**上:recovery_stand(实测 3.8/s)是蜷缩站姿的稳定
            # 收入,前三次只给小的 hold 上锁(≈2.3/s)所以推不动。换成带姿势门的版本,
            # 蜷缩站立的收入才真正归零 —— 只剩"躺平 0 / 直腿站 8+"两个选项。
            cfg.env.rewards['recovery_stand']=RewardTermCfg(
                func=mdp.motion_recovery_gated_stand,weight=8.,
                params={'home_pose':coach_home,'pose_gate':.25,
                        'pose_std_rad':.25,'pose_soft':.15})
    cfg.agent.experiment_name='microdinosaur_contact_'+skill
    return task,cfg
