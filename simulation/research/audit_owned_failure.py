"""Inspect preserved pre-fault states without treating them as an exact replay."""
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np


def main():
    dest=Path(__file__).parent/'20260914_imu_owned_terrain/failure'
    model=mujoco.MjModel.from_binary_path('original_model.mjb',assets={
        'original_model.mjb':(dest/'original_model.mjb').read_bytes()})
    data=mujoco.MjData(model)
    with np.load(dest/'original_nan_dump.npz',allow_pickle=True) as z:
        meta=z['_metadata'].item()
        keys=sorted(k for k in z.files if k.startswith('states_step_'))
        positions=[];velocities=[]
        for k in keys:
            assert np.isfinite(z[k]).all()
            mujoco.mj_setState(model,data,z[k][0],meta['state_spec'])
            positions.append(data.qpos.copy());velocities.append(data.qvel.copy())
    q=np.asarray(positions);v=np.asarray(velocities)
    r=dict(status='DIAGNOSTIC_INCOMPLETE',original_metadata=meta,
        pre_fault_states=len(keys),all_pre_fault_qpos_qvel_finite=True,
        root_quaternion_norm_range=[float(x) for x in (np.linalg.norm(q[:,3:7],axis=1).min(),np.linalg.norm(q[:,3:7],axis=1).max())],
        max_abs_root_linear_velocity_m_s=float(abs(v[:,:3]).max()),
        max_abs_root_angular_velocity_rad_s=float(abs(v[:,3:6]).max()),
        max_abs_joint_velocity_rad_s=float(abs(v[:,6:]).max()),
        first_root_position_m=q[0,:3].tolist(),last_root_position_m=q[-1,:3].tolist(),
        post_fault_fields_available=False,randomized_model_fields_available=False,
        original_numeric_root_cause_resolved=False,
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.glob('original_*')})
    (dest/'audit.json').write_text(json.dumps(r,indent=2))
    (dest/'FINDINGS.md').write_text(f'''# 原训练中断与恢复边界

首次进程完成217个PPO更新后，env330在第83665个物理保护检查步触发非有限数检查。原保护先写入NPZ和MJB，再因Windows符号链接权限错误退出。原运行保持FAILED，未抹去失败记录。

已用保存模型的mj_setState读取100帧异常前状态，qpos/qvel均有限，根四元数范数范围为{r['root_quaternion_norm_range']}；根线速度最大绝对分量{r['max_abs_root_linear_velocity_m_s']:.6f}m/s、角速度最大绝对分量{r['max_abs_root_angular_velocity_rad_s']:.6f}rad/s，关节速度最大绝对分量{r['max_abs_joint_velocity_rad_s']:.6f}rad/s。这些状态并不能说明触发瞬间正常。

原转储缺少异常后的qacc、qacc_warmstart和sensordata，且编译模型不包含每环境随机化字段，无法从现有证据确认首先失效的字段或数值根因，也无法精确复演故障。没有以“无明显爆速”推导“假警报”。

本研究入口新增直接写入NPZ/MJB/JSON的保护，保存触发时六组字段并立即抛出FloatingPointError；不创建符号链接，不改系统权限，不改安装包文件，不跳过坏环境。故障注入测试验证了写入与中止行为，额外64×5短训通过。

从最近有限的model_16350.pt保留201个更新，再补100个更新，最终保留301个更新（3,698,688条转移）。原进程另外16个更新未保存、已丢弃；实际两个正式进程共记录317个更新（3,895,296条转移），短训另计。续跑重建环境和随机采样，不能称为同一物理/RNG轨迹精确续跑。100轮续跑完成且未再次触发保护，只能证明这次续跑通过，不能证明根因消失。

机器可读证据见audit.json；原转储、编译模型和旧保护源代码保留在本目录。训练计数、检查点有限性、实际学习率与正式导出对照见../training_audit.json。
''',encoding='utf-8')
    print(json.dumps(r,indent=2))


if __name__=='__main__':main()
