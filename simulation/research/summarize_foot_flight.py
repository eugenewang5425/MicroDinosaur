"""Feet-first report. Reject airborne feet caused by falls or body support."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from foot_flight_cfg import OUT


def summarize(rows,directory):
    m=[]
    for row in rows:
        if row['status']!='COMPLETE':continue
        item=dict(row['metrics']);a=np.load(directory/(row['name']+'.npz'))['physics']
        gap=a[:,14:16].min(-1)
        valid=(a[:,0]>=1.8)&(a[:,0]<2.8)&(a[:,4]<.05)&(a[:,8]<30)&(a[:,13]<=.02)
        item['valid_peak_gap_mm']=float(max(0.,gap[valid].max() if valid.any() else 0.)*1000)
        m.append(item)
    out=dict(planned=len(rows),executed=len(m),rejected=len(rows)-len(m))
    for key in ['feet_goal','deep_crouch_achieved','feet_and_landing_success','complete_deep_tuck_jump',
                'settled_final_second','fell','nonfoot_ground_contact']:
        out[key]=sum(bool(x[key]) for x in m)
    out['joint_limit_failures']=sum(x['joint_limit_excess_rad']>.02 for x in m)
    for key,factor in [('valid_peak_gap_mm',1),('continuous_5mm_s',1000),('actual_crouch_m',1000),
                       ('final_second_speed_m_s',1000),('camera_heading_error_rms_deg',1),('camera_roll_rms_deg',1),
                       ('camera_pitch_rms_deg',1),('maximum_mesh_penetration_m',1000)]:
        values=np.array([x[key]*factor for x in m]);out[key]=dict(mean=float(values.mean()),min=float(values.min()),max=float(values.max())) if len(values) else None
    return out


def main():
    data={label:json.loads((OUT/'evaluation'/label/'matrix.json').read_text()) for label in ['previous','candidate']}
    result={};lines=['# 双足腾空确认', '', '主要目标：同一次两足最低点>5mm持续≥60ms，期间双足最低净空峰值≥10mm，无身体借地支撑且姿态/限位合格。质心上升不决定此目标。', '',
        '净空只统计起跳时窗内无支撑、倾斜<30°和未越限的样本；翻倒抬脚不会计入有效净空。失败启动保留分母，数值统计只来自已执行记录。', '',
        '| 模型 | 请求蹲深 | 执行方式 | 脚底目标 | 脚底目标+落稳 | 完整深蹲弹跳 | 实际蹲深 mm | 有效双足净空 mm | 连续>5mm ms | 残余速度 mm/s |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for label,rows in data.items():
        result[label]={}
        for depth in [.03,.05]:
            for ret in [False,True]:
                subset=[r for r in rows if r['requested_depth_m']==depth and r['return_to_base']==ret and r['seed']!=604]
                s=summarize(subset,OUT/'evaluation'/label);result[label][f'c{round(depth*1000)}_return{int(ret)}']=s
                n=s['planned'];h=s['valid_peak_gap_mm'];d=s['actual_crouch_m'];f=s['continuous_5mm_s'];v=s['final_second_speed_m_s']
                lines.append(f"| {label} | {depth*1000:.0f}mm | {'交回v7' if ret else '专家单独'} | {s['feet_goal']}/{n} | "
                    f"{s['feet_and_landing_success']}/{n} | {s['complete_deep_tuck_jump']}/{n} | {d['mean']:.2f} | "
                    f"{h['mean']:.2f}（{h['min']:.2f}–{h['max']:.2f}） | {f['mean']:.2f}（{f['min']:.2f}–{f['max']:.2f}） | {v['mean']:.2f} |")
        for row in rows:
            if row['seed']==604:result[label][row['name']]=summarize([row],OUT/'evaluation'/label)
    lines+=['', '## 压力条件', '', '| 模型 | 条件 | 执行方式 | 脚底目标 | 完整深蹲弹跳 | 有效峰值净空 mm | 最长>5mm ms |', '|---|---|---|---|---|---:|---:|']
    for label,rows in data.items():
        for row in rows:
            if row['seed']!=604:continue
            s=result[label][row['name']];h=s['valid_peak_gap_mm'];t=s['continuous_5mm_s']
            lines.append(f"| {label} | {row['hardware']['name']} | {'交回v7' if row['return_to_base'] else '专家单独'} | {s['feet_goal']} | {s['complete_deep_tuck_jump']} | {h['mean']:.2f} | {t['mean']:.2f} |" if h else f"| {label} | {row['hardware']['name']} | 拒绝 | 0 | 0 | — | — |")
    (OUT/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (OUT/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for column,depth in enumerate([30,50]):
        for label,color in [('previous','#697787'),('candidate','#137c71')]:
            a=np.load(OUT/'evaluation'/label/f'c{depth}_d10_s601_s288_protocol_return0.npz')['physics'];mask=(a[:,0]>=1.5)&(a[:,0]<=2.7)
            axes[0,column].plot(a[mask,0],a[mask,14:16].min(-1)*1000,label=label,color=color)
            axes[1,column].plot(a[mask,0],a[mask,4],label=label,color=color)
        axes[0,column].set_title(f'{depth} mm crouch request / same seed 601, delay 10 ms')
        axes[0,column].axhline(5,ls='--',color='#a6653d',label='5 mm duration threshold')
        axes[0,column].axhline(10,ls=':',color='#707070',label='10 mm peak target')
        axes[0,column].set_ylabel('Minimum of both complete foot meshes [mm]')
        axes[1,column].set_ylabel('Total ground support [N]')
        for ax in axes[:,column]:ax.grid(alpha=.2);ax.set_xlabel('Simulation time [s]');ax.legend(fontsize=8)
    fig.suptitle('Feet clearance and real support; COM is not the jump score')
    fig.savefig(OUT/'feet_comparison.png',dpi=160);plt.close(fig)
    print(json.dumps({label:{key:{k:s[k] for k in ['planned','executed','feet_goal','complete_deep_tuck_jump','fell']}
        for key,s in groups.items() if key in ['c30_return0','c50_return0','c50_return1']} for label,groups in result.items()},indent=2))


if __name__=='__main__':main()
