"""Current joint-limited FK poses, explicitly not a collision or dynamics test."""
import json
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'20260914_pose_preview'


def main():
    OUT.mkdir(exist_ok=True)
    plant=ROOT/'20260914_run_jump/plant'
    model=mujoco.MjModel.from_binary_path('x.mjb',assets={'x.mjb':(plant/'nominal.mjb').read_bytes()})
    data=mujoco.MjData(model);contract=json.loads((plant/'contract.json').read_text())
    names=contract['action_names'];jids=[model.joint('robot/'+n).id for n in names]
    adr=model.jnt_qposadr[jids];home=np.array(contract['action_offset'][0])
    row=next(x for x in json.loads((ROOT/'20260914_fold_recovery/fold_geometry.json').read_text()) if x['depth_mm']==50)
    folded=np.array(row['target']);roll=names.index('head_roll')
    cases=[('站立参考',home,.117182,90),('5cm深蹲的关节参考',folded,.067182,90)]
    for deg in [-20,20]:
        q=home.copy();q[roll]=np.deg2rad(deg);cases.append((f'头部横滚 {deg:+d}°',q,.117182,0))
    renderer=mujoco.Renderer(model,width=624,height=432);panels=[]
    option=mujoco.MjvOption();option.geomgroup[3]=0
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)
    small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    for index,(title,q,height,azimuth) in enumerate(cases):
        assert np.all(q>=model.jnt_range[jids,0]) and np.all(q<=model.jnt_range[jids,1])
        mujoco.mj_resetData(model,data);data.qpos[:3]=[0,0,height];data.qpos[3:7]=[1,0,0,0];data.qpos[adr]=q
        mujoco.mj_forward(model,data)
        camera=mujoco.MjvCamera();camera.azimuth=azimuth
        camera.elevation=-7 if index<2 else -2;camera.distance=.68 if index<2 else .40
        camera.lookat[:]=[-.02,0,.137] if index<2 else data.xpos[model.body('robot/jaw_soft').id]
        renderer.update_scene(data,camera=camera,scene_option=option)
        image=Image.fromarray(renderer.render());draw=ImageDraw.Draw(image)
        draw.rectangle((0,0,624,44),fill=(22,29,34));draw.text((14,7),title,font=font,fill='white')
        draw.rectangle((0,396,624,432),fill=(22,29,34))
        footer='仅关节几何预览 · 未验证碰撞、线束或动力'
        draw.text((14,405),footer,font=small,fill='white');panels.append(image)
    renderer.close();canvas=Image.new('RGB',(1248,864))
    for i,panel in enumerate(panels):canvas.paste(panel,((i%2)*624,(i//2)*432))
    canvas.save(OUT/'fold_head.png')
    result=dict(kind='Forward kinematic pose preview only',dynamics_evaluated=False,collision_validated=False,
        head_roll_range_deg=np.rad2deg(model.jnt_range[jids[roll]]).tolist(),
        shown_roll_deg=[-20,20],fold_ik_reference_depth_mm=50,
        fold_projected_knee_angles=row['knee_angles'],fold_ik_foot_error_mm=row['foot_position_error_mm'],
        limiting_joints=row['limiting_joints'],policy_modified=False)
    (OUT/'pose_info.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Saved pose preview; geometry only.')


if __name__=='__main__':main()
