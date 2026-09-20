"""Render the exact measured CAD intersection, with a simplified display mesh."""
import sys,json
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image,ImageDraw,ImageFont
OUT=Path(__file__).resolve().parent/'20260914_compact_fold';sys.path.insert(0,str(OUT/'tools'))
import manifold3d as md
rows=json.loads((OUT/'collision_parts_indexed.json').read_text());data=np.load(OUT/'collision_parts_indexed.npz')
frames={r['name']:r['bodies'] for r in json.loads((OUT/'cad_pose_frames.json').read_text())}
cases=[('左踝：5 cm 收腿','d50_t5','S288_left_ankle_CASE','ankle_left__032'),
       ('髋部：3 cm 收腿','d30_t5','hip_l__018','S288_left_hip_pitch_OUT_1_M2x5_0')]
canvas=Image.new('RGB',(1200,620));font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',23)
small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
for index,(label,frame,a,b) in enumerate(cases):
    solids=[]
    for name in (a,b):
        r=next(x for x in rows if x['name']==name);k=r['key']
        solid=md.Manifold(md.Mesh((data[k+'_v']*1000).astype(np.float32),data[k+'_f']))
        transform=np.array(frames[frame][r['body']])[:3];transform[:,3]*=1000
        solids.append(solid.transform(transform))
    cross=solids[0]^solids[1];solids.append(cross)
    spec=mujoco.MjSpec();spec.visual.global_.offwidth=600;spec.visual.global_.offheight=620
    spec.worldbody.add_light(pos=[0,-.2,.5],dir=[0,1,-1],diffuse=[.9,.9,.9],ambient=[.5,.5,.5])
    colors=[[.50,.65,.75,.32],[.65,.72,.77,.32],[1.,.05,.03,1.]]
    for j,solid in enumerate(solids):
        view=solid.simplify(.03).to_mesh() if j<2 else solid.to_mesh()
        spec.add_mesh(name=f'p{j}',uservert=(view.vert_properties[:,:3]/1000).ravel().tolist(),userface=view.tri_verts.ravel().tolist())
        spec.worldbody.add_geom(name=f'p{j}',type=mujoco.mjtGeom.mjGEOM_MESH,meshname=f'p{j}',density=0,
                               rgba=colors[j],contype=0,conaffinity=0)
    m=spec.compile();d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    renderer=mujoco.Renderer(m,width=600,height=620)
    cam=mujoco.MjvCamera();box=np.array(cross.bounding_box()).reshape(2,3)
    cam.lookat[:]=box.mean(0)/1000;cam.distance=.095 if index==0 else .055
    cam.azimuth=110;cam.elevation=-15
    renderer.update_scene(d,camera=cam);panel=Image.fromarray(renderer.render());renderer.close()
    draw=ImageDraw.Draw(panel);draw.rectangle((0,0,600,84),fill=(20,29,38))
    draw.text((14,10),label,font=font,fill='white')
    draw.text((14,49),f'红色：实体交叠 {cross.volume():.2f} mm³',font=small,fill=(255,140,120))
    draw.rectangle((0,550,600,620),fill=(20,29,38))
    draw.text((12,559),a,font=small,fill='white');draw.text((12,586),b,font=small,fill='white')
    canvas.paste(panel,(600*index,0))
canvas.save(OUT/'cad_interference.png')
