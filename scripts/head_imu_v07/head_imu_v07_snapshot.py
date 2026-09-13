"""Freeze the current assembly, never a historical generator's input."""
import bpy, json, hashlib, shutil, sys
import numpy as np
from pathlib import Path
from mathutils import Vector
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from reference_pose import set_reference
P=R/'work_in_progress/head_imu_v07'
H='656665835b6ea6c170a2648cca2630ce6dbc186811fbceea633f3260b9ebceb4'
assert hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest()==H
assert not (P/'basis.json').exists(),'Snapshot already frozen'
P.mkdir(parents=True,exist_ok=True)
set_reference()
rows={};a={}
for o in bpy.data.objects:
 r=dict(parent=o.parent.name if o.parent else None,matrix=[list(x) for x in o.matrix_world],type=o.type,collections=[c.name for c in o.users_collection],props={k:str(o[k]) for k in o.keys()},constraint_count=len(o.constraints))
 r['drivers']=[dict(path=f.data_path,index=f.array_index,expression=f.driver.expression,variables=[dict(name=v.name,type=v.type,targets=[dict(id=t.id.name if t.id else None,data_path=t.data_path) for t in v.targets]) for v in f.driver.variables]) for f in o.animation_data.drivers] if o.animation_data else []
 if o.type=='MESH':
  o.data.calc_loop_triangles();f=np.array([tuple(t.vertices) for t in o.data.loop_triangles],np.int32);v=np.array([q.co[:] for q in o.data.vertices],np.float32)
  r['local_mesh_hash']=hashlib.sha256(v.tobytes()+f.tobytes()).hexdigest();r['materials']=[m.name for m in o.data.materials]
  w=np.array([tuple(o.matrix_world@q.co) for q in o.data.vertices])*1000;r['bounds_mm']=[w.min(0).tolist(),w.max(0).tolist()]
  if '09_STUDIO' not in r['collections']:a[o.name+'__v']=w;a[o.name+'__f']=f
 rows[o.name]=r
assert not (P/'basis.json').exists(),'Snapshot already frozen'
np.savez_compressed(P/'basis_meshes.npz',**a)
(P/'basis.json').write_text(json.dumps(dict(source=bpy.data.filepath,source_sha256=H,objects=rows),indent=2))
shutil.copy2(Path(bpy.data.filepath),P/'BEFORE_HEAD_IMU_RETIRED_INPUT.blender')
for n in ['model_manifest.json','delivery_status.json','mass_estimate.json','parts_mass_estimate.csv','battery_design_target.json','s288_joint_bus_map.json','fastener_interface_schedule.csv']:
 shutil.copy2(R/'current'/n,P/('source_'+n))
for n in ['CURRENT_DESIGN.md','AGENTS.md','MICRODINOSAUR_V1_REVIEW.md','ELECTRONICS_CAMERA_IMU_PROPOSAL_20260911.md']:
 shutil.copy2(R/n,P/('source_'+n))
bpy.data.libraries.write(str(P/'BEFORE_HEAD_LOCAL_NOT_CURRENT.blend'),{o for o in bpy.data.objects if o.parent and o.parent.name=='BODY_jaw_soft'},fake_user=True,compress=True)
s=bpy.context.scene;s.render.engine='BLENDER_WORKBENCH';s.render.resolution_x=1400;s.render.resolution_y=1100;s.render.resolution_percentage=100
s.display.shading.light='STUDIO';s.display.shading.color_type='MATERIAL';s.display.shading.show_cavity=True;s.display.shading.cavity_type='BOTH';s.display.shading.background_type='WORLD';s.world.color=(.72,.75,.78);s.view_settings.view_transform='Standard';s.view_settings.look='None';s.view_settings.exposure=.35
head={n for n,r in rows.items() if r['parent']=='BODY_jaw_soft' and r['type']=='MESH'}
def render(name,names,off):
 for o in bpy.data.objects:
  if o.type=='MESH':o.hide_render=o.name not in names
 t=Vector((.027,0,.245));cam=s.camera;cam.location=t+Vector(off);cam.rotation_euler=(t-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.type='ORTHO';cam.data.ortho_scale=.155;s.render.filepath=str(P/(name+'.png'));bpy.ops.render.render(write_still=True)
render('before_head_open',head-{n for n in head if 'top_head_shell' in n},(-.25,-.30,.35))
render('before_carrier',{'Rex_Camera_Pi_Carrier','V10_PI_ZERO_2W_PCB'},(-.25,-.30,.35))
print('FROZEN',len(rows),flush=True)
