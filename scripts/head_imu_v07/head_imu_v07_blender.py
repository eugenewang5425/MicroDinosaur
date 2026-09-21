import bpy,bmesh,json,hashlib,sys,csv,argparse
import numpy as np
from pathlib import Path
from mathutils import Matrix,Vector
R=Path(__file__).resolve().parents[2];P=R/'work_in_progress/head_imu_v07'
sys.path.insert(0,str(Path(__file__).resolve().parent))
from reference_pose import set_reference
parser=argparse.ArgumentParser(description='v07 recipe: render current, or build/verify with local historical evidence.')
parser.add_argument('mode',choices=['build','verify','render'])
parser.add_argument('--output',type=Path,default=R/'renders/head_imu_v07')
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
mode=args.mode
if mode!='render' and not (P/'basis.json').is_file():
 raise SystemExit('Build/verify require excluded local frozen inputs. Use scripts/verify_current.py to verify this latest-only checkout.')
B=json.loads((P/'basis.json').read_text()) if mode!='render' else None
D=json.loads((P/'delta.json').read_text());A=np.load(P/'delta_meshes.npz');candidate=P/'MicroDinosaur_v1.blender'
set_reference()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(n,x):(P/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
flags=['hardware_enable','hardware_enabled','electrical_release','manufacturing_release','continuous_collision_release','full_motion_range_release','gait_release','valid_com','strength_release','print_release']
def assign_world(o):
 o.parent=bpy.data.objects['BODY_jaw_soft'];o.matrix_basis=Matrix.Identity(4);o.matrix_parent_inverse=o.parent.matrix_world.inverted()
if mode=='build':
 assert sha(Path(bpy.data.filepath))==B['source_sha256']
 q=json.loads((P/'static_checks.json').read_text());assert not q['new_intersections'] and not q['tool_hits'] and not any(r['hits'] for r in q['nut_insertion_paths']+q['reservations'])
 mats={}
 for key,color in [('PA12',(.53,.54,.56,1)),('PCB',(.035,.32,.18,1)),('CAP',(.58,.61,.65,1)),('STEEL',(.29,.33,.37,1))]:
  m=bpy.data.materials.get('JY_HEAD_'+key) or bpy.data.materials.new('JY_HEAD_'+key);m.diffuse_color=color;mats[key]=m
 col=bpy.data.collections.new('26_ACTIVE_HEAD_JY61P');bpy.context.scene.collection.children.link(col)
 mass=json.loads((P/'source_mass_estimate.json').read_text());by={r['name']:r for r in mass['rows']}
 for n,p in D['parts'].items():
  me=bpy.data.meshes.new(n+'_v07');me.from_pydata((A[n+'__v']*.001).tolist(),[],A[n+'__f'].tolist());me.update()
  if n in D['replaced']:
   o=bpy.data.objects[n]
   for mat in o.data.materials:me.materials.append(mat)
   o.data=me
  else:
   o=bpy.data.objects.new(n,me);col.objects.link(o)
   mk='PCB' if 'PCB' in n else 'CAP' if 'CAP' in n else 'STEEL' if 'M2' in n else 'PA12';me.materials.append(mats[mk])
  assign_world(o)
  o['part_status']='ACTIVE';o['geometry_revision']='MicroDinosaur_v1_HEAD_IMU_v07';o['revision_reference']='HEAD_IMU_V07_REVIEW.md'
  o['print_part']=n in D['replaced'] or 'CLIP' in n
  o['actual_received_hardware_measured']=False;o['manufacturing_release']=False;o['actual_header_fit_verified']=False
  o['nominal_board_mm']=[15.24,15.24,2.8];o['dimension_source']=D['drawing_url']
  o['mount_note']='Head-rigid carrier-integrated upper shelf; no drilling PCB; two short-edge clamps and M2 through bolts/open-bottom hex nut pockets. Standard JY61P drawing baseline; enhanced shield/headers not measured.'
  if 'M2' in n:o['thread_representation']='Nominal M2 screw/nut envelopes; actual thread and tightening torque unverified'
  v=A[n+'__v'].astype(float);f=A[n+'__f'];tr=v[f];vols=np.einsum('ij,ij->i',tr[:,0],np.cross(tr[:,1],tr[:,2]))/6;vol=vols.sum();center=(vols[:,None]*tr.sum(1)/4).sum(0)/vol
  if 'PCB' in n:matname,density='FR4/copper PCB envelope estimate',1.85
  elif 'CAP' in n:matname,density='Shield/components mixed envelope estimate',2.
  elif 'M2' in n:matname,density='Steel estimate; hardware grade not verified',7.85
  else:matname,density='PA12',1.02
  r=dict(name=n,parent='BODY_jaw_soft',material_design=matname,mass_estimate_g=abs(vol)*density/1000,uncertainty_fraction=.5 if 'ENVELOPE' in n else .25,method='Nominal CAD volume x assumed material density; not sliced or weighed',volume_mm3=abs(vol),center_mm=center.tolist(),watertight=True,measured=False,material_verified_by_procurement=False)
  by[n]=r;o['mass_estimate_g']=r['mass_estimate_g'];o['material_design']=matname;o['mass_status']='MODEL_ESTIMATE_UNWEIGHED_INCOMPLETE_ELECTRICAL_BOM';o['mass_method']=r['method']
  if 'mass_kg' in o:o['mass_kg']=r['mass_estimate_g']/1000
  if 'com_local_m' in o:o['com_local_m']=o.matrix_world.inverted()@(Vector(center)*.001)
 # Parent frame for simulation extrinsics. This is a coordinate Empty, not a new part.
 frame=bpy.data.objects.new('FRAME_JY61P_HEAD_SENSOR',None);col.objects.link(frame);frame.parent=bpy.data.objects['BODY_jaw_soft'];frame.matrix_world=Matrix.Translation(Vector([35,0,260.8])*.001)
 frame.empty_display_type='ARROWS';frame.empty_display_size=.007;frame['part_status']='SENSOR_FRAME_NOT_PART';frame['mass_estimate_g']=0.;frame['axis_convention']='Provisional PCB X=head forward, Y=head left, Z=up at reference; package axes must be verified and calibrated';frame['actual_sensor_die_offset_known']=False
 reservecol=bpy.data.collections.get('90_DESIGN_RESERVATIONS_NOT_PARTS')
 for n,r in D['reserves'].items():
  o=bpy.data.objects.new(n,None);reservecol.objects.link(o);o.parent=bpy.data.objects['BODY_jaw_soft'];o.matrix_world=Matrix.Translation(Vector(r['center_mm'])*.001);o.scale=Vector(r['size_mm'])*.0005;o.empty_display_type='CUBE';o.empty_display_size=1
  o['part_status']='DESIGN_RESERVATION_NOT_PART';o['material_design']='NOT_A_PHYSICAL_PART';o['mass_estimate_g']=0.;o['note']=r['role']
 for me in list(bpy.data.meshes):
  if me.users==0:bpy.data.meshes.remove(me)
 mass['rows']=list(by.values());total=sum(r['mass_estimate_g'] for r in mass['rows']);com=sum(np.array(r['center_mm'])*r['mass_estimate_g'] for r in mass['rows'])/total
 mass['mass_estimate_g']=total;mass['com_mm']=com.tolist();mass['com_fore_aft_offset_mm']=com[0]-mass['foot_bbox_center_mm'][0];mass['status']='Head JY61P nominal mounting added; incomplete electrical BOM; unweighed'
 mass['not_included']='Remaining dual bus adapters, distribution/fuses/estop/regeneration, real harness/USB/SD and second buck. Both head and body nominal JY61P envelopes are included; actual headers/wiring are not. Charger/spare batteries external.'
 groups={}
 for r in mass['rows']:groups[r['material_design']]=groups.get(r['material_design'],0)+r['mass_estimate_g']
 mass['material_groups_g']=groups;dump('mass_estimate.json',mass)
 fields=list(dict.fromkeys(k for r in mass['rows'] for k in r))
 with (P/'parts_mass_estimate.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(mass['rows'])
 s=bpy.context.scene
 for k in flags:s[k]=False
 s['variant']='MicroDinosaur_v1 with head and body JY61P nominal modules; v07 mechanical review';s['review_document']='HEAD_IMU_V07_REVIEW.md';s['mass_estimate_g']=total;s['com_estimate_mm']=com.tolist();s['modeled_imu_count']=2;s['head_imu_sensor_frame']='FRAME_JY61P_HEAD_SENSOR'
 import importlib.util
 spec=importlib.util.spec_from_file_location('head_imu_metadata',Path(__file__).with_name('head_imu_v07_scene_metadata.py'));meta=importlib.util.module_from_spec(spec);spec.loader.exec_module(meta);meta.clean_scene(bpy)
 bpy.context.view_layer.update();bpy.context.preferences.filepaths.save_version=0;bpy.ops.wm.save_as_mainfile(filepath=str(candidate),check_existing=False)
 print('HEAD_IMU_SAVED',sha(candidate),'mass',total,flush=True)
elif mode=='verify':
 fail=[];expected=set(B['objects'])|set(D['added'])|set(D['reserves'])|{'FRAME_JY61P_HEAD_SENSOR'};maxerr=0.
 if set(bpy.data.objects)!=set():pass
 if set(bpy.data.objects.keys())!=expected:fail.append(['object_set'])
 for n,r in B['objects'].items():
  o=bpy.data.objects[n]
  if (o.parent.name if o.parent else None)!=r['parent']:fail.append(['parent',n])
  actual=[dict(path=f.data_path,index=f.array_index,expression=f.driver.expression,variables=[dict(name=v.name,type=v.type,targets=[dict(id=t.id.name if t.id else None,data_path=t.data_path) for t in v.targets]) for v in f.driver.variables]) for f in o.animation_data.drivers] if o.animation_data else []
  if actual!=r['drivers']:fail.append(['driver',n])
  if len(o.constraints)!=r['constraint_count']:fail.append(['constraints',n])
  for k in ['angle_deg','reference_deg','min_deg','max_deg','hardware_enable','servo_bus','servo_bus_id']:
   if k in r['props'] and str(o.get(k))!=r['props'][k]:fail.append(['property',n,k])
  if n in D['replaced']:continue
  e=float(np.max(abs(np.array(o.matrix_world)-np.array(r['matrix']))));maxerr=max(maxerr,e)
  if e>1e-6:fail.append(['matrix',n,e])
  if r.get('local_mesh_hash'):
   o.data.calc_loop_triangles();v=np.array([q.co[:] for q in o.data.vertices],np.float32);f=np.array([tuple(t.vertices) for t in o.data.loop_triangles],np.int32)
   if hashlib.sha256(v.tobytes()+f.tobytes()).hexdigest()!=r['local_mesh_hash']:fail.append(['mesh_hash',n])
 for n in D['parts']:
  o=bpy.data.objects[n];bm=bmesh.new();bm.from_mesh(o.data);bad=sum(not e.is_manifold for e in bm.edges);bm.free()
  if bad:fail.append(['nonmanifold',n,bad])
  e=np.max(abs(np.array([o.matrix_world@v.co for v in o.data.vertices])*1000-A[n+'__v']))
  if e>.001:fail.append(['worldmesh',n,float(e)])
 hidden=[o.name for o in bpy.data.objects if o.type=='MESH' and (o.hide_get() or o.hide_render or o.hide_viewport)];orphans=[m.name for m in bpy.data.meshes if not m.users]
 if hidden or orphans:fail.append(['hidden_or_orphans',hidden,orphans])
 ds=[dict(owner=o.name,expression=f.driver.expression,valid=f.driver.is_valid) for o in bpy.data.objects if o.animation_data for f in o.animation_data.drivers]
 if len(ds)!=19 or not all(d['valid'] for d in ds):fail.append(['driver_count_validity'])
 poseerr=0.;count=0
 for infile,array in [('source_regression_poses.json','source_regression_transforms.npz'),('leg_poses.json','leg_transforms.npz'),('dense_head_input.json','dense_head_transforms.npz')]:
  root=R/'work_in_progress/lower_module_v05';info=json.loads((root/infile).read_text());ts=np.load(root/array)['transforms'];groups=info['groups'];set_reference();inv={n:bpy.data.objects[n].matrix_world.inverted() for n in groups}
  poses=[p['angles_deg'] for p in info['poses']] if 'poses' in info else [dict(zip(info['controls'],p)) for p in info['angles']]
  for pi,p in enumerate(poses):
   set_reference()
   for n,a in p.items():bpy.data.objects[n]['angle_deg']=a;bpy.data.objects[n].update_tag()
   bpy.context.view_layer.update()
   for j,n in enumerate(groups):
    t=np.array(bpy.data.objects[n].matrix_world@inv[n]);t[:3,3]*=1000;poseerr=max(poseerr,float(np.max(abs(t-ts[pi,j]))))
   count+=1
 if poseerr>.002:fail.append(['pose_matrix',poseerr])
 set_reference()
 if any(bpy.context.scene.get(k,False) for k in flags):fail.append(['release_flags'])
 mass=json.loads((P/'mass_estimate.json').read_text());total=sum(float(o.get('mass_estimate_g',0)) for o in bpy.data.objects)
 if abs(total-mass['mass_estimate_g'])>.0001:fail.append(['mass',total])
 frame=bpy.data.objects['FRAME_JY61P_HEAD_SENSOR'];q=dict(file=Path(bpy.data.filepath).resolve().relative_to(R.resolve()).as_posix(),sha256=sha(Path(bpy.data.filepath)),failures=fail,objects=len(bpy.data.objects),mesh_datablocks=len(bpy.data.meshes),mass_rows=len(mass['rows']),drivers=ds,hidden=hidden,orphans=orphans,unchanged_objects=len(B['objects'])-1,max_matrix_error_m=maxerr,pose_count=count,max_pose_matrix_error_mm=poseerr,mass_estimate_g=total,head_imu_frame_head_local_m=[list(x) for x in frame.parent.matrix_world.inverted()@frame.matrix_world],head_imu_reference_world_m=[list(x) for x in frame.matrix_world],source_blueprint_sha256=sha(R/'exports/Microduck_assembly.blend'),**{k:False for k in flags})
 dump('current_reopen_checks.json' if Path(bpy.data.filepath).parent.name=='current' else 'reopen_checks.json',q);print('REOPEN',q,flush=True)
elif mode=='render':
 output=args.output.resolve()
 for protected in [R/'current',P,R/'versions',R/'exports',R/'source',R/'scripts']:
  if output==protected.resolve() or output.is_relative_to(protected.resolve()):
   raise SystemExit('Render output must be outside current, evidence, archives and source directories.')
 render_names=['head_imu_open','head_imu_mount','head_imu_top','head_imu_bottom','head_imu_closed','assembly_three_quarter']
 if any((output/(n+'.png')).exists() for n in render_names):
  raise SystemExit('Render output already contains preview files; select a new --output directory.')
 output.mkdir(parents=True,exist_ok=True)
 s=bpy.context.scene;s.render.engine='BLENDER_WORKBENCH';s.render.resolution_x=1500;s.render.resolution_y=1100;s.render.resolution_percentage=100;s.display.shading.light='STUDIO';s.display.shading.color_type='MATERIAL';s.display.shading.show_cavity=True;s.display.shading.cavity_type='BOTH';s.display.shading.background_type='WORLD';s.world.color=(.72,.75,.78);s.view_settings.view_transform='Standard';s.view_settings.look='None';s.view_settings.exposure=.35
 mesh={o.name for o in bpy.data.objects if o.type=='MESH' and '09_STUDIO' not in [c.name for c in o.users_collection]};head={n for n in mesh if bpy.data.objects[n].parent and bpy.data.objects[n].parent.name=='BODY_jaw_soft'}
 def render(n,target,off,scale,names):
  for o in bpy.data.objects:
   if o.type=='MESH':o.hide_render=o.name not in names
  t=Vector(target);cam=s.camera;cam.location=t+Vector(off);cam.rotation_euler=(t-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.type='ORTHO';cam.data.ortho_scale=scale;s.render.filepath=str(output/(n+'.png'));bpy.ops.render.render(write_still=True)
 render('head_imu_open',(.026,0,.246),(-.24,-.32,.35),.15,head-{'Rex_Skull_Shell'})
 render('head_imu_mount',(.037,0,.261),(-.18,-.27,.35),.065,set(D['parts'])|{'V10_PI_ZERO_2W_PCB'})
 render('head_imu_top',(.037,0,.261),(0,0,.5),.070,set(D['parts'])|{'V10_PI_ZERO_2W_PCB'})
 render('head_imu_bottom',(.036,0,.259),(-.15,-.22,-.30),.060,set(D['parts']))
 render('head_imu_closed',(.02,0,.245),(.3,-.35,.24),.17,head)
 render('assembly_three_quarter',(-.09,0,.15),(.32,-.6,.24),.63,mesh)
 print('RENDERED_NO_SAVE',flush=True)
