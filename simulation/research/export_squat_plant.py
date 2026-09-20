"""Regenerate an isolated v07 export with actual rigid-parent ancestry."""
import sys,json
from pathlib import Path
import numpy as np
import bpy
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'research/20260914_squat_specialist'
sys.path.insert(0,str(ROOT/'microdinosaur'))
import blend2mjcf as conv
sys.argv=['export','--','--blend',str(ROOT/'design_source/current/MicroDinosaur_v1.blender'),
    '--mode','gen','--out',str(OUT/'robot'),'--name','microdinosaur_v07',
    '--with-tail','--with-arms','--with-jaw','--no-xml-actuators',
    '--mass-ledger',str(ROOT/'design_source/current/mass_estimate.json'),
    '--imu-mounts',str(ROOT/'design_source/current/imu_mounts.json')]
conv.main()
# Regression: these are actual upstream CAD attachments, never moving bolts.
checks={
    'S288_left_hip_pitch_OUT_1_M2x5_0':'hip_l',
    'S288_left_hip_pitch_DISC_1':'hip_l',
    'S288_left_hip_pitch_CASE':'upper_leg_left',
    'S288_left_ankle_CASE':'leg',
    'S288_left_ankle_DISC_1':'ankle_left',
    'S288_head_pitch_OUT_1_M2x5_0':'neck_pitch',
    'S288_left_knee_OUT_1_M2x5_0':'leg'}
parts=conv.body_parts(conv.blend.empties,conv.blend.meshes)
assigned={o.name:b for b,objects in parts.items() for o in objects}
for name,body in checks.items():assert assigned[name]==body,(name,assigned[name],body)
# Every declared rigid fastener attachment must resolve to the same body.
attachment_checks=[]
for obj in conv.blend.meshes:
    attachment=obj.get('attachment')
    if obj.name not in assigned or attachment not in assigned:continue
    assert assigned[obj.name]==assigned[attachment],(obj.name,attachment,assigned[obj.name],assigned[attachment])
    attachment_checks.append([obj.name,attachment,assigned[obj.name]])
(OUT/'attachment_regression.json').write_text(json.dumps(dict(checked=checks,all_parts=assigned,
    fastener_attachment_checks=attachment_checks),indent=2),encoding='utf-8')
