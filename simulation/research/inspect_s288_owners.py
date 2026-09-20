import sys,json
from pathlib import Path
import bpy
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'microdinosaur'))
import blend2mjcf as c
c.WITH_TAIL=c.WITH_ARMS=c.WITH_JAW=True
bpy.ops.wm.open_mainfile(filepath=str(ROOT/'design_source/current/MicroDinosaur_v1.blender'))
empties={o.name:o for o in bpy.data.objects if o.type=='EMPTY'}
rows=[]
for o in bpy.data.objects:
    if not o.name.startswith('S288_'):continue
    if not ('_DISC_' in o.name or '_CASE' in o.name or '_M2x5_0' in o.name):continue
    attachment=o.get('attachment');part=bpy.data.objects.get(attachment) if attachment else None
    rows.append(dict(name=o.name,parent=o.parent.name if o.parent else None,
        owner=c.anchor_body(o,empties),attachment=attachment,
        attachment_owner=c.anchor_body(part,empties) if part else None))
(ROOT/'research/20260914_squat_specialist/owner_audit.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows))
