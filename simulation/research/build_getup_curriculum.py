"""Add physically generated leaning states to bridge standing and full falls."""
import json
from pathlib import Path
from build_contact_motion_bank import OUT,experiment,prepare_fall
selected=json.loads((OUT/'selected_contact.json').read_text())
base=json.loads((OUT/('getup_reset_bank'+selected['bank_suffix']+'.json')).read_text())
rows=[]
for name,direction in [('left',[0,1,0]),('right',[0,-1,0]),('front',[-1,0,0]),('back',[1,0,0])]:
    e,cal=experiment(2);trace,frames=prepare_fall(e,direction,capture_tilts=True)
    for record in e.lean_snapshots:
        record.update(label=name+'_'+record['label'],seed=2,calibration=cal)
        base['records'].append(record);rows.append(record['label'])
base.update(curriculum='Mix standing, physically generated leaning states and resting fallen states. Initial distribution only. Full-fall evaluation remains separate.',
    lean_states=rows)
name='getup_reset_bank'+selected['bank_suffix']+'_curriculum.json'
dest=OUT/name;assert not dest.exists();dest.write_text(json.dumps(base,indent=2))
selected['getup_bank']=name;(OUT/'selected_contact.json').write_text(json.dumps(selected,indent=2))
print('curriculum',len(base['records']),rows,flush=True)
