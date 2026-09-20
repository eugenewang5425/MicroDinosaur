"""Reject any measured reset with a new solid CAD intersection."""
import json
from pathlib import Path
OUT=Path(__file__).parent/'20260914_contact_motion'
rows=json.loads((OUT/'bank_cad/solid_intersections.json').read_text())
baseline={tuple(p['parts']):p['intersection_mm3'] for p in rows[0]['pairs']}
rejected={}
for row in rows[1:]:
    index=int(row['frame'].rsplit('_f',1)[1])
    failures=[p for p in row['pairs'] if p['intersection_mm3']-baseline[tuple(p['parts'])]>.001]
    if failures:rejected[index]=failures
bank=json.loads((OUT/'getup_reset_bank_normal005_headsafe.json').read_text())
bank['records']=[r for i,r in enumerate(bank['records']) if i not in rejected]
bank.update(head_pitch_command_range_deg=[5,27],
    reset_geometry='Every retained reset checked against solid CAD; source snapshots retain their original physical state and sensor histories. Controller slews from measured old targets.',
    excluded_initial_indices=rejected)
assert len(bank['records'])>=10 and any(r['label']=='right' for r in bank['records'])
name='getup_reset_bank_normal005_cadfiltered.json';target=OUT/name;assert not target.exists()
target.write_text(json.dumps(bank,indent=2))
selected=json.loads((OUT/'selected_contact.json').read_text());selected['getup_bank']=name
(OUT/'selected_contact.json').write_text(json.dumps(selected,indent=2))
print('retained',len(bank['records']),'rejected',rejected)
