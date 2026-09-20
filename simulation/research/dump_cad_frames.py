"""Read-only reference frames for checking CAD-to-MuJoCo conversion."""
import bpy
import json
import sys
from pathlib import Path

out = Path(sys.argv[sys.argv.index('--') + 1])
frames = {o.name: [list(row) for row in o.matrix_world]
          for o in bpy.data.objects
          if o.name.startswith(('FRAME_trunk_base', 'BODY_', 'AXIS_', 'FRAME_JY61P_HEAD_SENSOR'))}
out.write_text(json.dumps(frames, indent=2), encoding='utf-8')
print(f'Exported {len(frames)} reference frames, CAD not saved')
