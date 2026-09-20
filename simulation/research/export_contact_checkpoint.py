from pathlib import Path
from copy import deepcopy
import argparse,json
from mjlab.tasks.registry import register_mjlab_task,load_runner_cls
from mjlab_microduck.export import run_export,ExportConfig
from contact_motion_cfg import build_config
from evaluate_policy import sha
p=argparse.ArgumentParser();p.add_argument('--skill',required=True);p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert not a.output.exists();a.output.parent.mkdir(exist_ok=True)
task,cfg=build_config(a.skill,1);runner=load_runner_cls(task);task='Mjlab-Contact-Eval-'+a.skill
register_mjlab_task(task,cfg.env,deepcopy(cfg.env),cfg.agent,runner)
r=run_export(task,ExportConfig(checkpoint_file=str(a.checkpoint),onnx_file=str(a.output),num_envs=1))
a.output.with_suffix('.json').write_text(json.dumps(dict(source=str(a.checkpoint),source_sha256=sha(a.checkpoint),onnx_sha256=sha(a.output)),indent=2))
