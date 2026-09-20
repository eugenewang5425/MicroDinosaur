"""Portable fail-fast physics diagnostics for this research run on Windows."""
from datetime import datetime
import importlib
import json
from pathlib import Path
from functools import partial
import mujoco
import numpy as np
import torch
from mjlab.utils.nan_guard import NanGuard


class ResearchNanGuard(NanGuard):
    def __init__(self,*args,output_dir,**kwargs):
        super().__init__(*args,**kwargs)
        self.output_dir=Path(output_dir)

    def check_and_dump(self,data):
        if not self.enabled:return False
        bad=self.detect_nans(data)
        if not bad.any():return False
        ids=torch.where(bad)[0].cpu().tolist();selected=ids[:self.max_envs_to_dump]
        self.output_dir.mkdir(parents=True,exist_ok=True)
        stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f');base=self.output_dir/f'physics_{stamp}'
        arrays={};nonfinite={}
        for name in ('qpos','qvel','qacc','qacc_warmstart','sensordata','ctrl'):
            if hasattr(data,name):
                value=getattr(data,name)[selected].cpu().numpy();arrays['post_'+name]=value
                nonfinite[name]=np.argwhere(~np.isfinite(value)).tolist()
        for name in ('qpos','qvel'):
            arrays['history_'+name]=np.stack([r[name][selected].cpu().numpy() for r in self.buffer]) if self.buffer else np.empty(0)
        arrays['history_steps']=np.asarray([r['step'] for r in self.buffer])
        np.savez_compressed(base.with_suffix('.npz'),**arrays)
        buffer=np.empty(mujoco.mj_sizeModel(self.mj_model),dtype=np.uint8)
        mujoco.mj_saveModel(self.mj_model,buffer=buffer);base.with_suffix('.mjb').write_bytes(buffer.tobytes())
        record=dict(step=self.step_counter,bad_env_ids=ids,dumped_env_ids=selected,nonfinite_indices=nonfinite,
            model_note='Compile-time model only; per-environment randomized model fields are not captured.',
            action='Aborted before the next PPO update. No symlinks or permission changes.')
        base.with_suffix('.json').write_text(json.dumps(record,indent=2))
        self._dumped=True
        raise FloatingPointError(f'Non-finite physics in envs {ids}; diagnostic saved to {base}.json')


def install(output_dir):
    # Opt-in only, before the environment is built. Leave installed mjlab files
    # and all other skills/runs unchanged.
    module=importlib.import_module('mjlab.sim.sim')
    module.NanGuard=partial(ResearchNanGuard,output_dir=output_dir)
