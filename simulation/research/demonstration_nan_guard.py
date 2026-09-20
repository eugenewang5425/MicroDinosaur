"""Keep fail-fast behavior and preserve randomized model fields on a fault."""
import importlib
import json
from functools import partial
import numpy as np
from research_nan_guard import ResearchNanGuard


class DemonstrationNanGuard(ResearchNanGuard):
    simulation=None
    def check_and_dump(self,data):
        try:return super().check_and_dump(data)
        except FloatingPointError:
            # The parent has saved its full post-fault state and stopped the
            # physics step. Capture DR fields while that state is still intact.
            sim=self.simulation
            if sim is not None:
                latest=max(self.output_dir.glob('physics_*.json'),key=lambda p:p.name)
                record=json.loads(latest.read_text());ids=record['dumped_env_ids'];values={}
                try:
                    for name in sorted(sim.expanded_fields):
                        array=getattr(sim.model,name);values['model_'+name]=array[ids].cpu().numpy()
                    for name in ('qfrc_applied','xfrc_applied','actuator_force','actuator_velocity'):
                        if hasattr(data,name):values['data_'+name]=getattr(data,name)[ids].cpu().numpy()
                    path=latest.with_name(latest.stem+'_randomized.npz');np.savez_compressed(path,**values)
                    record['randomized_context_file']=path.name;record['randomized_fields']=sorted(sim.expanded_fields)
                    record['model_note']='Compile-time MJB plus selected per-environment expanded fields; simulator internals may still prevent exact replay.'
                except Exception as error:record['randomized_context_error']=str(error)
                latest.write_text(json.dumps(record,indent=2))
            raise


def install(output_dir):
    module=importlib.import_module('mjlab.sim.sim')
    module.NanGuard=partial(DemonstrationNanGuard,output_dir=output_dir)
    original=module.Simulation.__init__
    def init(self,*args,**kwargs):
        original(self,*args,**kwargs)
        self.nan_guard.simulation=self
    module.Simulation.__init__=init
