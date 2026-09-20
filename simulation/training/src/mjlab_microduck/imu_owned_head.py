"""Opt-in, sensor-anchored head action; previous control adapters are untouched."""
from dataclasses import dataclass
from mjlab_microduck.head_attitude_torch import TorchHeadController
from mjlab_microduck.head_imu_action import static_kinematics
from mjlab_microduck.calibrated_head_action import CalibratedHeadAction,CalibratedHeadActionCfg


class TorchImuOwnedHeadController(TorchHeadController):
    def __init__(self,*args,owned_indices=(0,1,2),**kwargs):
        super().__init__(*args,**kwargs)
        self.owned_indices=list(owned_indices)

    def update(self,nominal,angles,orientation,gyro,desired,omega,dt,age):
        old=self.last_target.clone()
        measured=angles[:,1:];baseline=nominal.clone()
        baseline[:,self.owned_indices]=measured[:,self.owned_indices]
        self.nominal_filtered[:,self.owned_indices]=measured[:,self.owned_indices]
        result=super().update(baseline,angles,orientation,gyro,desired,omega,dt,age)
        self.correction[self.stale]=0
        self.last_target[self.stale]=old[self.stale]
        self.nominal_filtered[self.stale]=old[self.stale]
        result[self.stale]=old[self.stale]
        return result


@dataclass(kw_only=True)
class ImuOwnedActionCfg(CalibratedHeadActionCfg):
    owned_head_indices: tuple[int,...]=(1,)
    def build(self,env):return ImuOwnedAction(self,env)


class ImuOwnedAction(CalibratedHeadAction):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.controller=TorchImuOwnedHeadController(static_kinematics(env.sim.mj_model),env.num_envs,env.device,
            owned_indices=cfg.owned_head_indices)
