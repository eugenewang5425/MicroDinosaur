"""Head IMU owns pitch/yaw/roll; gait policy retains neck and the other joints."""
from dataclasses import replace
import numpy as np
from head_attitude import HeadController


class ImuOwnedHeadController(HeadController):
    def __init__(self,kinematics,config=None,owned_indices=(0,1,2)):
        super().__init__(kinematics,config)
        self.owned_indices=list(owned_indices)

    def update(self,nominal_target,joint_angles,orientation,gyro,desired_orientation,
               desired_omega_world,dt,measurement_age):
        # A bounded position increment is anchored to delayed measured joints,
        # not a competing learned head target. No world-state feedback.
        old=self.last_target.copy()
        measured=np.asarray(joint_angles)[1:]
        baseline=np.asarray(nominal_target).copy()
        baseline[self.owned_indices]=measured[self.owned_indices]
        self.nominal_filtered[self.owned_indices]=measured[self.owned_indices]
        result=super().update(baseline,joint_angles,orientation,gyro,desired_orientation,
            desired_omega_world,dt,measurement_age)
        if self.stale:
            # Stale feedback holds the last commanded head pose. Do not
            # repeatedly add a decaying correction to last_target.
            self.correction[:]=0;self.integral[:]=0
            self.last_target=old;self.nominal_filtered=old.copy()
            return old.copy()
        return result


def configure_owned(experiment,owned_indices=(0,1,2)):
    experiment.head_config=replace(experiment.head_config,nominal_target_filter_tau_s=.1)
    experiment.head_controller=ImuOwnedHeadController(experiment.kinematics,experiment.head_config,owned_indices)
    return experiment
