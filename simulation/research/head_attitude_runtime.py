"""Operational simulation profile after the frozen slow-transport probe.

At >40 ms source age, release head IMU correction and retain the bounded,
filtered gait head target. This is loss of stabilization, not a new estimate.
"""
from dataclasses import replace
from run_head_attitude import make_experiment
from head_attitude import HeadController


def make_operating_experiment(*args, **kwargs):
    e = make_experiment(*args, **kwargs)
    e.head_config = replace(e.head_config, max_measurement_age_s=.04)
    e.head_controller = HeadController(e.kinematics, e.head_config)
    return e
