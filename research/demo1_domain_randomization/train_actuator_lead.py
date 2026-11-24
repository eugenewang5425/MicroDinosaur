"""Continue paired-phase training with a calibrated, bounded position-target lead."""

import argparse
from pathlib import Path

import numpy as np

import continue_training as base
from demo1_domain_randomization import train as simulator
from probe_actuator_prefilter import ServoLead
from train_paired_phases import PairedPhaseCampaign
from train_ready_to_stop import ready_metrics, ready_score


class ActuatorLeadCampaign(PairedPhaseCampaign):
    def __init__(self, args):
        fit = args.output / "lead_profile.json"
        identity_file = args.output / "actuator_lead_variant.json"
        identity = dict(method="paired_phases_and_servo_lead", sha256=base.sha(Path(__file__)),
                        response_sha256=base.sha(fit),
                        prefilter_sha256=base.sha(Path(__file__).with_name("probe_actuator_prefilter.py")))
        if identity_file.exists() and base.read(identity_file) != identity:
            raise ValueError("Actuator lead trainer or response report changed")
        if not identity_file.exists():
            base.replace_json(identity_file, identity)
        super().__init__(args)
        profile = base.read(fit)
        indices = self.box.legs.ravel()
        if [row["axis"] for row in profile] != [self.box.box.plant.names[i] for i in indices]:
            raise ValueError("Servo response report does not match the actuator order")
        self.stop.lead_ticks = np.array([row["approximate_lead_ticks"] for row in profile])
        self.stop.gain = .35


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("study", "output", "pilot"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    args = parser.parse_args()
    for field in ("study", "output", "pilot"):
        setattr(args, field, getattr(args, field).resolve())
    if (args.output / "STOP").exists():
        return
    base.StopResidual = ServoLead
    base.score = ready_score
    simulator.metrics = ready_metrics
    try:
        ActuatorLeadCampaign(args).run()
    except base.StopRequested:
        return


if __name__ == "__main__":
    main()
