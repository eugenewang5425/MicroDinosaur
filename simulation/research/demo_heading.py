"""Run one reproducible simulation using the fixed heading-loop configuration."""
import argparse,json
from pathlib import Path
from dataclasses import asdict
import numpy as np
from heading_sim import HeadingExperiment,ImuTransportConfig,TRACE_COLUMNS
from run_heading_stable_start import StableStartExperiment,CALIBRATOR,POLICY
from imu_heading import HeadingConfig,EstimatorConfig
from hardware_sim import HardwareCase
from evaluate_policy import sha


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--policy',choices=['v7','s42_no_neck'],default='v7')
    p.add_argument('--mode',choices=['open','imu'],default='imu')
    p.add_argument('--scenario',choices=['stand','straight','left_then_hold','right_then_hold','s_turn'],default='straight')
    p.add_argument('--seed',type=int,default=1);p.add_argument('--seconds',type=float,default=12.)
    p.add_argument('--out',required=True)
    args=p.parse_args();out=Path(args.out)
    if out.exists():raise FileExistsError('Choose a new output folder to preserve earlier evidence')
    if not 1<=args.seconds<=120:raise ValueError('Use a bounded 1-120 s rollout')
    out.mkdir(parents=True)
    policy=CALIBRATOR if args.policy=='v7' else POLICY
    kind=HeadingExperiment if args.policy=='v7' else StableStartExperiment
    physics=HardwareCase(physics_dt=.00125)
    sim=kind(Path(__file__).parent/'20260913_handoff/native_v07',policy,physics)
    metrics,trace=sim.run(args.scenario,args.mode,args.seed,args.seconds,True)
    result={'request':vars(args),'metrics':metrics,'policy_sha256':sha(policy),
        'heading_config':asdict(HeadingConfig()),'estimator_config':asdict(EstimatorConfig()),
        'outer_imu':asdict(ImuTransportConfig()),'hardware':asdict(physics),'real_hardware_connection':False}
    (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    np.savez_compressed(out/'trace.npz',trace=trace,columns=np.array(TRACE_COLUMNS))
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
