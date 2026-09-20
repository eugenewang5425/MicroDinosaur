"""Isolate slower outer IMU transport from the joint-delay changes in phase A."""
import json
from dataclasses import asdict
from pathlib import Path
from heading_sim import HeadingExperiment,ImuTransportConfig
from hardware_sim import HardwareCase
from evaluate_policy import sha

ROOT=Path(__file__).resolve().parent


def main():
    out=ROOT/'20260914_imu_heading/isolated_transport.json'
    assert not out.exists(),'Do not overwrite this follow-up'
    policy=ROOT/'20260913_handoff/v7_reference.onnx'
    hardware=HardwareCase(physics_dt=.00125)
    transport=ImuTransportConfig(period_ms=40,latency_ms=60)
    records=[]
    for seed in (1,2):
        for mode in ('open','imu'):
            sim=HeadingExperiment(ROOT/'20260913_handoff/native_v07',policy,hardware,transport)
            metrics,_=sim.run('straight',mode,seed,12)
            records.append(metrics)
            print(seed,mode,metrics['heading_error_rms_deg'],flush=True)
    out.write_text(json.dumps({'status':'COMPLETE','post_hoc':True,
        'reason':'Phase A changed motor/joint timing as well as IMU transport and rejected stationary calibration; isolate the outer IMU transport',
        'hardware':asdict(hardware),'outer_imu':asdict(transport),'policy_sha256':sha(policy),
        'records':records,'source_hashes':{p.name:sha(p) for p in [Path(__file__),ROOT/'heading_sim.py',ROOT/'imu_heading.py',ROOT/'hardware_sim.py',ROOT/'evaluate_policy.py']}},indent=2),encoding='utf-8')


if __name__=='__main__':main()
