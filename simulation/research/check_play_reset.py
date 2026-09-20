"""Regression checks on zcode's interactive simulator, using the real ONNX."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, r'D:\microduck_rl\scripts')
import microdinosaur_play as play

onnx = Path(__file__).parent / '20260913_handoff/v7_reference.onnx'
sim = play.Sim(play.DEFAULT_XML['fine'], str(onnx), kp=7, kv=.8, iters=20)
cmd = np.zeros(18, dtype=np.float32)


def trajectory():
    sim.reset()
    result = []
    for _ in range(30):
        sim.step(cmd)
        result.append(sim.data.qpos.copy())
    return np.array(result)


first = trajectory()
turn = cmd.copy(); turn[2] = .9
for _ in range(50):
    sim.step(turn)
assert np.array_equal(first, trajectory()), 'Filter history leaked through reset'
sim.reset()
obs = sim.obs(cmd)
expected = sim.sess.run(None, {'obs': obs})[0][0]
sim.step(cmd, 1.15)
assert np.array_equal(sim.last_action, expected), 'Amplitude scaled the raw-action feedback'
print('PASS: reset reproduces trajectory exactly; scale1.15 preserves raw-action feedback')
