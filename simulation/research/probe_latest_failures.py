"""最新纠正组模型在 5/15ms 台阶/停稳上的实际失败初态探针。

依据 20260914_corrective/RESULTS.md 下一步:"从最新模型在5/15ms下实际访问的
失败状态继续采集纠正,而非只重复旧学生状态。先复查关节越限、停止残余运动
及静止校准启动。" 本脚本只做探针,不采集、不训练。

输出:probe_latest.json — 每个 (delay, seed) 的通过性/失败点/根高/倾角,
供 collect_corrective_v2 选定采集初态。
"""
from pathlib import Path
import sys
import json

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from evaluate_owned_terrain import cases, experiment, evaluate  # noqa: E402
from evaluate_policy import sha  # noqa: E402

OUT = ROOT / '20260914_corrective'
STUDENT = Path('D:/microduck_rl/logs/rsl_rl/microdinosaur_corrective_corrective'
               '/20260914_train_512x201/candidate.onnx')
DELAY_JOBS = (5, 15)          # 10ms 上一轮已达标率最高,先盯最差两档
STATES = list(range(31, 37)) + [201, 202, 203]   # 旧训练态 + 独立确认态


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    results = []
    step_case = next(c for c in cases() if c['terrain'] == 'steps_10' and c['speed'] == .2)
    stop_case = next(c for c in cases() if c['program'] == 'resume' and c['delay'] == 5)
    for delay in DELAY_JOBS:
        for seed in STATES:
            for kind, case in (('recovery', dict(step_case, delay=delay)),
                               ('stop', dict(stop_case, delay=delay))):
                e = experiment(STUDENT, case)
                try:
                    m, _trace = evaluate(e, case, seed)
                except Exception as exc:
                    results.append(dict(kind=kind, delay=delay, seed=seed,
                                        error=repr(exc)))
                    continue
                q = np.asarray(e.sim.data.qpos[e.sim.jadr]) if False else None
                row = dict(kind=kind, delay=delay, seed=seed,
                           fell=bool(m.get('fell')),
                           traversed=m.get('terrain_traversed'),
                           forward_m=float(m.get('forward_displacement_m', float('nan'))),
                           joint_excess_rad=float(m.get('joint_limit_excess_rad', 0.) or 0.),
                           stop_speed_mm_s=float(
                               (m.get('post_stop') or {}).get('planar_speed_mean_mm_s',
                                                              float('nan'))),
                           root_z_end_mm=float(e.sim.data.qpos[2] * 1000))
                results.append(row)
                tag = 'FAIL' if (m.get('fell') or
                                 (kind == 'recovery' and not m.get('terrain_traversed')) or
                                 (kind == 'stop' and row['stop_speed_mm_s'] > 10.)) else 'pass'
                print(f"[{tag}] {kind:8s} d{delay:2d} s{seed}: "
                      f"fell={row['fell']} trav={row['traversed']} "
                      f"fwd={row['forward_m']:.2f}m "
                      f"stop={row['stop_speed_mm_s']:.1f}mm/s", flush=True)
    dest = OUT / 'probe_latest.json'
    dest.write_text(json.dumps(dict(
        student=str(STUDENT), student_sha256=sha(STUDENT), results=results,
        note='最新纠正组模型的 5/15ms 失败初态探针;采集初态按 FAIL 行选'),
        ensure_ascii=False, indent=2), encoding='utf-8')
    n_fail = sum(1 for r in results
                 if r.get('fell') or
                 (r.get('kind') == 'recovery' and not r.get('traversed')) or
                 (r.get('kind') == 'stop' and r.get('stop_speed_mm_s', 0) > 10.))
    print(f'[write] {dest}  失败 {n_fail}/{len(results)}')


if __name__ == '__main__':
    main()
