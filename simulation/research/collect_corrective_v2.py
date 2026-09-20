"""纠正采集 v2:学生换最新纠正组模型,初态瞄准探针出的实际失败组合。

与 collect_corrective.py 的差异(20260917):
  1. 学生 = microdinosaur_corrective_corrective/20260914_train_512x201/candidate.onnx
     (上一轮 512x201 训练产物,即"最新模型");
  2. 初态 = probe_latest.json 的失败组合 + 新种子 221-226(201-203 保留盲测,不采集);
  3. stop 只采 d5(探针 9/9 残速 ~35mm/s 全败;d15 全过不重采);
  4. 落盘到 collection_v2/ 与 *_v2.npz,不覆盖冻结研究产物。
准入门与原版逐字相同:无跌倒、越限≤0.02rad、转速≤16.5、力矩≤0.6001Nm、
接管后前进≥0.4m(总≥0.8m) / 停稳≤10mm/s 且制动位移≤80mm。
"""
from pathlib import Path
import sys
import json

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from corrective_common import OUT, STOP_TEACHER  # noqa: E402
from collect_corrective import Takeover  # noqa: E402
from demonstration_expert import experiment as make_expert  # noqa: E402
from evaluate_owned_terrain import cases, experiment, evaluate  # noqa: E402
from evaluate_policy import sha  # noqa: E402

STUDENT = Path('D:/microduck_rl/logs/rsl_rl/microdinosaur_corrective_corrective'
               '/20260914_train_512x201/candidate.onnx')
DEST = OUT / 'collection_v2'
SEEDS = list(range(31, 37)) + list(range(221, 227))
HOLDOUT = [201, 202, 203, 401, 402, 403]


def run(kind, delay, seed, switch):
    case = next(c for c in cases() if c['terrain'] == 'steps_10' and c['speed'] == .2) \
        if kind == 'recovery' \
        else next(c for c in cases() if c['program'] == 'resume' and c['delay'] == 5)
    case = dict(case, delay=delay)
    e = experiment(str(STUDENT), case)
    if kind == 'recovery':
        unused, teacher = make_expert(.50, delay, .03)
        del unused
    else:
        teacher = ort.InferenceSession(str(STOP_TEACHER),
                                       providers=['CPUExecutionProvider'])
    recorder = Takeover(e.sim.session, teacher, round(switch / .02))
    e.sim.session = recorder
    q = []
    before = e.sim.substep_callback

    def capture():
        before()
        if e.active:
            q.append(e.sim.data.qpos[e.sim.jadr].copy())
    e.sim.substep_callback = capture
    m, t = evaluate(e, case, seed)
    q = np.asarray(q)
    f = np.asarray(e.sim.gaze_trace)
    limits = e.sim.model.jnt_range[e.sim.jids]
    violation = float(np.maximum(np.maximum(limits[:, 0] - q, q - limits[:, 1]),
                                 0).max())
    after = f[f[:, 0] >= 6 + switch - 1e-8]
    progress = float(after[-1, 1] - after[0, 1])
    torque = float(max(np.max(np.abs(row[1])) for row in e.sim.trace[-len(q):]))
    admitted = (not m['fell'] and violation <= .02 and torque <= .6001
                and m['physics']['joint_speed_max_rad_s'] <= 16.5)
    if kind == 'recovery':
        admitted = admitted and m['terrain_traversed'] and progress >= .4
    else:
        admitted = admitted and m['post_stop']['planar_speed_mean_mm_s'] <= 10 \
            and m['braking_net_displacement_mm'] <= 80
    obs = np.asarray(recorder.obs, np.float32)
    actions = np.asarray(recorder.actions, np.float32)
    mask = np.asarray(recorder.teacher_mask)
    np.testing.assert_array_equal(obs[1:, 44:63], actions[:-1])
    m.update(accepted=bool(admitted), joint_limit_violation_max_rad=violation,
             torque_peak_nm=torque, after_switch_progress_m=progress,
             switch_x=float(after[0, 1]),
             after_switch_vx=float(after[:, 13].mean()))
    arrays = dict(obs=obs, actions=actions, teacher_mask=mask, gaze=f, joints=q,
                  trace=t, head_controls=np.asarray(e.head_rows),
                  head_physics=np.asarray(e.head_physics))
    if kind == 'recovery':
        arrays['expert_obs'] = np.asarray(teacher.expert_obs)
    return m, arrays


def main():
    kind = sys.argv[1] if len(sys.argv) > 1 else 'recovery'
    if kind == 'recovery':
        jobs = [(d, s, (4., 6., 8.)[(si) % 3])
                for d in (5, 15) for si, s in enumerate(SEEDS)]
    else:
        jobs = [(5, s, 17.) for s in SEEDS]
    dest = DEST / kind
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'plan.json').write_text(json.dumps(dict(
        jobs=jobs, student_sha256=sha(STUDENT), student=str(STUDENT), kind=kind,
        holdout_seeds=HOLDOUT, state_reset=False,
        basis='probe_latest.json 失败组合 + 新种子;准入门与 collect_corrective 逐字相同'),
        indent=2), encoding='utf-8')
    records = []
    bank = {k: [] for k in ('obs', 'actions', 'trial_id', 'frontier')}
    for i, (delay, seed, switch) in enumerate(jobs):
        name = f'lag{delay}_s{seed}_switch{switch:g}'
        try:
            m, arrays = run(kind, delay, seed, switch)
        except ValueError as exc:
            if 'Calibration' not in repr(exc):
                raise
            # RECIPE 惯例:静止校准拒绝单独保留,不算策略跌倒,不放宽阈值
            record = dict(name=name, delay=delay, seed=seed, switch=switch,
                          calibration_failed=True, error=repr(exc))
            records.append(record)
            (dest / f'{name}.json').write_text(json.dumps(record, indent=2))
            print(json.dumps(dict(kind=kind, name=name, calibration_failed=True)),
                  flush=True)
            continue
        record = dict(name=name, delay=delay, seed=seed, switch=switch, metrics=m)
        (dest / f'{name}.json').write_text(json.dumps(record, indent=2))
        np.savez_compressed(dest / f'{name}.npz', **arrays)
        records.append(record)
        if m['accepted']:
            mask = arrays['teacher_mask']
            count = int(mask.sum())
            bank['obs'].extend(arrays['obs'][mask])
            bank['actions'].extend(arrays['actions'][mask])
            bank['trial_id'].extend([i] * count)
            bank['frontier'].extend((np.arange(count) < 75).tolist())
        print(json.dumps(dict(kind=kind, name=name, accepted=m['accepted'],
                              fell=m['fell'],
                              progress=m['after_switch_progress_m'],
                              limit=m['joint_limit_violation_max_rad'])), flush=True)
        (dest / 'matrix.json').write_text(json.dumps(records, indent=2))
    assert len(bank['obs']) >= 300, 'Too few physically accepted correction frames'
    dataset = OUT / f'{kind}_demonstrations_v2.npz'
    np.savez_compressed(dataset, **{
        k: np.asarray(v, np.float32 if k in ('obs', 'actions') else np.int32)
        for k, v in bank.items()})
    (OUT / f'{kind}_dataset_v2.json').write_text(json.dumps(dict(
        frames=len(bank['obs']),
        accepted=sum(1 for r in records if r.get('metrics', {}).get('accepted')),
        attempted=sum(1 for r in records if 'metrics' in r),
        calibration_failed=sum(1 for r in records if r.get('calibration_failed')),
        sha256=sha(dataset),
        meaning='Only physically executed teacher actions after state-preserving '
                'takeover; frontier is first 1.5 seconds.'), indent=2),
        encoding='utf-8')
    print(f'[write] {dataset}  frames={len(bank["obs"])}')


if __name__ == '__main__':
    main()
