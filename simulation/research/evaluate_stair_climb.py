"""上高台阶分层验收:逐高度 × 逐速度,给出通过梯度。

与 corrective 验收(单点 steps_10)不同:爬台阶模组要一张**能力梯度表**,
才能看出"最陡能上多高、余量在哪一档耗尽"。
"""
from pathlib import Path
import sys
import json
import argparse

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from corrective_common import OUT  # noqa: E402
from evaluate_owned_terrain import experiment, evaluate  # noqa: E402
from evaluate_policy import sha  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', type=Path, required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--heights', type=int, nargs='+', default=[10, 15, 20, 25, 30, 35])
    ap.add_argument('--speeds', type=float, nargs='+', default=[.35, .55])
    ap.add_argument('--delays', type=int, nargs='+', default=[5, 10, 15])
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    a = ap.parse_args()
    dest = OUT / 'evaluation' / a.label
    dest.mkdir(parents=True, exist_ok=True)
    print(f'策略 {a.policy.name} sha={sha(a.policy)[:12]}', flush=True)
    records = []
    for h in a.heights:
        for v in a.speeds:
            for d in a.delays:
                row = dict(height_mm_per_step=h, total_mm=h * 3, speed=v, delay=d,
                           n=0, passed=0, fell=0, fwd=[])
                for sd in a.seeds:
                    case = dict(terrain=f'steps_{h}', scenario='straight',
                                program='legacy', depth=0, delay=d, speed=v,
                                seconds=14)
                    try:
                        e = experiment(str(a.policy), case)
                        m, _ = evaluate(e, case, sd)
                    except Exception as exc:
                        row['n'] += 1
                        row.setdefault('errors', []).append(repr(exc)[:60])
                        continue
                    fwd = float(m.get('forward_displacement_m', 0.))
                    row['n'] += 1
                    row['fell'] += 1 if m['fell'] else 0
                    row['passed'] += 1 if (not m['fell'] and fwd >= .8) else 0
                    row['fwd'].append(round(fwd, 2))
                records.append(row)
                print(f"  台阶{h:2d}mm/级(总{h * 3:3d}mm) v{v} d{d:2d}ms: "
                      f"{row['passed']}/{row['n']} 通过  跌倒{row['fell']}  "
                      f"前进{row['fwd']}", flush=True)
        (dest / 'matrix.json').write_text(json.dumps(records, indent=2),
                                          encoding='utf-8')
    # 梯度摘要:每个高度取"最好一档速度×延迟"的通过率
    print('\n=== 能力梯度(每高度最好档) ===')
    for h in a.heights:
        sub = [r for r in records if r['height_mm_per_step'] == h]
        best = max(sub, key=lambda r: (r['passed'], -r['fell'], max(r['fwd'] or [0])))
        print(f"  台阶{h:2d}mm/级(总{h * 3:3d}mm): 最好 {best['passed']}/{best['n']}"
              f"  (v{best['speed']} d{best['delay']}ms)")
    (dest / 'complete.json').write_text(json.dumps(
        dict(status='COMPLETE', policy_sha256=sha(a.policy), records=records),
        indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
