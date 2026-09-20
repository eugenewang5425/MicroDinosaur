"""Unattended get-up progress watcher.

Polls a training run for new checkpoints, exports each to ONNX, runs the acceptance
matrix, and appends the pass rate to a progress file. Keeps the long training run
observable without a human in the loop, and stops when the run finishes.

The matrix here is deliberately small (4 directions x 2 seeds x 1 delay = 8). It is
a progress signal, not the 36-case promotion gate.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import getup_acceptance_margin as margin

ROOT = Path(__file__).resolve().parent
CONTACT = ROOT / '20260914_contact_motion'
PY = Path('D:/microduck_rl/.venv/Scripts/python.exe')

# 无进展预算(硬门)。整体 passed 恒为 0 时,唯一的梯度是"最接近通过的那一例、
# 它最差判据的达标率"。连续这么多轮推不动就停在原地,不再陪着烧 GPU。
# 0.02 会吃掉真实抬头:实测 iter22000 的 score 从 0.917 升到 0.935(该轮全程最高),
# 差 0.018 < 0.02,被判成"无进展"并触发停手 —— 恰好在最好的一次评估上停。
# 容差只用来压浮点/分位噪声,不用来定义"什么算进展"。
NO_PROGRESS_TOL = 0.005
NO_PROGRESS_LIMIT = 3


def latest_checkpoint(run_dir):
    files = sorted(run_dir.glob('model_*.pt'), key=lambda p: int(p.stem.split('_')[-1]))
    return files[-1] if files else None


def run(cmd, log):
    with open(log, 'a', encoding='utf-8') as handle:
        handle.write(f'\n$ {" ".join(str(c) for c in cmd)}\n')
        handle.flush()
        return subprocess.run(cmd, cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT).returncode


def summarise(out_dir, count):
    rows = json.loads((out_dir / 'summary.json').read_text())
    passed = sum(1 for r in rows if r.get('passed'))
    detail = {r['key']: dict(passed=r.get('passed'), tilt=round(r.get('final_tilt_deg', -1), 1),
                             h=round(r.get('final_height_mm', -1), 1),
                             frac=round(r.get('standing_final_fraction', 0.), 3),
                             neck=round(r.get('neck_pitch_max_deg', -1), 1)) for r in rows}
    # 逐条 margin:把二值 passed 拆成"每条判据离阈值多远"。判据来自
    # getup_acceptance_margin(与 evaluate_contact_motion 的 good 掩码同源),
    # 这里不重复定义,避免两处漂移。
    margins = {}
    for r in rows:
        m = margin.load_case(r, out_dir / f"{r['key']}.npz")
        if m:
            margins[r['key']] = dict(binding=m['binding'],
                                     binding_frac=round(m['binding_frac'], 3),
                                     worst_value=m['hold_worst'].get(m['binding']),
                                     final_height_mm=m['final_height_mm'],
                                     final_tilt_deg=m['final_tilt_deg'])
    return dict(iteration=count, cases=len(rows), passed=passed, detail=detail, margin=margins)


def progress_score(record):
    """进度标量 = 最接近通过的那一例,其最差判据的达标率。

    整体达标率是 8 条判据的合取,恒为 0 排不出先后;这个标量会随训练动,
    是循环唯一能看到的梯度。"""
    margins = record.get('margin') or {}
    return max((m['binding_frac'] for m in margins.values()), default=0.0)


def replay_progress(history):
    """从已有 progress.json 恢复 (历史最佳 score, 连续无进展轮数)。

    只有带 margin 的记录参与计数;旧格式记录没有梯度信息,不算数也不清零。
    这样重启 watcher 不会把"连续无进展"洗掉。"""
    best, streak = 0.0, 0
    for record in history:
        if not record.get('margin'):
            continue
        score = progress_score(record)
        previous = best
        best = max(best, score)          # 真实最佳,用于如实汇报
        streak = 0 if score >= previous + NO_PROGRESS_TOL else streak + 1
    return best, streak


def print_margins(record):
    """把逐条 margin 打到 stdout —— 这是监控循环的观察位。"""
    margins = record.get('margin') or {}
    if not margins:
        return
    print(f"  margin@iter{record['iteration']}: score={progress_score(record):.3f}"
          f"(最差判据达标率,需 0.95)", flush=True)
    for key, m in sorted(margins.items(), key=lambda kv: -kv[1]['binding_frac']):
        print(f"    {key:14s} 卡在 [{m['binding']}] {m['binding_frac'] * 100:4.0f}%"
              f"   终态 {m['final_tilt_deg']:6.1f}° / {m['final_height_mm']:6.1f}mm", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--iterations', type=int, nargs='+', required=True,
                   help='Only evaluate checkpoints at or after each of these iteration numbers.')
    p.add_argument('--seeds', type=int, nargs='+', default=[941, 942])
    p.add_argument('--delays', type=int, nargs='+', default=[10])
    p.add_argument('--tag', default='getup_watch')
    a = p.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')

    dest = CONTACT / a.tag
    dest.mkdir(parents=True, exist_ok=True)
    log = dest / 'watch.log'
    progress = dest / 'progress.json'
    done = set()
    targets = sorted(a.iterations)
    started = time.time()
    history = json.loads(progress.read_text()) if progress.exists() else []
    best_score, no_progress = replay_progress(history)
    if no_progress:
        print(f'[resume] 历史 best score={best_score:.3f},已连续 {no_progress} 轮无进展', flush=True)
    while True:
        checkpoint = latest_checkpoint(a.run_dir)
        if checkpoint is None:
            time.sleep(30)
            continue
        count = int(checkpoint.stem.split('_')[-1])
        due = [t for t in targets if t not in done and count >= t]
        if not due:
            # Stop once the run is finished and nothing else is due.
            if (a.run_dir / 'candidate.onnx').exists() and not due:
                print('training finished; nothing due', flush=True)
                break
            if time.time() - started > 6 * 3600:
                print('watch timeout', flush=True)
                break
            time.sleep(60)
            continue
        for target in due:
            checkpoint = latest_checkpoint(a.run_dir)
            count = int(checkpoint.stem.split('_')[-1])
            case = dest / f'iter{count:06d}'
            onnx = case / 'policy.onnx'
            case.mkdir(parents=True, exist_ok=True)
            if not onnx.exists():
                rc = run([str(PY), 'export_contact_checkpoint.py', '--skill', 'getup',
                          '--checkpoint', str(checkpoint), '--output', str(onnx)], log)
                if rc != 0 or not onnx.exists():
                    print(f'export failed for {checkpoint.name}', flush=True)
                    done.add(target)
                    continue
            out = f'{a.tag}_iter{count:06d}'
            if not (CONTACT / out / 'summary.json').exists():
                run([str(PY), 'evaluate_contact_motion.py', '--skill', 'getup', '--policy', str(onnx),
                     '--out', out, '--seeds', *[str(s) for s in a.seeds],
                     '--delays', *[str(d) for d in a.delays]], log)
            if (CONTACT / out / 'summary.json').exists():
                record = summarise(CONTACT / out, count)
                rows = json.loads(progress.read_text()) if progress.exists() else []
                rows.append(record)
                progress.write_text(json.dumps(rows, indent=2), encoding='utf-8')
                print(json.dumps({k: record[k] for k in ('iteration', 'cases', 'passed')}), flush=True)
                print_margins(record)

                # 无进展预算(硬门)。这里是工具自己停,不是"提醒一句继续跑"。
                score = progress_score(record)
                previous = best_score
                best_score = max(best_score, score)
                no_progress = 0 if score >= previous + NO_PROGRESS_TOL else no_progress + 1
                print(f'  进度: score={score:.3f} 历史最佳={best_score:.3f} '
                      f'连续无进展={no_progress}/{NO_PROGRESS_LIMIT}', flush=True)
                if no_progress >= NO_PROGRESS_LIMIT:
                    stop = dict(reason='no_progress', limit=NO_PROGRESS_LIMIT, score=score,
                                best_score=best_score, at_iteration=count,
                                binding=[dict(case=k, criterion=m['binding'], frac=m['binding_frac'])
                                         for k, m in sorted((record.get('margin') or {}).items(),
                                                            key=lambda kv: -kv[1]['binding_frac'])],
                                note='按 GETUP_TASK.md 第六节铁律 2 停手:绑判据达标率未抬头,继续加时间不改变结果。')
                    (dest / 'NO_PROGRESS_STOP.json').write_text(
                        json.dumps(stop, ensure_ascii=False, indent=2), encoding='utf-8')
                    print(f'  [STOP] 连续 {NO_PROGRESS_LIMIT} 轮无进展(历史最佳 score={best_score:.3f} 未抬头)。'
                          f'停手,不自动续训;详见 {dest / "NO_PROGRESS_STOP.json"}', flush=True)
                    print('watcher exit', flush=True)
                    return
            done.add(target)
    print('watcher exit', flush=True)


if __name__ == '__main__':
    main()
