"""训练守护:崩溃自动重启 + 断点续训。

为什么需要(2026-09-18 实测):terrain generator 路径在本机 MuJoCo Warp 上
偶发 Non-finite(三次,均在 33-70 迭代早期,不同 env id,随机化参数检查无异常),
而 512 env 的样本吞吐是 256 env 的 1.65 倍(4082 vs 2477 条/秒),不该为偶发
崩溃牺牲吞吐。所以:回 512 + 崩溃后自动从最新 checkpoint 续训。

用法:
  python train_watchdog.py --out <目录> --envs 512 --iterations 600 \
      --difficulty 2.0 [--config stair_climb_cfg] [--max-restarts 6]

行为:
  - 首次用 --out 原目录;崩溃后改用 <out>_r1/_r2…,源指向上一轮最新 checkpoint;
  - 每轮都传入 save-interval(默认 10),保证崩溃时一定有 checkpoint 可续;
  - 只有产出 candidate.onnx 才算成功;超过 --max-restarts 或连续两轮"零进展"
    (最新 checkpoint 迭代号没变)则停手上报,不无限重启。
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
PY = Path('D:/microduck_rl/.venv/Scripts/python.exe')


def latest_checkpoint(folder: Path):
    files = sorted(folder.glob('model_*.pt'),
                   key=lambda p: int(p.stem.split('_')[-1]))
    return files[-1] if files else None


def last_iteration(log: Path):
    if not log.exists():
        return None
    txt = log.read_text(encoding='utf-8', errors='ignore')
    m = re.findall(r'Learning iteration (\d+)/(\d+)', txt)
    return int(m[-1][0]) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--envs', type=int, default=512)
    ap.add_argument('--iterations', type=int, required=True)
    ap.add_argument('--difficulty', type=float, default=None)
    ap.add_argument('--config', default='stair_climb_cfg')
    ap.add_argument('--script', default='train_stair_climb.py',
                    help='被守护的训练脚本(如 train_single_leg.py)')
    ap.add_argument('--save-interval', type=int, default=10)
    ap.add_argument('--max-restarts', type=int, default=6)
    ap.add_argument('--seed', type=int, default=47)
    a, extra = ap.parse_known_args()   # 任务专属参数(--target-height 等)原样透传
    sys.stdout.reconfigure(encoding='utf-8')
    base = a.out.resolve()
    source = os.environ.get('STAIR_SOURCE')
    initial_source = source          # 启动期崩溃(第1次 step 就 NaN)没有 checkpoint,
                                     # 要用原源重试,不能当成"无进展"停手
    best_iter = -1
    stalls = 0
    for attempt in range(a.max_restarts + 1):
        out = base if attempt == 0 else base.with_name(f'{base.name}_r{attempt}')
        log = ROOT / f'wd_{out.name}.log'
        if out.exists() and (out / 'candidate.onnx').exists():
            print(f'[wd] {out.name} 已完成', flush=True)
            break
        cmd = [str(PY), a.script, '--out', str(out),
               '--envs', str(a.envs), '--iterations', str(a.iterations),
               '--seed', str(a.seed), '--save-interval', str(a.save_interval)]
        if a.difficulty is not None:
            cmd += ['--difficulty', str(a.difficulty)]
        cmd += extra
        env = dict(os.environ)
        if source:
            env['STAIR_SOURCE'] = source
        print(f'[wd] 第 {attempt + 1} 轮: {out.name}  源={source or "(默认)"}',
              flush=True)
        with open(log, 'w', encoding='utf-8') as fh:
            rc = subprocess.run(cmd, cwd=str(ROOT), stdout=fh,
                                stderr=subprocess.STDOUT, env=env).returncode
        if rc == 0 and (out / 'candidate.onnx').exists():
            print(f'[wd] 成功: {out}/candidate.onnx', flush=True)
            (base.parent / f'{base.name}_WATCHDOG.json').write_text(json.dumps(
                dict(status='COMPLETE', final=str(out), attempts=attempt + 1),
                indent=2), encoding='utf-8')
            return
        # 崩溃:找最新 checkpoint 续训
        ck = latest_checkpoint(out) or latest_checkpoint(base)
        it = last_iteration(log)
        print(f'[wd] 第 {attempt + 1} 轮退出码 {rc},日志末迭代 {it},'
              f'最新 checkpoint {ck.name if ck else "无"}', flush=True)
        if ck is None:
            # 启动期偶发崩溃:本轮一个 checkpoint 都没存下,用原源重试
            if attempt < a.max_restarts:
                print(f'[wd] 本轮无 checkpoint(疑似启动期偶发 NaN),用原源重试',
                      flush=True)
                source = initial_source
                time.sleep(10)
                continue
            print('[wd] 无 checkpoint 可续且达重试上限,停手上报', flush=True)
            break
        this_iter = int(ck.stem.split('_')[-1])
        if this_iter <= best_iter:
            stalls += 1
            if stalls >= 2:
                print(f'[wd] 连续 {stalls} 轮无净进展(迭代号未前进),停手上报。'
                      f'崩溃可能是确定性的,不是偶发。', flush=True)
                (base.parent / f'{base.name}_WATCHDOG.json').write_text(json.dumps(
                    dict(status='NO_PROGRESS', attempts=attempt + 1,
                         last_checkpoint=str(ck), stall_rounds=stalls,
                         note='崩溃非偶发时不该继续重启'), indent=2),
                    encoding='utf-8')
                return
        else:
            best_iter, stalls = this_iter, 0
        source = str(ck)
        time.sleep(10)
    (base.parent / f'{base.name}_WATCHDOG.json').write_text(json.dumps(
        dict(status='STOPPED', attempts=a.max_restarts + 1), indent=2),
        encoding='utf-8')
    print('[wd] 达到重启上限,停手上报', flush=True)


if __name__ == '__main__':
    main()
