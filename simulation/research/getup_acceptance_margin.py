"""起身验收逐条 margin 报告。

把二值的 `passed: 0/8` 拆成"每条判据离阈值还差多远",让监控循环有梯度可看:
单项达标率会先抬头,整体 passed 才会跟着动;只看 passed 永远只能看到 0。

判据与 `evaluate_contact_motion.py` 的 `good` 掩码逐条对应(同一 trace 列):
    tail = trace[trace[:,0] >= 10]          # 最后 2s(physics_dt=0.00125 -> 1600 tick)
    good = 根高∈(0.1035,0.1255) & 倾角<10 & 平速<0.03 & 角速<0.3
           & 双脚力各>0.5 & 非足支地<0.2
    passed = good.mean() >= 0.95            # 最后 2s 的 95%
再 AND 上整段 trace 的门禁: 穿透、关节越限、踝、头俯仰、neck_pitch。

用法:
    python getup_acceptance_margin.py <eval_dir | summary.json> [...]
    python getup_acceptance_margin.py <eval_dir> --json margins.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np

# trace 列(见 evaluate_contact_motion.py 的 good/门禁段)
C_TIME, C_HEIGHT, C_SUNK = 0, 1, 2
C_FOOT_L, C_FOOT_R, C_NONFOOT = 5, 6, 7
C_TILT = 8
C_PLANAR = slice(9, 11)
C_JOINT_EXCESS = 13
C_PENETRATION = 16
C_ANGVEL = 17
C_ANKLE = 18
C_HEAD = 19

HOLD_WINDOW_S = 10.0   # 最后 2s 的起算点(总时长 12s)
HOLD_FRACTION = 0.95   # 最后 2s 需达标的比例

DEFAULT_NECK_LIMIT_DEG = 38.0


def neck_limit_deg() -> float:
    """从 recovery_bounds.py 读唯一真源,读不到就退回文档值并提示。"""
    src = Path("D:/microduck_rl/src/mjlab_microduck/tasks/recovery_bounds.py")
    try:
        m = re.search(r"NECK_BATTERY_TERMINATE_DEG\s*=\s*([0-9.]+)", src.read_text(encoding="utf-8"))
        if m:
            return float(m.group(1))
    except OSError:
        pass
    print(f"  [warn] 未读到 recovery_bounds.py,neck 界退回 {DEFAULT_NECK_LIMIT_DEG}°", file=sys.stderr)
    return DEFAULT_NECK_LIMIT_DEG


def dwidth(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in str(s))


def pad(s: str, n: int, right: bool = False) -> str:
    s = str(s)
    gap = max(0, n - dwidth(s))
    return (" " * gap + s) if right else (s + " " * gap)


def hold_worst(tail: np.ndarray) -> dict:
    """最后 2s 窗口内每条判据的最坏观测值(le 取最大,ge 取最小)。"""
    return {
        "根高下限": float(tail[:, C_HEIGHT].min() * 1000),
        "根高上限": float(tail[:, C_HEIGHT].max() * 1000),
        "倾角": float(tail[:, C_TILT].max()),
        "平速": float(np.linalg.norm(tail[:, C_PLANAR], axis=1).max()),
        "角速": float(tail[:, C_ANGVEL].max()),
        "左脚力": float(tail[:, C_FOOT_L].min()),
        "右脚力": float(tail[:, C_FOOT_R].min()),
        "非足支地": float(tail[:, C_NONFOOT].max()),
    }


def hold_frac(tail: np.ndarray) -> dict:
    """每条判据在窗口内的达标率。"""
    return {name: float(ok.mean()) for name, ok in hold_masks(tail).items()}


def hold_masks(tail: np.ndarray) -> dict:
    """八条窗口判据的逐样本掩码。**这是判据的唯一真源**。

    `evaluate_contact_motion.py` 的 `good` 掩码由这里生成,本模块的 margin 报告也由
    这里生成,两者物理上不可能漂。此前 neck 界就漂过一次:训练侧 clamp 35°、评估侧
    完全没有界,于是评估看得见训练看不见的东西。
    """
    return {
        "根高下限": tail[:, C_HEIGHT] > 0.1035,
        "根高上限": tail[:, C_HEIGHT] < 0.1255,
        "倾角": tail[:, C_TILT] < 10,
        "平速": np.linalg.norm(tail[:, C_PLANAR], axis=1) < 0.03,
        "角速": tail[:, C_ANGVEL] < 0.3,
        "左脚力": tail[:, C_FOOT_L] > 0.5,
        "右脚力": tail[:, C_FOOT_R] > 0.5,
        "非足支地": tail[:, C_NONFOOT] < 0.2,
    }


def good_mask(tail: np.ndarray) -> np.ndarray:
    """八条同时成立的合取 —— `passed` 要求它的均值 >= 0.95。"""
    total = np.ones(len(tail), dtype=bool)
    for ok in hold_masks(tail).values():
        total &= ok
    return total


def gate_frac(name: str, value: float | None, rule: str) -> float:
    """把硬门折算成"达标率",好和窗口判据放在同一把尺子上排序。

    之前 binding 只在 8 条窗口判据里取最小,硬门(穿透/站姿/朝向/头俯仰/neck)完全在
    视野外 —— 于是 `left` 报"卡在右脚力 98.9%",而真正挡住它的是 `yaw_drift 107.7°`
    (达标率 28%)。看那个 98.9% 就是在庆祝一个和通过与否无关的数。这里把两类判据
    合并,绑定判据才是真的上界。
    """
    if value is None:
        return 0.0
    if rule.startswith("<="):
        limit = float(rule[2:].strip().rstrip("°%m").strip() or 0.)
        return 1.0 if value <= limit else min(1.0, limit / value) if value > 0 else 1.0
    if rule.startswith(">="):
        limit = float(rule[2:].strip().rstrip("°%m").strip() or 0.)
        return 1.0 if value >= limit else (value / limit if limit else 1.0)
    if rule.startswith("<"):
        limit = float(rule[1:].strip().rstrip("°%m").strip() or 0.)
        return 1.0 if value < limit else min(1.0, limit / value) if value > 0 else 1.0
    if rule.startswith(">"):
        limit = float(rule[1:].strip().rstrip("°%m").strip() or 0.)
        return 1.0 if value > limit else (value / limit if limit else 1.0)
    return 1.0


def binding(frac: dict) -> tuple[str, float]:
    """绑定判据 = 单独达标率最低的那条;它的达标率是整例通过率的上界。"""
    name = min(frac, key=lambda k: frac[k])
    return name, frac[name]


def load_case(entry: dict, npz: Path) -> dict | None:
    """重算该 case 的逐条判据达标率与余量。"""
    try:
        trace = np.load(npz, allow_pickle=True)["physics"]
    except OSError:
        return None
    tail = trace[trace[:, C_TIME] >= HOLD_WINDOW_S]
    if len(tail) == 0:
        return None

    head = np.rad2deg(trace[:, C_HEAD])
    window_frac = hold_frac(tail)

    # 整段 trace 的门禁(不在 95% 窗口里,是硬门)
    pen = trace[:, C_PENETRATION].max() * 1000
    jex = np.rad2deg(trace[:, C_JOINT_EXCESS].max())
    ankle = np.rad2deg(trace[:, C_ANKLE].max())

    neck_max = entry.get("neck_pitch_max_deg")
    # 末态姿态三条(2026-09-16 加):旧判据全部量不出"站姿对不对" ——
    # 蜷缩躯干的根高 116mm 与 v7 正确站姿 114mm 几乎相同,十二条全过而姿势全错。
    # 这里只读 evaluator 算好的值(它有模型,本模块保持纯 numpy)。
    att = [
        ("站姿", entry.get("stance_max_dev_deg"), "<= 15°",
         entry.get("stance_max_dev_deg") is not None and entry["stance_max_dev_deg"] <= 15.),
        ("脚底平贴", entry.get("sole_tilt_max_deg"), "<= 8°",
         entry.get("sole_tilt_max_deg") is not None and entry["sole_tilt_max_deg"] <= 8.),
        ("朝向保持", entry.get("yaw_drift_deg"), "<= 30°",
         entry.get("yaw_drift_deg") is not None and entry["yaw_drift_deg"] <= 30.),
    ]
    gates = [
        ("穿透", pen, "<= 2 mm", pen <= 2),
        ("关节越限", jex, "<= 0.5°", jex <= 0.5),
        ("踝", ankle, "< 55°", ankle < 55),
        ("头俯仰低", head.min(), ">= -2°", head.min() >= -2),
        ("头俯仰高", head.max(), "<= 29°", head.max() <= 29),
        ("neck_pitch", neck_max, f"<= {neck_limit_deg():.0f}°",
         neck_max is not None and neck_max <= neck_limit_deg()),
        *att,
    ]

    # 绑定判据在**全部**判据里取最小,不再只盯着窗口那 8 条。
    full_frac = dict(window_frac)
    pen_rule = "<= 2 mm"
    full_frac["穿透"] = gate_frac("穿透", pen, pen_rule)
    full_frac["关节越限"] = gate_frac("关节越限", jex, "<= 0.5°")
    full_frac["踝"] = gate_frac("踝", ankle, "< 55°")
    full_frac["头俯仰低"] = gate_frac("头俯仰低", float(head.min()), ">= -2°")
    full_frac["头俯仰高"] = gate_frac("头俯仰高", float(head.max()), "<= 29°")
    full_frac["neck_pitch"] = gate_frac("neck_pitch", neck_max, f"<= {neck_limit_deg():.0f}°")
    for name, v, rule, _ in att:
        full_frac[name] = gate_frac(name, None if v is None else float(v), rule)
    binding_name, binding_value = binding(full_frac)

    return {
        "key": entry["key"],
        "reported_passed": bool(entry.get("passed")),
        "hold_fraction": float(good_mask(tail).mean()),
        "hold_frac": window_frac,
        "all_frac": full_frac,
        "hold_worst": hold_worst(tail),
        "gates": {name: {"value": None if v is None else float(v), "rule": rule, "ok": bool(ok)}
                  for name, v, rule, ok in gates},
        "binding": binding_name,
        "binding_frac": binding_value,
        "start_tilt_deg": float(entry.get("start_tilt_deg", 0.0)),
        "final_tilt_deg": float(entry.get("final_tilt_deg", 0.0)),
        "final_height_mm": float(entry.get("final_height_mm", 0.0)),
        "final_speed_m_s": float(entry.get("final_speed_m_s", 0.0)),
    }


def report(cases: list[dict]) -> None:
    hold_names = list(cases[0]["hold_frac"].keys())

    print("\n" + "=" * 96)
    print("起身验收 · 逐条 margin(最后 2s 各判据的达标率;passed 需要全部 >=95%)")
    print("=" * 96)
    widths = [14] + [max(9, dwidth(n) + 2) for n in hold_names] + [8]
    header = pad("case", widths[0]) + "".join(pad(n, w, True) for n, w in zip(hold_names, widths[1:-1]))
    header += pad("整体", widths[-1], True)
    print(header)
    print("-" * dwidth(header))
    for c in cases:
        row = pad(c["key"], widths[0])
        for n, w in zip(hold_names, widths[1:-1]):
            v = c["hold_frac"][n]
            cell = f"{v * 100:.0f}%" if v < 1.0 else "OK"
            row += pad(cell, w, True)
        row += pad(f"{c['hold_fraction'] * 100:.1f}%", widths[-1], True)
        print(row)

    print("\n" + "-" * 96)
    print("每例的绑定判据(达标率最低的一条):")
    for c in cases:
        state = (f"{c['final_tilt_deg']:6.1f}° / {c['final_height_mm']:6.1f}mm"
                 f" / 末速 {c['final_speed_m_s']:.4f}")
        print(f"  {pad(c['key'], 15)} 起点 {c['start_tilt_deg']:6.1f}° -> {state}"
              f"   卡在 [{c['binding']}]  {c['binding_frac'] * 100:.0f}%")

    print("\n" + "-" * 96)
    print("整段门禁(硬门,不参与 95% 窗口):")
    print(pad("case", 15) + "".join(pad(n, max(10, dwidth(n) + 2), True) for n in cases[0]["gates"]))
    for c in cases:
        row = pad(c["key"], 15)
        for g in c["gates"].values():
            v = g["value"]
            cell = "n/a" if v is None else f"{v:.2f}"
            row += pad(cell if g["ok"] else f"{cell}!", max(10, dwidth(cell) + 4), True)
        print(row)
    print("  (带 ! 的是越界项)")

    for c in cases:
        c["attitude_ok"] = all(g["ok"] for g in list(c["gates"].values())[-3:])
    ok_all = sum(1 for c in cases if c["reported_passed"] and c["attitude_ok"])
    print("\n" + "=" * 96)
    print(f"汇总: {ok_all}/{len(cases)} 通过验收")
    # 整体达标率是 8 条判据的合取,几乎恒为 0,排不出先后;
    # 用"最差那条判据的达标率"当作距离(越大越接近),它才是能动的那个数。
    tight = max(cases, key=lambda c: (c["binding_frac"], c["hold_fraction"]))
    print(f"最接近的是 {tight['key']}:最差判据 [{tight['binding']}] 已达 "
          f"{tight['binding_frac'] * 100:.0f}%(整体合取 {tight['hold_fraction'] * 100:.1f}%,"
          f"需 {HOLD_FRACTION * 100:.0f}%)")
    print("提示:整体达标率是 8 条同时满足的比例,它恒为 0 不代表没进展;"
          "看单条达标率抬头才是梯度。")
    print("=" * 96 + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="起身验收逐条 margin 报告")
    ap.add_argument("targets", nargs="+", help="评估目录或 summary.json")
    ap.add_argument("--json", dest="json_out", help="把 margin 落盘,便于跨轮 diff")
    a = ap.parse_args()

    cases: list[dict] = []
    for t in a.targets:
        p = Path(t)
        summary = p / "summary.json" if p.is_dir() else p
        if not summary.exists():
            print(f"[skip] {summary} 不存在", file=sys.stderr)
            continue
        entries = json.loads(summary.read_text(encoding="utf-8"))
        print(f"[read] {summary}  ({len(entries)} case)")
        for e in entries:
            c = load_case(e, summary.parent / f"{e['key']}.npz")
            if c:
                cases.append(c)
            else:
                print(f"  [skip] {e['key']}: 无 npz 或 trace 为空", file=sys.stderr)

    if not cases:
        print("没有可分析的 case", file=sys.stderr)
        return 1

    report(cases)
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[write] {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
