"""Compare frozen development selections on a fresh physical replay suite."""

import argparse
from pathlib import Path
from statistics import median

import numpy as np

import continue_training as base
from train_foot_capture import FootCapture
from train_stop import WarmStop, braking_score


AUDIT_SEED = 18914453
TRAINERS = {
    "overnight_v2": base.StopResidual,
    "overnight_v3": WarmStop,
    "overnight_v4": FootCapture,
    "overnight_v5": FootCapture,
    "overnight_v6": FootCapture,
}


def audit_suite():
    rng = np.random.default_rng(AUDIT_SEED)
    designs = (
        ("nominal", 4.), ("mild", 4.), ("mild", 4.), ("moderate", 4.), ("moderate", 4.),
        ("mild", 3.68), ("mild", 4.36), ("moderate", 3.68), ("moderate", 4.36),
    )
    scenarios = []
    for severity, release in designs:
        domain = base.draw_domain(severity, rng)
        scenarios.extend((domain, release, dt, 10.) for dt in (base.BUS_SLOT, base.BUS_SLOT / 2))
    return scenarios


def statistics(rows):
    result = base.summary(rows)
    released = [row["metrics"]["parking"] for row in rows if row["metrics"]["parking"]["duration_s"] > 0.]
    fields = ("longest_unloaded_s", "foot_shift_mm", "landings", "drift_mm", "peak_tilt_deg")
    result["median_parking"] = {key: median(row[key] for row in released) for key in fields} if released else {}
    result["completed_at_most_two_landings"] = sum(row["complete"] and row["landings"] <= 2 for row in released)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "output", "pilot", "campaign-root"):
        p.add_argument(f"--{name}", type=Path, required=True)
    args = p.parse_args()
    for name in ("study", "output", "pilot", "campaign_root"):
        setattr(args, name, getattr(args, name).resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    selection = args.output / "selection.json"
    if selection.exists():
        profiles = base.read(selection)
    else:
        profiles = {"frozen": {"weights": np.zeros(base.TOTAL).tolist(), "generation": None}}
        for name in TRAINERS:
            champion = base.read(args.campaign_root / name / "checkpoint.json")["best"]
            profiles[name] = {"weights": champion["weights"], "generation": champion["generation"]}
        base.replace_json(selection, profiles)
    base.score = braking_score
    runner = base.Campaign(args)
    suite = audit_suite()
    results = {}
    try:
        for index, (name, profile) in enumerate(profiles.items()):
            runner.stop = TRAINERS.get(name, FootCapture)(runner.box, args.study)
            rows = runner.evaluate(np.asarray(profile["weights"]), suite, name, index)
            main = [row for row in rows if row["release_s"] == 4.]
            phases = [row for row in rows if row["release_s"] != 4.]
            results[name] = dict(selected_generation=profile["generation"], primary=statistics(main),
                                 other_release_phases=statistics(phases), rows=rows)
            base.replace_json(args.output / "report.json", {
                "scope": "Fresh paired audit on frozen development selections; zero command from release through 10 s",
                "audit_seed": AUDIT_SEED,
                "results": results,
            })
    except base.StopRequested:
        runner.heartbeat("stopped")
        return
    runner.heartbeat("completed")


if __name__ == "__main__":
    main()
