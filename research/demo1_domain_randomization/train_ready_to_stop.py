"""Learn a gait whose support phase can be interrupted by a zero command."""

import argparse
from pathlib import Path

import numpy as np

import continue_training as base
from demo1_domain_randomization import train as simulator
from train_foot_capture import FootCapture
from train_stop import StopCampaign, braking_score, rank_cases


STANDARD_METRICS = simulator.metrics


def ready_metrics(rows, release, hold, ended, load_transfers):
    result = STANDARD_METRICS(rows, release, hold, ended, load_transfers)
    support = np.asarray([min(row["loads"]) > .5 for row in rows
                          if .4 <= row["t"] < release - base.EPSILON])
    anchor = np.flatnonzero(np.r_[True, support, True])
    gaps = base.PERIOD * (np.diff(anchor) - 1)
    result["walk"]["p95_non_dual_s"] = float(np.percentile(gaps, 95))
    after = [row for row in rows if row["t"] >= release - base.EPSILON]
    first_overlap = None
    if after:
        original_support = int(np.argmax(after[0]["loads"]))
        for row in after:
            if row["loads"][original_support] <= .5:
                break
            if min(row["loads"]) > .5:
                first_overlap = row["t"] - release
                break
    result["parking"]["first_retained_dual_support_s"] = first_overlap
    return result


def ready_score(row, release, hold):
    score = braking_score(row, release, hold)
    if not row["walk"]["stable"] or not row["parking"]["complete"]:
        return score
    # Penalize unusually long non-dual intervals during gait, even away from
    # the release phase. The policy has no information about the future zero.
    excess = max(0., row["walk"]["p95_non_dual_s"] / .12 - 1.)
    overlap = row["parking"]["first_retained_dual_support_s"]
    overlap_delay = .60 if overlap is None else overlap
    return score - 1.2 * min(excess, 3.) - .65 * min(6., overlap_delay / .06)


class ReadyCampaign(StopCampaign):
    def __init__(self, args):
        args.output.mkdir(parents=True, exist_ok=True)
        seed = args.output / "ready_seed.json"
        marker = args.output / "ready_variant.json"
        identity = dict(method="joint_gait_and_foot_capture", sha256=base.sha(Path(__file__)),
                        seed_sha256=base.sha(seed))
        if marker.exists() and base.read(marker) != identity:
            raise ValueError("Gait-readiness source or seed changed; use a new run directory")
        if not marker.exists():
            base.replace_json(marker, identity)
        super().__init__(args)
        self.seed = np.array(base.read(seed))
        if self.seed.shape != (base.TOTAL,):
            raise ValueError("Expected a 32-parameter continuation seed")
        if self.state["generation"] == 0:
            self.state.update(center=self.seed.tolist(), incumbent=self.seed.tolist(),
                              sigma=np.r_[np.full(16, .025), np.full(16, .037)].tolist())
            base.replace_json(self.state_path, self.state)

    def generation(self):
        i = self.state["generation"]
        rng = np.random.default_rng(base.SEED + 9049 + i * 17)
        release = 4. if i % 4 == 0 else .02 * int(rng.integers(174, 223))
        level = ("nominal", "mild", "moderate")[(i // 4) % 3]
        domain = base.draw_domain(level, rng)
        cases = [(domain, release, step, 3.) for step in (base.BUS_SLOT, base.BUS_SLOT / 2)]
        if i % 3 == 2:
            cases.append((base.draw_domain("nominal", rng), 4., base.BUS_SLOT, 3.))
        center = np.array(self.state["center"])
        sigma = np.array(self.state["sigma"])
        incumbent = np.array(self.state["incumbent"])
        population = [incumbent, self.seed]
        for k in range(base.POPULATION - 2):
            offset = rng.normal(size=base.TOTAL) * sigma
            if k < 3:
                offset[16:] = 0.
            elif k < 5:
                offset[:16] = 0.
            origin = center if k < 6 else self.seed
            population.append(np.clip(origin + offset, -.18, .18))

        ranking = []
        for candidate, weights in enumerate(population):
            trials = self.evaluate(weights, cases, "gait_readiness", candidate)
            values = [row["objective"] for row in trials]
            utility = .55 * float(np.mean(values)) + .45 * min(values)
            ranking.append((utility, weights))
        ranking.sort(key=lambda row: row[0], reverse=True)
        best = ranking[0][1]
        elite = np.asarray([weights for _, weights in ranking[:3]])
        center = .4 * center + .6 * elite.mean(axis=0)
        sigma = np.clip(.75 * sigma + .25 * elite.std(axis=0), .015, .055)
        if i % 12 == 11:
            sigma = np.maximum(sigma, .030)
        self.state.update(center=center.tolist(), sigma=sigma.tolist(), incumbent=best.tolist())
        if i % 3 == 0:
            metrics, rank = rank_cases(self.evaluate(best, self.development, "ready_development", i))
            if rank > tuple(self.state["best"]["rank"]):
                self.state["best"] = dict(generation=i, rank=rank, metrics=metrics, weights=best.tolist())
                base.replace_json(self.output / "candidate.json", self.state["best"])
        self.state.update(generation=i + 1, evaluations=self.count)
        base.replace_json(self.state_path, self.state)
        if i % 6 == 0:
            self.validate(i)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("study", "output", "pilot"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    args = parser.parse_args()
    for name in ("study", "output", "pilot"):
        setattr(args, name, getattr(args, name).resolve())
    if (args.output / "STOP").exists():
        return
    simulator.metrics = ready_metrics
    base.StopResidual = FootCapture
    base.score = ready_score
    try:
        ReadyCampaign(args).run()
    except base.StopRequested:
        return


if __name__ == "__main__":
    main()
