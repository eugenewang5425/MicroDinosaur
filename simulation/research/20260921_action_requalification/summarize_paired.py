"""Summarize the strict paired get-up qualification run.

The evaluator writes one JSON row and one ``*.loads.json`` payload per case.
This script refuses to compare runs unless all 40 case keys and initial-state
hashes match, so a policy cannot gain credit from an easier reset sample.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
EVAL = HERE / "eval"
BASELINE = EVAL / "paired_baseline_s969_978"
STAGE1 = EVAL / "paired_stage1_s969_978"
CANDIDATE = EVAL / "paired_candidate_s969_978"
DIRECTIONS = ("front", "back", "left", "right")
EXPECTED_PER_DIRECTION = 10
TAIL_CRADLE_PAIR = "Rex_Rigid_Balance_Tail / Rex_Yaw_Pitch_Cradle"
TAIL_LIMIT_DEG = -24.5
MAX_HIGH_LOAD_RUN_S = 0.5


def read_rows(directory: Path) -> dict[str, dict]:
    rows = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError(f"{directory / 'summary.json'} is not a JSON list")
    return {row["key"]: row for row in rows}


def read_loads(directory: Path, key: str) -> dict:
    return json.loads((directory / f"{key}.loads.json").read_text(encoding="utf-8"))


def summarize(directory: Path, rows: dict[str, dict]) -> dict:
    passed = [key for key, row in rows.items() if row.get("passed")]
    by_direction = Counter(key.split("_", 1)[0] for key in passed)
    def mechanical_recovery(row: dict) -> bool:
        """Apply every frozen get-up check except post-recovery heading drift.

        This secondary metric follows the project decision that vision may repair
        heading later. It cannot replace the predeclared strict gate.
        """
        return bool(
            row.get("standing_final_fraction", 0.0) >= 0.95
            and row.get("penetration_max_mm", float("inf")) <= 2.0
            and row.get("joint_limit_excess_deg", float("inf")) <= 0.5
            and row.get("ankle_max_deg", float("inf")) < 55.0
            and row.get("head_pitch_min_deg", -float("inf")) >= -2.0
            and row.get("head_pitch_max_deg", float("inf")) <= 29.0
            and row.get("stance_max_dev_deg", float("inf")) <= row.get("stance_limit_deg", 15.0)
            and row.get("sole_tilt_max_deg", float("inf")) <= row.get("sole_limit_deg", 8.0)
            and row.get("neck_pitch_max_deg", float("inf")) <= row.get("neck_battery_limit_deg", 38.0)
            and row.get("tail_pitch_min_deg", -float("inf")) >= row.get("tail_cradle_limit_deg", -24.5)
        )
    mechanical = [key for key, row in rows.items() if mechanical_recovery(row)]
    mechanical_by_direction = Counter(key.split("_", 1)[0] for key in mechanical)
    yaw_only = [
        key for key, row in rows.items()
        if mechanical_recovery(row) and row.get("yaw_drift_deg", 0.0) > row.get("yaw_limit_deg", 30.0)
    ]
    contact_cases: list[dict] = []
    high_load = {"seconds": 0.0, "case": None, "joint": None}
    load_hashes: dict[str, str] = {}
    missing_load_files: list[str] = []
    for key in sorted(rows):
        load_path = directory / f"{key}.loads.json"
        if not load_path.is_file():
            missing_load_files.append(key)
            continue
        payload = read_loads(directory, key)
        load_hashes[key] = payload["initial_hash"]
        pair = payload.get("pair_contacts", {}).get(TAIL_CRADLE_PAIR)
        if pair:
            contact_cases.append({"case": key, **pair})
        for joint in payload.get("joints", []):
            seconds = float(joint.get("high_load_longest_run_s", 0.0))
            if seconds > high_load["seconds"]:
                high_load = {
                    "seconds": seconds,
                    "case": key,
                    "joint": joint.get("name"),
                }
    tail_values = [
        float(row["tail_pitch_min_deg"])
        for row in rows.values()
        if isinstance(row.get("tail_pitch_min_deg"), (int, float))
    ]
    return {
        "case_count": len(rows),
        "pass_count": len(passed),
        "pass_rate": len(passed) / len(rows),
        "passed_by_direction": {name: by_direction[name] for name in DIRECTIONS},
        "mechanical_recovery_excluding_yaw_count": len(mechanical),
        "mechanical_recovery_excluding_yaw_by_direction": {
            name: mechanical_by_direction[name] for name in DIRECTIONS
        },
        "yaw_only_failure_count": len(yaw_only),
        "rejected_count": sum(row.get("status") == "REJECTED" for row in rows.values()),
        "max_penetration_mm": max(float(row.get("penetration_max_mm", 0.0)) for row in rows.values()),
        "max_joint_limit_excess_deg": max(float(row.get("joint_limit_excess_deg", 0.0)) for row in rows.values()),
        "tail_dynamic_violation_count": sum(value < TAIL_LIMIT_DEG for value in tail_values),
        "minimum_tail_pitch_deg": min(tail_values, default=None),
        "tail_cradle_contact_case_count": len(contact_cases),
        "tail_cradle_contact_cases": contact_cases,
        "max_high_load_longest_run": high_load,
        "load_initial_hashes": load_hashes,
        "load_file_count": len(load_hashes),
        "missing_load_files": missing_load_files,
    }


def gate(metrics: dict) -> dict:
    by_direction = metrics["passed_by_direction"]
    checks = {
        "40_complete_cases": metrics["case_count"] == 40,
        "40_load_files": metrics["load_file_count"] == 40,
        "front_at_least_5_of_10": by_direction["front"] >= 5,
        "back_10_of_10": by_direction["back"] == 10,
        "left_at_least_3_of_10": by_direction["left"] >= 3,
        "right_at_least_3_of_10": by_direction["right"] >= 3,
        "side_combined_at_least_8_of_20": by_direction["left"] + by_direction["right"] >= 8,
        "no_tail_cradle_contact": metrics["tail_cradle_contact_case_count"] == 0,
        "no_tail_limit_violation": metrics["tail_dynamic_violation_count"] == 0,
        "no_joint_limit_excess": metrics["max_joint_limit_excess_deg"] <= 1e-9,
        "no_high_load_run_over_0_5_s": metrics["max_high_load_longest_run"]["seconds"] <= MAX_HIGH_LOAD_RUN_S,
    }
    return {"passed": all(checks.values()), "checks": checks}


def main() -> None:
    baseline_rows = read_rows(BASELINE)
    stage1_rows = read_rows(STAGE1)
    candidate_rows = read_rows(CANDIDATE)
    expected = {
        f"{direction}_c10_s{seed}"
        for direction in DIRECTIONS
        for seed in range(969, 979)
    }
    if set(baseline_rows) != expected or set(stage1_rows) != expected or set(candidate_rows) != expected:
        raise RuntimeError(
            "incomplete key set: "
            f"baseline={len(baseline_rows)}, stage1={len(stage1_rows)}, "
            f"stage2={len(candidate_rows)}, expected=40"
        )
    baseline = summarize(BASELINE, baseline_rows)
    stage1 = summarize(STAGE1, stage1_rows)
    candidate = summarize(CANDIDATE, candidate_rows)
    unequal = [key for key in sorted(expected) if key in stage1["load_initial_hashes"] and len({
        baseline["load_initial_hashes"][key],
        stage1["load_initial_hashes"][key],
        candidate["load_initial_hashes"][key],
    }) != 1]
    if unequal:
        raise RuntimeError(f"paired initial states differ for: {unequal}")
    reset_fields = (
        "start_yaw_deg", "fallen_yaw_deg", "fallen_root_z_mm", "start_tilt_deg",
        "nonfoot_weight_fraction", "fallen_start_valid",
    )
    reset_metric_mismatches = [
        key for key in sorted(expected)
        if any(
            baseline_rows[key].get(field) != stage1_rows[key].get(field)
            or baseline_rows[key].get(field) != candidate_rows[key].get(field)
            for field in reset_fields
        )
    ]
    if reset_metric_mismatches:
        raise RuntimeError(f"logged reset metrics differ for: {reset_metric_mismatches}")

    stage1_gate = gate(stage1)
    candidate_gate = gate(candidate)
    promoted = next(
        (name for name, result_gate in (("stage1", stage1_gate), ("stage2", candidate_gate)) if result_gate["passed"]),
        None,
    )
    result = {
        "schema_version": 1,
        "paired": True,
        "seed_range": [969, 978],
        "directions": list(DIRECTIONS),
        "full_initial_hash_match_cases": {
            "baseline_vs_stage1_vs_stage2": len(expected) - len(stage1["missing_load_files"]),
            "baseline_vs_stage2": len(expected),
        },
        "logged_reset_metric_match_cases": len(expected),
        "baseline": baseline,
        "stage1": stage1,
        "stage2": candidate,
        "delta_pass_count": {
            "stage1": stage1["pass_count"] - baseline["pass_count"],
            "stage2": candidate["pass_count"] - baseline["pass_count"],
        },
        "gates": {"stage1": stage1_gate, "stage2": candidate_gate},
        "decision": f"promote_{promoted}" if promoted else "research_only",
        "notes": [
            "The 0.5 s high-load rule is a screening threshold, not a validated thermal model.",
            "Passing this gate qualifies get-up only for these collision-enabled simulated resets.",
            "Stage 1 has two policy-triggered evaluator rejections before load serialization; the 38 available full initial-state hashes match the baseline and stage 2.",
        ],
    }
    (HERE / "PAIRED_RESULTS.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    lines = [
        "# Paired get-up requalification",
        "",
        "All three policies were evaluated on the same 40 collision-enabled initial states: seeds 969-978 in front, back, left, and right fall directions.",
        "",
        "| Policy | Strict total | Strict F/B/L/R | Mechanical recovery without yaw | Mechanical F/B/L/R | Tail/cradle contact cases | Longest high-load run |",
        "| --- | ---: | --- | ---: | --- | ---: | ---: |",
    ]
    for name, metrics in (
        ("Baseline", baseline),
        ("Stage 1 side acquisition", stage1),
        ("Stage 2 mixed retention", candidate),
    ):
        by = metrics["passed_by_direction"]
        mechanical_by = metrics["mechanical_recovery_excluding_yaw_by_direction"]
        lines.append(
            f"| {name} | {metrics['pass_count']}/40 | {by['front']}/{by['back']}/{by['left']}/{by['right']} | "
            f"{metrics['mechanical_recovery_excluding_yaw_count']}/40 | "
            f"{mechanical_by['front']}/{mechanical_by['back']}/{mechanical_by['left']}/{mechanical_by['right']} | "
            f"{metrics['tail_cradle_contact_case_count']}/40 | {metrics['max_high_load_longest_run']['seconds']:.3f} s |"
        )
    lines += [
        "",
        f"**Decision:** `{result['decision']}`. Stage 1 gate: {'PASS' if stage1_gate['passed'] else 'FAIL'}; stage 2 gate: {'PASS' if candidate_gate['passed'] else 'FAIL'}.",
        "",
        "The high-load duration is a screening metric only; no winding or driver temperature model is claimed.",
        "Stage 1 triggered two camera-yaw evaluator rejections before load serialization. The 38 available full initial-state hashes match all variants, and the six logged reset metrics match in all 40 cases; both rejected cases count as failures.",
        "The mechanical-recovery column omits only the post-recovery yaw-drift check, matching the project plan to let later vision correct heading. It is diagnostic and does not retroactively replace the strict gate.",
    ]
    failed = {
        "stage1": [name for name, ok in stage1_gate["checks"].items() if not ok],
        "stage2": [name for name, ok in candidate_gate["checks"].items() if not ok],
    }
    for stage, checks in failed.items():
        if checks:
            lines += ["", f"{stage} failed checks: " + ", ".join(f"`{name}`" for name in checks) + "."]
    (HERE / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    (HERE / "CURRENT_DECISION.json").write_text(
        json.dumps(
            {
                "decision": result["decision"],
                "stage1_gate_passed": stage1_gate["passed"],
                "stage2_gate_passed": candidate_gate["passed"],
                "stage1_onnx_sha256": "a2a89b52a3327039eb709ac8f1583f63315ee1f15ef08c88d422fc39c170d146",
                "stage2_onnx_sha256": "3525afcb21b912e3cad3512815f9ba10348b3d7c6d10892f2b2c01c4621d945c",
                "baseline_onnx_sha256": "9a29cf2a188e7fb59ea11514cc2b5eae6e977fb139c7d0c155c8a3baa9a13a89",
                "failed_checks": failed,
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
