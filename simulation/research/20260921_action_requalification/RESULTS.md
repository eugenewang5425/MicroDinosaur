# Paired get-up requalification

All three policies were evaluated on the same 40 collision-enabled initial states: seeds 969-978 in front, back, left, and right fall directions.

| Policy | Strict total | Strict F/B/L/R | Mechanical recovery without yaw | Mechanical F/B/L/R | Tail/cradle contact cases | Longest high-load run |
| --- | ---: | --- | ---: | --- | ---: | ---: |
| Baseline | 15/40 | 2/10/0/3 | 28/40 | 9/10/2/7 | 0/40 | 2.969 s |
| Stage 1 side acquisition | 17/40 | 4/10/0/3 | 26/40 | 7/10/4/5 | 0/40 | 5.290 s |
| Stage 2 mixed retention | 18/40 | 5/10/1/2 | 29/40 | 8/10/5/6 | 1/40 | 0.960 s |

**Decision:** `research_only`. Stage 1 gate: FAIL; stage 2 gate: FAIL.

The high-load duration is a screening metric only; no winding or driver temperature model is claimed.
Stage 1 triggered two camera-yaw evaluator rejections before load serialization. The 38 available full initial-state hashes match all variants, and the six logged reset metrics match in all 40 cases; both rejected cases count as failures.
The mechanical-recovery column omits only the post-recovery yaw-drift check, matching the project plan to let later vision correct heading. It is diagnostic and does not retroactively replace the strict gate.

stage1 failed checks: `40_load_files`, `front_at_least_5_of_10`, `left_at_least_3_of_10`, `side_combined_at_least_8_of_20`, `no_tail_limit_violation`, `no_joint_limit_excess`, `no_high_load_run_over_0_5_s`.

stage2 failed checks: `left_at_least_3_of_10`, `right_at_least_3_of_10`, `side_combined_at_least_8_of_20`, `no_tail_cradle_contact`, `no_tail_limit_violation`, `no_joint_limit_excess`, `no_high_load_run_over_0_5_s`.
