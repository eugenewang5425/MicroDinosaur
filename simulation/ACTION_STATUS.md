# Action qualification status

Updated 2026-09-21. “Pass” below always means a bounded simulation condition,
not hardware qualification. A training reward increase is never counted as a
pass by itself.

| Action | Current status | Evidence | Practical meaning |
| --- | --- | --- | --- |
| Basic walking with body/head IMU loop | Bounded simulation reference | 14/14 packaged head/heading tests pass; historical 12 s v7 trials did not fall | Suitable as the retained locomotion reference, pending hardware tests |
| 25 mm squat and return | Pass in the tested flat-ground condition | Collision off 20/20; collision on 20/20, with identical paired labels | This is a shallow reference squat, not the requested fully folded posture |
| Fully folded special squat | Not qualified | The earlier folded candidate exceeded its joint gate and did not complete the full acceptance matrix | Do not schedule as a deployable action |
| Get-up from physical falls | Research only | Latest paired test: strict 18/40; mechanical recovery without heading 29/40. Front/back/left/right strict: 5/10, 10/10, 1/10, 2/10 | Stage 2 improved the baseline strict score from 15/40 but side recovery remains unreliable; no policy was promoted |
| Fast running | Not qualified | Strict collision-off/on comparison was 0/20 in both modes; the archived “fast walk” has no confirmed flight phase | Keep labeled fast walk, not run |
| Small jump with return | Research only | 7/9 on new nominal initial states; 3/7 under stress | Foot flight exists, but joint-limit and robustness gates fail |
| Stairs / continuous high steps | Deferred | No current release gate | Resume after locomotion and recovery actions meet their own gates |

The get-up report also separates post-recovery yaw from mechanical recovery.
This follows the project decision that later visual navigation may correct
heading, while joint limits, self-contact, stance, sole angle, head/battery
clearance and tail clearance still remain mandatory. Omitting yaw raises the
latest result to 29/40, only one case above the 28/40 baseline, so it does not
justify promotion.

See [the paired get-up evidence](research/20260921_action_requalification/RESULTS.md),
[the collision on/off audit](../README.md#校验与边界), and
[the archived run/jump report](research/20260914_run_jump/RESULTS.md).
