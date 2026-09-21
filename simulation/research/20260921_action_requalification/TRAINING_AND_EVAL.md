# Get-up requalification recipe

This folder records the 2026-09-21 bounded continuation experiment on the
v07.1+r7 collision model. It does not replace a released policy.

## Frozen contract

- MuJoCo physics step: 1.25 ms.
- S288 protocol profile at 12 V, including the project motor curve.
- Policy I/O: 81 observations and 19 actions.
- Positive command/position/velocity delays; evaluation uses 10 ms command
  delay and the existing feedback contract.
- Collision plant SHA256:
  `67ea4d2af69b3172b40016d379b31d34f4aecc1e22a3d11740b465f667c4b98d`.
- Reward, termination, joint limits, head range and tail limit remain frozen.
  ERPO is not used.

## Continuation

The starting checkpoint is the previous get-up research baseline. Its exported
ONNX is `getup_baseline.onnx`, SHA256
`9a29cf2a188e7fb59ea11514cc2b5eae6e977fb139c7d0c155c8a3baa9a13a89`.

1. **Side acquisition:** 32 environments, 100 updates, seed 984, fixed learning
   rate `5e-6`, using `side_getup_reset_bank_v071_r7s_stage1.json`.
2. **Mixed retention:** resume stage 1; 32 environments, 100 updates, seed 985,
   fixed learning rate `5e-6`, using
   `side_getup_reset_bank_v071_r7s_stage2.json`.

The two stages add 153,600 simulated transitions in total. Optimizer state and
the learned exploration parameters are retained across continuation. The stage
1 ONNX is `getup_stage1_side_candidate.onnx`, SHA256
`a2a89b52a3327039eb709ac8f1583f63315ee1f15ef08c88d422fc39c170d146`.
The stage 2 ONNX is `getup_two_stage_research_candidate.onnx`, SHA256
`3525afcb21b912e3cad3512815f9ba10348b3d7c6d10892f2b2c01c4621d945c`.

The reset banks and their generator are included. The generator depends on the
larger internal experiment worktree, while the generated banks are self-contained
JSON evidence. Full PPO checkpoints and raw TensorBoard logs are intentionally
excluded from the public repository, so exact optimizer-state retraining cannot
be resumed from this package; an ONNX file is inference-only.

## Independent paired evaluation

Both exported policies use the exact same 40 collision-enabled initial states:
seeds 969-978 for front, back, left and right falls. Every `*.loads.json` file
contains an `initial_hash`; `summarize_paired.py` refuses to compare the runs if
any key or hash differs.

Rebuild the published summary with standard Python:

```bash
python simulation/research/20260921_action_requalification/summarize_paired.py
```

The result and decision are in [RESULTS.md](RESULTS.md) and
[`PAIRED_RESULTS.json`](PAIRED_RESULTS.json). High-load duration is only a
screening metric; this package has no validated driver or winding thermal model.
