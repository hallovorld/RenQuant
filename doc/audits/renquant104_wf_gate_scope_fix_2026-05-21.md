# renquant_104 WF Gate Scope Fix — 2026-05-21

## Finding

Cut-level parallelism works, but the WF gate had a deeper scope bug:

- A static production artifact trained on 2026-05-18 cannot be replayed into
  2024/2025 sim windows. The leakage guard correctly rejects that path.
- The historical walk-forward path evaluates a manifest of point-in-time
  retrains, not the static artifact itself.
- Therefore a manifest-scoped WF result is valid promotion evidence only when
  the manifest artifacts match the candidate artifact's training recipe.

The current prod artifact has 172 features. The existing
`strategy_config.sim_wl200.json` manifest samples have 169 features and are
missing:

- `mean_sentiment`
- `n_articles_log`
- `sentiment_pos_share`

So the existing manifest is not valid evidence for the current production
recipe.

## Fix

`scripts/run_wf_gate.py` now distinguishes two valid scopes:

1. `static_artifact`: the sim config directly loads the candidate artifact.
   This is only valid when the leakage guard allows the eval window.
2. `walkforward_manifest`: the sim config loads a historical retrain manifest.
   This is valid only when sampled manifest artifacts match the candidate
   recipe fingerprint.

The recipe fingerprint covers:

- model kind
- ordered feature columns
- label column
- lookahead horizon
- learner params

If the manifest recipe does not match, the gate fails before spending sim
compute.

## Current Verdict

The corrected gate rejects the current prod artifact against the old manifest
because the recipe does not match. This is the right failure mode.

Next required work is to generate a fresh 172-feature walk-forward manifest,
then rerun:

```bash
.venv/bin/python scripts/research_acceptance_104.py \
  --target wf-gate \
  --artifact backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.json \
  --wf-jobs 3
```

Only after that manifest passes can the weekly promote path produce meaningful
WF evidence for the current production recipe.
