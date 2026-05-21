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

## 172-Feature Retest

Codex generated a fresh 43-cutoff 172-feature manifest using:

```bash
.venv/bin/python scripts/train_walkforward_panel.py \
  --start-date 2023-10-02 \
  --end-date 2026-03-09 \
  --cadence-days 21 \
  --artifact-root walkforward_172_sentiment \
  --manifest-output backtesting/renquant_104/artifacts/sim/walkforward_manifest_172_sentiment.json \
  --jobs 2
```

Then ran the WF gate with a generated sim config pointing at that manifest:

```bash
.venv/bin/python scripts/research_acceptance_104.py \
  --target wf-gate \
  --artifact backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.json \
  --strategy-config strategy_config.sim_wl200_172_sentiment.json \
  --wf-jobs 3
```

Scope was now correct:

- `wf_eval_scope=walkforward_manifest`
- `recipe_validated=true`
- candidate recipe fingerprint: `sha256:31e45b8d2f17e006`
- sampled manifest artifacts: 172 features, same fingerprint, no feature drift

But performance still failed:

| Cut | Sharpe | APY | SPY Sharpe | Delta Sharpe |
| --- | ---: | ---: | ---: | ---: |
| 2024-01-02 to 2024-12-31 | -0.670 | -6.50% | +1.778 | -2.448 |
| 2024-07-01 to 2025-06-30 | -0.370 | -3.30% | +0.715 | -1.085 |
| 2025-04-01 to 2026-03-28 | +0.340 | +10.00% | +0.749 | -0.409 |

Aggregate:

- Mean Sharpe: `-0.233`
- Mean APY: `+0.07%`
- Positive cuts: `1/3`
- Beat SPY Sharpe: `0/3`
- Beat SPY APY: `0/3`
- Mean SPY Sharpe: `+1.081`
- Mean strategy-minus-SPY Sharpe: `-1.314`

Sanity battery:

- Real IC: `+0.0775`
- Shuffled IC: `-0.0016`, passes
- Time-shift placebo IC: `+0.0394`, fails narrowly versus threshold
  `0.5 * real_ic = +0.0388`

Conclusion: the current 172-feature XGB panel-LTR production recipe still
does **not** pass walk-forward promotion. This is no longer a scope bug; it is
a real strategy-performance failure under the current decision tree / sizing /
exit stack.
