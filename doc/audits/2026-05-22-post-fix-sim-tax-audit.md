# RenQuant 104 Post-Fix Sim + Tax Audit

Date: 2026-05-22
Owner: Codex
Scope: renquant_104 production XGB/panel-LTR path, strict contract preflight, walk-forward sim, and tax ledger forensic output.

## Executive Verdict

The code/contracts are materially cleaner, but the model is not tradeable.

Correct 172-feature walk-forward validation failed hard:

| Cut | Sharpe | APY | SPY Sharpe | Delta Sharpe |
|---|---:|---:|---:|---:|
| 2024-01-02 to 2024-12-31 | -0.420 | +2.60% | +1.778 | -2.198 |
| 2024-07-01 to 2025-06-30 | -1.190 | +0.50% | +0.715 | -1.905 |
| 2025-04-01 to 2026-03-28 | -2.360 | -1.20% | +0.749 | -3.109 |
| Mean | -1.323 | +0.63% | +1.081 | -2.404 |

Result: `FAIL`, 0/3 cuts beat SPY on Sharpe or APY. Production preflight now fails closed on `P-WF-GATE`, so live buys are blocked until a WF-passing artifact is promoted.

## Important Validation Correction

The default `strategy_config.sim_wl200.json` walk-forward manifest is not valid evidence for the current production artifact.

The runner refused it because the manifest recipe has 169 features while the production artifact has 172 features:

- missing in manifest: `mean_sentiment`, `n_articles_log`, `sentiment_pos_share`
- current candidate recipe fingerprint: `sha256:31e45b8d2f17e006`
- 169-feature manifest fingerprint: `sha256:25a3e23d73af9367`

The valid post-fix run used:

- config: `strategy_config.sim_wl200_172_sentiment.json`
- manifest: `artifacts/sim/walkforward_manifest_172_sentiment.json`
- trace dir: `backtesting/renquant_104/artifacts/diagnostics/post_fix_20260522/wf_traces_172_sentiment`

This matters: any previous claim that evaluated the current 172-feature production model using the 169-feature manifest is not comparable evidence.

## Tax Finding

The actual sim cash/tax path is now clean at the sell-event level:

- no sell event taxed a net loss
- maximum event-level `tax / gross_gain` was `0.50`, matching the configured short-term tax rate
- no non-finite gross/tax/net values in the regenerated round-trip ledger

The remaining `gross < tax` symptom was in the forensic round-trip CSV, not the simulator cash ledger. Root cause: `scripts/sim_trade_ledger.py` allocated one sell event's tax across matched lots by share count. A mixed-lot sell could therefore assign positive tax to a losing lot, or make a tiny winning lot show `tax > gross`.

Fix: allocate event-level tax only across profitable matched lots, proportional to positive gross P&L. This preserves the simulator's total tax debit while making every row economically sane.

Post-fix regenerated ledger:

| Scope | Closed Round Trips | Gross PnL | Tax | Net After Tax | tax > positive gross rows | tax on loss rows | non-finite gross/tax/net |
|---|---:|---:|---:|---:|---:|---:|---:|
| pooled | 164 | +16,351.08 | 13,916.32 | +2,434.76 | 0 | 0 | 0 |

Per-cut regenerated ledger:

| Cut | Closed | Gross PnL | Tax | Net After Tax | max tax/gross on gains |
|---|---:|---:|---:|---:|---:|
| 2024-01-02 to 2024-12-31 | 84 | +10,669.73 | 7,259.07 | +3,410.66 | 0.50 |
| 2024-07-01 to 2025-06-30 | 47 | +4,231.76 | 3,783.08 | +448.68 | 0.50 |
| 2025-04-01 to 2026-03-28 | 33 | +1,449.59 | 2,874.17 | -1,424.59 | 0.50 |

Interpretation: tax reporting is no longer nonsensical, but tax drag is severe because the strategy realizes short-term gains while still suffering stop-loss/max-hold drawdowns. The poor Sharpe is not a tax accounting illusion.

## Why The Strategy Still Loses

Trade attribution is concentrated in BULL regimes, where this strategy should work if the ranker is useful:

| Entry Regime | Round Trips | Gross PnL | Tax | Net After Tax | Win Rate | Median Hold |
|---|---:|---:|---:|---:|---:|---:|
| BULL_CALM | 152 | +15,814.20 | 13,514.37 | +2,299.83 | 63.16% | 22d |
| BULL_VOLATILE | 12 | +536.88 | 401.95 | +134.93 | 83.33% | 18d |

The problem is not that the model only trades bear markets. It trades mostly BULL_CALM and still has negative Sharpe because:

- APY is too small versus realized volatility and drawdown.
- Many wins are short-term and taxed immediately.
- Worst losses are ordinary BULL_CALM names, e.g. MSFT, PANW, WFC, BAC, MO, PM.
- Exit reasons on large losers are mainly `stop_loss`, `max_hold`, and `panel_conviction`; this points to weak entry edge and/or exit timing, not just one broken tax calculation.

## Fixes Landed In This Pass

Code/contracts:

- `kernel/preflight.py`: added `P-WF-GATE`; a known failed WF artifact hard-fails live decisions.
- `scripts/train_104.py`: dry-run now enforces hard preflight failures instead of silently succeeding.
- `kernel/preflight.py`: PatchTST `.pt` shadow artifacts are validated as binary sequence checkpoints with summary JSON instead of being decoded as UTF-8 panel JSON.
- `kernel/calibrator_quality.py` and `preflight.py`: fixed flat-region diagnostics and made expected-return flat regions hard-fail.
- `training_panel/global_calibrator.py`: expected-return head now uses robust smooth bounded calibration instead of hard clipping plateaus; save-time flat-region guard added.
- `scripts/fit_calibrator_alpha158_fund.py`: expected-return labels must come from raw return units, not standardized ranking labels.
- `scripts/sim_trade_ledger.py`: forensic tax allocation now avoids impossible lot-level tax rows.
- `strategy_config.shadow.json`: shadow risk/contract fields aligned with production for the patched checks.

Artifacts:

- refit production calibrator: `artifacts/prod/panel-rank-calibration.json`
- refit shadow calibrator: `artifacts/shadow/panel-rank-calibration.shadow.json`
- stamped production artifact with failed 172-feature WF metadata

## Verification

Commands run:

```bash
python scripts/run_wf_gate.py \
  --artifact backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.json \
  --strategy-config strategy_config.sim_wl200_172_sentiment.json \
  --skip-sanity \
  --jobs 3 \
  --trace-dir artifacts/diagnostics/post_fix_20260522/wf_traces_172_sentiment

python scripts/train_104.py --dry-run --strategy-config-name strategy_config.json
python scripts/train_104.py --dry-run --strategy-config-name strategy_config.shadow.json
python -m pytest -q
```

Results:

- WF verdict: `FAIL`
- prod dry-run: exit `2`, hard fail on `P-WF-GATE`, no orders
- shadow dry-run: exit `0`, PatchTST shadow checkpoint contract OK
- full tests: `11967 passed, 7955 skipped, 1 xfailed, 178 warnings`

## PatchTST Status

The older Claude Code 3-way PatchTST experiment was not killed. At this audit point it was still running cross-stock attention `cut5_unwind`, progressing from seed 42 to seed 43.

Existing completed comparison before the still-running tail:

- best min-regime IC mean: cross-stock PatchTST `+0.0506`
- FiLM `+0.0477`
- baseline `+0.0467`
- cross-stock advantage over runner-up: `+0.0029`, below the Tier-2 threshold `0.0050`

Current interpretation: cross-stock PatchTST is interesting as shadow research, but not yet a promotion candidate.

## Current Operating Recommendation

Do not enable production buys for this artifact. Keep production fail-closed. Continue shadow only.

The next valid promotion path must provide:

- recipe-matched 172-feature WF evidence
- positive mean Sharpe, at least 2/3 positive cuts
- explicit SPY comparison
- clean trade contract and clean regenerated tax ledger
- full preflight pass with `P-WF-GATE`
