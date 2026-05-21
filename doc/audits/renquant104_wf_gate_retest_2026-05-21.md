# renquant_104 WF Gate Retest After Contract Fixes - 2026-05-21

## Verdict

Do not promote any strict/candidate production artifact from the 2026-05-21 run.

The gate infrastructure is now stricter and more auditable, but the corrected
walk-forward result is still weak. The evaluated path here is the XGB
walk-forward manifest chain from `strategy_config.sim_wl200.json`, not a static
candidate artifact.

No live or daily real-money run was executed.

## What Was Fixed Before Retest

1. `run_wf_gate.py` now verifies whether the sim config actually evaluates the
   candidate artifact passed by `--artifact`.
2. `promote()` rejects `wf_gate_metadata` when `candidate_artifact_used=false`.
3. WF metadata now records SPY benchmark context, strategy-minus-SPY metrics,
   and regime counts per cut.
4. The sim calibrator was refit from the same pre-2024 scorer/window using
   Platt calibration:
   - `prob_unique=100`
   - `prob_iqr=0.1133`
   - `expected_return maxabs=0.0595`
5. The calibrator saturation guard was narrowed:
   - low daily probability IQR is diagnostic only
   - new-buy abstain now requires true score collapse or upper-tail saturation

Regression tests:

```bash
.venv/bin/python -m pytest \
  tests/test_calibrator_saturation_guards.py \
  tests/test_wf_gate_cli_contract.py \
  tests/test_promote_wf_gate.py -q
```

Result: `42 passed`.

## Corrected 3-Cut WF Result

Command pattern:

```bash
.venv/bin/python scripts/run_sim_104.py \
  --strategy-config-name strategy_config.sim_wl200.json \
  --start <cut-start> --end <cut-end> \
  --no-compare --no-persist --skip-preflight \
  --equity-json /tmp/renquant104_corrected_wf2/<cut>_equity.json
```

| Cut | Window | Strategy Sharpe | Strategy APY | SPY Sharpe | SPY APY | Delta Sharpe | Dominant HMM Regime |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 2024-01-02 to 2024-12-31 | `-0.38` | `-0.5%` | `+1.78` | `+24.1%` | `-2.16` | `BULL_VOLATILE` |
| 2 | 2024-07-01 to 2025-06-30 | `-0.36` | `-1.0%` | `+0.72` | `+13.5%` | `-1.08` | `BULL_VOLATILE` |
| 3 | 2025-04-01 to 2026-03-28 | `-0.03` | `+3.9%` | `+0.75` | `+13.3%` | `-0.78` | `BULL_VOLATILE` |

Aggregate:

- mean strategy Sharpe: `-0.257`
- mean SPY Sharpe: `+1.081`
- mean strategy minus SPY Sharpe: `-1.338`
- cuts beating SPY Sharpe: 0 / 3
- positive APY cuts: 1 / 3
- positive Sharpe cuts: 0 / 3

## Regime Context

HMM regime counts:

- Cut 1: `BULL_VOLATILE=223`, `CHOPPY=14`, `BULL_CALM=8`, `BEAR=7`
- Cut 2: `BULL_VOLATILE=198`, `BEAR=39`, `CHOPPY=13`
- Cut 3: `BULL_VOLATILE=207`, `BEAR=27`, `CHOPPY=15`

Interpretation: these are mostly bull/volatile windows where SPY performed
well. A long-biased equity strategy should not be judged only against zero; it
also needs to clear a benchmark-relative hurdle. This retest does not.

## Decision-Tree / Execution Diagnostics

| Cut | QP_BUY log events | Abstain-new-buys | Low-IQR diagnostic | NoTrade alerts | Buy-blocked days/events |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 307 | 0 | 252 | 0 | 38 |
| 2 | 298 | 0 | 246 | 13 | 104 |
| 3 | 298 | 12 | 237 | 11 | 65 |

Notes:

- The false global buy disable from low IQR is fixed: cuts 1 and 2 have zero
  calibrator abstains, and all three cuts emit QP buys.
- Cut 3 still has 12 true `score_collapse` abstains. That is acceptable as a
  safety action, but it should be inspected if it clusters in important dates.
- Low-IQR diagnostics remain frequent. Platt compression is not automatically
  fatal, but it means calibrated probabilities have narrow daily dispersion.
  Ranking is mostly carried by monotone ordering, not wide probability spread.
- Cut 2 has many legitimate buy-blocked events from drawdown/EMA/regime gates.
  That explains part of the no-trade behavior, but not the benchmark lag.
- Logs still show recurring insufficient-cash messages on some QP buy outputs.
  That is an execution/sizing issue worth separating from alpha quality.

## Trust Assessment

Training process: improved, but not yet promotion-trustworthy.

Model/system: not promotion-trustworthy under this corrected WF evidence.

Reason:

- The previous gate could stamp a candidate artifact even when a walk-forward
  manifest, not the candidate, was being evaluated.
- That bug is fixed, but the corrected manifest-chain WF underperforms SPY in
  every cut and has negative mean Sharpe.
- Calibrator output is now bounded and non-collapsed at the artifact level, but
  daily calibrated probability spread remains narrow.

## Recommended Next Fixes

1. Add benchmark-relative hard criteria to WF gate:
   - require `strategy_minus_spy_sharpe_mean > 0` or an explicitly documented
     low-beta objective with positive alpha/IR
   - report beta, alpha, and information ratio per cut
2. Split WF gates by evaluation scope:
   - static candidate artifact gate
   - manifest/retraining-recipe gate
   - never let one stamp the other
3. Diagnose cut-level underperformance:
   - QP cash insufficiency and target-weight top-ups
   - drawdown/EMA buy-block distribution
   - per-regime return attribution
   - score-collapse dates in cut 3
4. Treat the current XGB walk-forward manifest as a research baseline, not a
   production promotion proof.

## Commits

- `9a4aa33` - bind WF gate to the actually evaluated artifact
- `ecaca18` - avoid false calibrator abstain on Platt compression

