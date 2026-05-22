# RenQuant 104 Signal/Decision Rebuild Plan

Date: 2026-05-22

## Executive Position

RenQuant 104 must be rebuilt around a two-layer contract:

1. **Raw signal contract**: before any QP sizing, top-up, stop, rotation, or tax
   logic, a panel scorer must beat simple cross-sectional top-K controls by
   regime.
2. **Execution contract**: only after raw signal passes do we optimize sizing,
   turnover, exits, and taxes. A losing raw signal is not fixed by a more complex
   decision tree.

This follows `CLAUDE.md`: regime-first evaluation, TDD, no new number without
A/A + shuffled/control + time-shift placebo, and mature references before new
machinery.

## Sources Read And How They Map

- **Microsoft Qlib TopkDropoutStrategy**:
  https://github.com/microsoft/qlib/blob/main/docs/component/strategy.rst
  Qlib's documented strategy ranks instruments by prediction score, holds TopK,
  and replaces low-ranked held names with high-ranked unheld names. RenQuant's
  new `scripts/eval_raw_signal_baseline.py` uses the same core idea as a
  scorer-only baseline, but with fixed-hold event returns so the signal can be
  audited without execution churn.

- **Qlib Alpha158 handler**:
  https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/handler.py
  The existing 104 feature set is already Qlib-style Alpha158 plus fundamentals
  and sentiment. The rebuild keeps the current feature artifact contract and
  tests signal quality before changing feature engineering.

- **PatchTST, Nie et al. 2023**:
  https://arxiv.org/abs/2211.14730
  PatchTST is a valid mature sequence-model family because it patches time
  series into tokens and uses channel independence to reduce attention cost.
  In RenQuant it remains shadow-only unless it beats the same raw-signal
  top-K controls per regime.

- **cvxportfolio / Boyd-style transaction-cost optimization**:
  https://www.cvxportfolio.com/en/stable/index.html and
  https://www.cvxportfolio.com/en/1.3.1/costs.html
  cvxportfolio separates forecasts, risks, transaction costs, and constraints.
  RenQuant should converge toward this separation: alpha first, then cost-aware
  optimizer. The current QP/top-up layer must not be allowed to invent alpha.

- **Lobo, Fazel, Boyd 2007 transaction-cost portfolio optimization**:
  https://web.stanford.edu/~boyd/papers/portfolio.html
  Linear transaction costs and risk constraints are convex and quickly solvable;
  fixed costs require relaxation or heuristics. This supports replacing ad hoc
  top-ups with a no-trade band / turnover-hurdle optimizer after alpha passes.

- **Ledoit-Wolf 2004 covariance shrinkage**:
  https://bse.eu/research/working-papers/honey-i-shrunk-sample-covariance-matrix
  Sample covariance is unstable for many stocks and few observations; shrinkage
  reduces estimation error. Any future optimizer covariance matrix should use a
  shrinkage estimator, not a raw sample covariance.

- **Bailey and Lopez de Prado 2014 Deflated Sharpe Ratio**:
  https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf
  Multiple trials and non-normal returns inflate Sharpe. RenQuant should keep
  DSR/PBO for model-promotion reports and treat backtest wins without controls
  as suspect.

## Old Versus New Design

| Layer | Old behavior | New contract |
|---|---|---|
| Scorer validation | IC and WF often mixed with full decision tree | Raw top-K scorer-only baseline first |
| Regime handling | Pooled Sharpe was repeatedly over-discussed | Per-regime tables first, pooled second |
| Controls | Some experiments had controls, not universal | A/A, shuffle, reverse, time-shift in evaluator |
| Optimizer | QP/top-up could trade weak compressed scores | Optimizer only after alpha passes controls |
| Tax/cost | Tax surfaced late in trade ledger | Cost-aware execution is a separate acceptance layer |
| PatchTST | Shadow experiments hard to compare to XGB | Same raw top-K baseline for XGB and PatchTST |

## Implementation Step 1: Raw Signal Baseline

Added:

- `scripts/eval_raw_signal_baseline.py`
- `tests/test_raw_signal_baseline.py`

The evaluator:

1. Loads an XGB panel-LTR JSON artifact or HF PatchTST checkpoint.
2. Scores cross-sections on rebalance dates.
3. Uses local OHLCV closes, not labels, to compute actual close-to-close event
   returns.
4. Buys equal-weight top-K, tracks bottom-K, SPY, and universe mean.
5. Reports APY, Sharpe, after-tax stress APY, alpha vs SPY, long-short spread,
   and win rate by regime first.
6. Runs controls:
   - A/A deterministic repeat
   - per-date score shuffle
   - reverse score
   - per-ticker stale-score time shift

Invariant: a model is not trustworthy unless actual top-K beats the controls in
the regimes where it is expected to trade.

## Initial Baseline Results

Reports:

- `doc/research/2026-05-22-raw-signal-baseline-xgb.md`
- `doc/research/2026-05-22-raw-signal-baseline-xgb-reb20.md`
- `doc/research/2026-05-22-raw-signal-baseline-patchtst-seed44.md`

### XGB prod, 60-day non-overlapping event study

Window: 2024-01-02 to 2026-03-28, top 10, bottom 10, hold 60 trading days,
rebalance every 60 trading days.

| Metric | Actual | Shuffle | Reverse | 20-day time-shift |
|---|---:|---:|---:|---:|
| Events | 9 | 9 | 9 | 8 |
| Pooled APY | +32.94% | +16.10% | +7.76% | +49.73% |
| Pooled Sharpe | +1.517 | +1.144 | +0.610 | +3.106 |
| Alpha vs SPY | +3.11% | -0.52% | -2.35% | +6.73% |
| Long-short spread | +5.46% | +1.69% | -5.46% | +8.68% |
| Pooled mean IC | +0.059 | n/a | n/a | n/a |

Interpretation: XGB has positive cross-sectional structure versus shuffle and
reverse, but it does **not** beat the 20-day stale-score control. That means the
current scorer is closer to a persistent quality/ranking exposure than a clean
short-horizon timing signal. This explains why a high-churn QP/top-up/exit tree
can destroy the edge.

### XGB prod, 20-day rebalance / 60-day hold diagnostic

This is an overlapping event study, so APY and Sharpe are diagnostic only and
likely overstated versus a self-financing portfolio. It increases sample count
from 9 to 27 events and keeps the same qualitative conclusion:

- actual pooled APY +152.43%, Sharpe +2.808, alpha vs SPY +4.21%
- shuffle pooled APY +49.40%, alpha vs SPY -0.22%
- reverse pooled APY +6.09%, alpha vs SPY -3.21%
- time-shift pooled APY +171.45%, alpha vs SPY +5.26%

Interpretation: more events confirm score directionality versus shuffle/reverse,
but time-shift remains competitive. Execution should therefore reduce churn and
respect no-trade bands; it should not chase tiny day-to-day score differences.

### PatchTST seed44 shadow, 60-day non-overlapping event study

Window: 2025-02-06 to 2026-02-10, top 10, bottom 10, hold 60 trading days,
rebalance every 60 trading days.

| Metric | Actual | Shuffle | Reverse | 20-day time-shift |
|---|---:|---:|---:|---:|
| Events | 5 | 5 | 5 | 4 |
| Pooled APY | +11.37% | +29.57% | +20.00% | +70.48% |
| Pooled Sharpe | +0.553 | +1.689 | +2.527 | +2.526 |
| Alpha vs SPY | -0.29% | +3.24% | +1.12% | +7.99% |
| Long-short spread | -1.41% | +1.46% | +1.41% | +7.81% |
| Pooled mean IC | -0.016 | n/a | n/a | n/a |

Interpretation: PatchTST seed44 fails the raw-signal contract. It can stay
shadow/research, but it is not a promotion candidate and should not be used as
primary unless a future 5-cut x 5-seed run beats these controls per regime.

## Next Refactor Steps

### P0: Decision Attribution Guard

Add a production trace assertion that every emitted order has:

- source job/task
- score before and after calibration
- regime label
- top-up or fresh-buy flag
- expected return / risk / cost inputs
- reason it passed the final hurdle

Invariant: no order exists without an accountable pipeline owner and score
state.

### P1: Config Parity Guard

The last WF bug showed `strategy_config.sim_wl200.json` could drift from the
current 172-feature prod artifact and decision semantics. Add a guard that
compares prod and WF configs for:

- panel feature count / artifact fingerprint
- scorer kind
- buy floor mode
- top-up floor
- QP enabled flag
- tax mode
- regime parameter map

Invariant: acceptance/WF cannot run on a stale or easier config unless the
filename explicitly marks it as exploratory.

### P2: Cost-Aware No-Trade Band

Replace unconditional top-ups with a no-trade band:

- trade only when expected alpha exceeds transaction cost, tax drag, and a
  turnover hurdle
- use regime-specific hurdle values
- use Ledoit-Wolf shrinkage covariance for risk

This follows the cvxportfolio/Boyd separation of forecasts, costs, risks, and
constraints.

### P3: Exit Tree Ablation

Run hard-only, soft-only, no-top-up, no-panel-conviction-exit, and max-hold
variants by regime. Current trade forensics suggest max-hold exits are the best
class while early churn is tax-negative.

Invariant: an exit rule must improve after-tax alpha versus holding to the raw
signal horizon in its own regime.

### P4: PatchTST Shadow Gate

PatchTST only enters shadow-primary candidate status if it beats XGB or adds
ensemble lift under the same raw-signal evaluator, with:

- per-regime IC
- top-K APY and alpha vs SPY
- shuffle/reverse/time-shift controls
- 5-cut x 5-seed stability

Invariant: architecture novelty is irrelevant unless it survives the same
baseline contract.

## First Commands

XGB raw-signal baseline:

```bash
.venv/bin/python scripts/eval_raw_signal_baseline.py \
  --artifact backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.json \
  --kind xgb \
  --start 2024-01-02 \
  --end 2026-03-28 \
  --top-k 10 \
  --bottom-k 10 \
  --hold-days 60 \
  --rebalance-days 60
```

PatchTST shadow raw-signal baseline:

```bash
.venv/bin/python scripts/eval_raw_signal_baseline.py \
  --artifact artifacts/patchtst_shadow/canonical_5seed_mps/seed_44/hf_patchtst_all_seed44_model.pt \
  --kind hf_patchtst \
  --start 2025-02-06 \
  --end 2026-02-10 \
  --top-k 10 \
  --bottom-k 10 \
  --hold-days 60 \
  --rebalance-days 60
```

## Acceptance Criteria

- Tests pass for the new evaluator.
- XGB and PatchTST can be compared by the same raw-signal contract.
- Reports are regime-first and include controls.
- If raw top-K is weak, do not tune QP or exits as a substitute for alpha.
- If raw top-K is strong but WF is weak, the next implementation target is
  optimizer/top-up/exit attribution, not model architecture.
