# renquant_104 Decision Tree Repair Plan — 2026-05-21

## Verdict

The current renquant_104 decision tree is not promotion-safe. The latest
forensics do **not** support the claim that every model has zero signal. They
support a narrower and more useful diagnosis:

- Candidate scores show weak positive 10-day cross-sectional edge.
- The edge is horizon-mismatched with execution: 1-day and 5-day selected
  outcomes are weak or negative.
- The trading tree converts a small gross edge into poor after-tax performance
  through short holding periods, stop-loss asymmetry, and unclear QP/Kelly
  expected-return semantics.
- Decision-trace observability was internally inconsistent: selected rows could
  still carry `blocked_by=kelly_zero:mu_none`.

This file is the repair contract for Claude/Codex agents. Do not promote a new
artifact or live config until these P0 gates are green.

## CLAUDE.md Constraints Applied

- Regime-first evaluation: report BULL_CALM separately from pooled metrics.
- Task/Job/Pipeline discipline: new decision logic must enter through existing
  pipeline tasks or persistence helpers, not notebook-only loops.
- Test-first bug fixes: every code fix needs a regression guard that would fail
  before the patch.
- No single-number promotion: performance claims need regime stratification and
  repeated-window evidence.
- Literature/canonical references: non-trivial portfolio, sizing, tax, and
  evaluation changes cite a concrete source and state the alignment/divergence.

## Evidence Snapshot

### WF Manifest Trade Ledger

Source:

- `/tmp/renquant_strict_wf_20260521/traces/*.round_trips.csv`
- Config reported by trace: `strategy_config.sim_wl200_172_sentiment.json`
- Important caveat: this evaluates the configured walk-forward manifest chain,
  not a single static candidate artifact.

Closed round trips:

- `164` closed trades
- Gross PnL: `+$16,351.08`
- Tax: `-$13,916.32`
- Net after tax: `+$2,434.76`
- Gross win rate: `64.63%`
- Net win rate: `60.98%`

Exit attribution:

- `stop_loss`: `12` trades, net `-$5,482.75`
- `max_hold`: `35` trades, net `+$3,962.88`
- `panel_conviction`: `91` trades, net `+$2,307.02`
- `qp_sell`: `25` trades, net `+$1,587.59`

Interpretation: the strategy has some gross edge, but the realized objective is
taxable after-cost PnL. The stop stack and short-term gain realization are
economically material, not accounting noise.

### Decision-Trace Candidate Layer

After backfilling `data/sim_runs.db::ticker_forward_returns`:

- Candidate rows joined to forward returns: `43,242`
- Date range: `2024-01-02` to `2026-03-02`
- Tickers: `82`

Rank-score evidence:

| Horizon | Selected Mean Return | Selected Excess vs SPY | Selected Hit Rate | BULL_CALM Spearman |
|---|---:|---:|---:|---:|
| 1d | `-0.167%` | `-0.231%` | `48.38%` | `-0.0038` |
| 5d | `+0.052%` | `-0.250%` | `54.70%` | `+0.0074` |
| 10d | `+1.439%` | `+0.843%` | `59.72%` | `+0.0271` |
| 20d | `+2.272%` | `+1.139%` | `58.43%` | `+0.0005` |

Interpretation: the model signal is weak, but it appears more consistent with a
10-day holding thesis than an immediate 1-day/5-day exit thesis. A decision tree
that frequently realizes short-term gains/losses is therefore horizon-mismatched.

## Fixes

### P0-1 — Decision Trace Must Not Mark Selected Rows As Blocked

Status: implemented in this branch.

Bug:

- `candidate_scores.selected=1` rows could also have
  `blocked_by=kelly_zero:mu_none`.
- `ticker_daily_state.selected=1` had the same possible contradiction.
- `scripts/analyze_decision_factors.py` grouped old selected rows by stale
  `blocked_by`, causing the audit to say selected trades were blocked.

Invariant:

- `selected = 1 => blocked_by IS NULL` in persisted decision tables.
- Analysis scripts must treat selected rows as selected even when old DB rows
  contain stale blocker labels.

Code:

- `backtesting/renquant_104/kernel/persistence.py`
- `scripts/analyze_decision_factors.py`

Tests:

- `tests/test_persistence.py::TestCandidateScores::test_selected_candidate_clears_stale_block_reason`
- `tests/test_persistence.py::TestCandidateScores::test_ticker_daily_state_selected_clears_stale_block_reason`
- `tests/test_analyze_decision_factors.py::TestBlockReasonAttribution::test_selected_rows_override_stale_blocked_by`
- `tests/test_repair_decision_trace_invariants.py`

Historical DB repair:

- Tool: `scripts/repair_decision_trace_invariants.py`
- Dry-run on `data/sim_runs.db` found `1234` stale `candidate_scores`
  violations and `0` `ticker_daily_state` violations.
- Applied repair cleared `1234` stale labels without deleting rows.
- Post-check: `remaining = 0`.
- Dry-run on `data/runs.paper.db` after the paper daily acceptance run found
  `24` stale `candidate_scores` violations and `12` stale
  `ticker_daily_state` violations, all from legacy `2026-05-07` to
  `2026-05-11` runs. The fresh `2026-05-21-live-b8542b38` run introduced no
  new violation.
- Applied repair cleared those `36` stale paper labels without deleting rows.
  Post-check: `remaining = 0`.

Reference support:

- This is an evidence/provenance invariant, not a portfolio theorem. It is
  required to make later statistical tests meaningful. It aligns with Bailey
  and López de Prado's backtest-overfitting warning that performance evidence
  must not mix selected outcomes with post-hoc selection/diagnostic labels.
  Reference: Bailey and López de Prado, *The Deflated Sharpe Ratio: Correcting
  for Selection Bias, Backtest Overfitting and Non-Normality*.
  <https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf>

### P0-2 — QP/Kelly μ Contract Must Be Executable Or Disabled

Status: implemented and guarded.

Bug-class:

- Current traces show `kelly_zero:mu_none` across candidate rows while QP/Kelly
  is enabled.
- If QP uses rank/probability as a mean vector, it violates unit consistency.
- If QP does not use μ but persistence says Kelly blocked every candidate, the
  evidence chain is broken.

Invariant:

- Any QP/Kelly path must have one and only one μ source in expected-return units:
  calibrator expected return, NGBoost μ, or explicit alpha-to-return transform.
- If μ is absent and no approved fallback is configured, the optimizer path
  fails strict acceptance rather than silently falling back.
- The trace row must persist the actual μ source and whether the row was sized
  by Kelly, QP, or non-optimizer fallback.

Reference support:

- Markowitz mean-variance optimization requires a coherent expected-return
  vector and covariance matrix. Reference: Markowitz, *Portfolio Selection*,
  Journal of Finance, 1952. DOI reference:
  <http://dx.doi.org/10.1111/j.1540-6261.1952.tb01525.x>
- Kelly sizing uses expected edge over variance; `f* = μ / σ²` is only valid
  when μ and σ share compatible return units. Reference: Kelly, *A New
  Interpretation of Information Rate*, Bell System Technical Journal, 1956.
  <https://www.princeton.edu/~wbialek/rome/refs/kelly_56.pdf>
- Shrinkage covariance is preferred over raw sample covariance for portfolio
  optimization because optimizers amplify estimation error. Reference:
  Ledoit and Wolf, *Honey, I Shrunk the Sample Covariance Matrix*, Journal of
  Portfolio Management, 2004. <https://ledoit.net/honey.pdf>

Design:

- Add a tiny validation task/helper before QP order emission that checks μ
  availability and unit source under strict configs.
- Add a persisted `mu_source` or run-bundle equivalent before broad rollout.
- If μ is missing in strict acceptance, QP/Kelly must emit zero buys and a hard
  counter, not a mixed strategy.

Acceptance:

- Unit test: strict QP/Kelly config with μ missing fails before order emission.
- Integration test: SimAdapter path records selected rows with μ source and no
  contradictory blocker.
- WF acceptance: no `selected=1 AND blocked_by IS NOT NULL` rows.

Implemented guardrails:

- `JointPortfolioQPJob` runs `ValidateQPMuContractTask` after μ-source forcing
  and Grinold-Kahn transform, before weight construction and solver emission.
- `strategy_config.json`, `strategy_config.golden.json`, and
  `strategy_config.sim_wl200_172_sentiment.json` set
  `rotation.joint_actions.qp_mu_contract = "strict"`.
- `ranking.kelly_sizing.use_calibrator_mu = true` wires the calibrated
  expected-return head into Kelly μ when NGBoost μ is unavailable.
- New production-chain acceptance:
  `tests/test_qp_integration.py::TestQPMuContractIntegration::test_strict_mu_contract_stops_before_solver_on_raw_score_fallback`.

Focused verification:

```text
36 passed in 3.00s
```

Paper daily acceptance, `2026-05-21-live-b8542b38`:

- Broker: `paper` only.
- Regime: `BULL_CALM`, confidence `0.59`, Hurst `0.71`.
- Candidate flow: `114` loaded model symbols, `93` raw candidates, `79`
  panel-scored candidates after earnings/wash-sale/realized-vol gates.
- Kelly/QP contract: `ApplyKellySizingTask` sized `59/79` candidates with
  non-zero Kelly target; the only zero reason was `mu_le_min_edge=20`.
  No `kelly_zero:mu_none` appeared in the fresh run.
- QP result: `0` buys and `0` sells because all proposed deltas were either
  inside the no-trade band or below `min_delta_weight`:
  `qp_delta_below_min_dw(73)`.
- Invariant check after run:
  `selected=1 AND blocked_by IS NOT NULL` returned `0` for the fresh run.

Interpretation: buy is enabled and the tree is no longer globally blocking
BULL_CALM buys; the current no-trade decision is coming from optimizer/no-trade
band economics, not from a stale `disable buy` or missing-μ bug.

### P0-3 — Align Exit Horizon With Observed Signal Horizon

Status: implemented for model-driven soft panel exits.

Observed issue:

- Selected candidates are negative vs SPY at 1d and 5d but positive at 10d and
  20d.
- Stop/panel exits realize many trades before the signal horizon has time to
  work.

Invariant:

- A signal trained/evaluated for 10-20 day forward edge cannot be promoted with
  an exit stack whose dominant realized holding period is materially shorter,
  unless a separate short-horizon alpha proves that early exits add value.

Reference support:

- Gu, Kelly, and Xiu evaluate ML signals as expected return forecasts in an
  empirical asset-pricing setting; model output must be tied to the forecast
  horizon used in portfolio construction. Reference: Gu, Kelly, Xiu,
  *Empirical Asset Pricing via Machine Learning*, Review of Financial Studies,
  2020. <https://academic.oup.com/rfs/article/33/5/2223/5758276>
- Almgren and Chriss frame execution as an explicit tradeoff between risk and
  trading costs; unnecessary turnover must be justified against expected alpha.
  Reference: Almgren and Chriss, *Optimal Execution of Portfolio Transactions*,
  2000. <https://docslib.org/doc/1384720/optimal-execution-of-portfolio-transactions>

Design:

- Do not globally loosen stops.
- `panel_conviction` is a model-driven soft exit. In `BULL_CALM`, it now waits
  `10` calendar days before firing, matching the first horizon where selected
  candidate evidence turned positive.
- Regime deterioration can still exit early because the configured minimum is
  keyed only to `BULL_CALM`; `CHOPPY`, `BULL_VOLATILE`, and `BEAR` are not
  delayed by this thesis-age gate.
- Hard path exits remain exempt: stop-loss, trailing stop, single-day loss,
  max-hold, rotation, Kelly trim, and joint/QP exits do not call this guard.
- The rule is shared by both legacy `PanelConvictionExitTask` and
  pipeline-level `CrossSectionalPanelExitTask` via
  `kernel/pipeline/soft_exit_guards.py`.

Acceptance:

- Report per-regime APY, Sharpe, max drawdown, tax, and trade count.
- Must beat SPY-relative excess return in BULL_CALM, not just pooled APY.
- Unit coverage:
  - `tests/test_panel_conviction_xs_exit.py::TestHorizonAndTaxGates`
  - `tests/test_panel_conviction_exit.py::TestSoftExitGuards`

### P0-4 — Tax-Aware Objective Must Match Taxable Account Reality

Status: implemented for model-driven soft panel exits.

Observed issue:

- Gross closed PnL is positive, but after-tax realized PnL is mostly consumed by
  short-term taxes.

Invariant:

- For a taxable-account strategy, promotion uses after-tax objective first.
- Any exit that realizes a short-term gain must beat a tax-adjusted hold
  alternative.

Reference support:

- Tax-aware portfolio optimization explicitly models long/short-term capital
  gains and wash-sale constraints. Reference: Moehle et al., *Tax-Aware
  Portfolio Construction via Convex Optimization*, 2021.
  <https://web.stanford.edu/~boyd/papers/tax_aware_portfolio.html>
- Leland's taxable portfolio work shows that capital gains taxes can justify
  allowing positions to drift farther from target before selling. Reference:
  Leland, *Optimal Portfolio Management with Transactions Costs and Capital
  Gains Taxes*. <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=206871>

Design:

- Added tax-adjusted soft-exit threshold:
  `expected_loss_avoided >= tax_drag + transaction_cost + min_exit_edge`.
- `expected_loss_avoided = max(0, -mu)` and `tax_drag` follows the existing
  project convention used by `kernel.rotation.tax_drag`: unrealized gain × tax
  rate, in fraction-of-position units.
- Default production config enables this only for short-term unrealized gains
  and only for model-driven `panel_conviction` exits.
- Hard path exits remain exempt. If a position hits stop-loss, trailing stop,
  single-day loss, or max-hold, tax does not block the exit.
- Bug fixed while implementing this: cross-sectional and legacy panel exits
  read `lt_hold_gate_days` from `risk`, but production stores the single source
  at the config root (`lt_hold_gate_days = 330`). The prior fallback acted like
  a 30-day LT gate in tests/stubs. The shared helper now prefers root-level
  config and falls back to risk-level legacy fields only when needed.

Acceptance:

- Realized tax as percentage of gross PnL must decline without increasing
  max drawdown beyond the regime-specific budget.
- Report gross PnL, tax, net PnL by exit reason.
- Unit coverage:
  - marginal short-term gain with weak negative μ is suppressed
  - large negative μ can still pay tax drag and exit
  - unrealized loss has no tax-drag suppression
  - root-level `lt_hold_gate_days=330` no longer behaves like a 30-day default

### P0-5 — PatchTST/CSA/FiLM Artifact Isolation

Status: not started; background jobs still running.

Observed issue:

- `hf_cross_stock_5cut_5seed_pt07` has overlapping driver processes writing the
  same artifact directory.
- CSA/FiLM have no final `aggregate.csv` / `raw_results.json` yet.

Invariant:

- Every experiment run writes to a unique immutable output directory stamped by
  start time, git SHA, config hash, and run id.
- A comparison script reads only complete runs with `25/25` summaries and a
  manifest that names every expected cut/seed.

Reference support:

- Backtest-overfitting and multiple-testing controls require knowing exactly
  how many trials were run and which results are complete. Reference: Bailey
  et al., *Probability of Backtest Overfitting* / Deflated Sharpe Ratio work.
  <https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf>

Acceptance:

- No aggregate file can be produced from a partial or concurrently written run.
- Comparison refuses duplicate output dirs.

## Immediate Execution Order

```
[now]        [+tests]       [+sim evidence]        [+WF]
P0-1 done -> targeted pass -> backfill/re-analyze -> doc + commit
P0-2 code -> unit/integration tests -> strict sim smoke -> WF only if green
P0-3/P0-4 A/B design -> regime-stratified sims -> promote only Tier 3
PatchTST monitor -> isolate if incomplete/dirty -> compare only complete runs
```

## Current Implemented Patch

P0-1 through P0-2 have been implemented and focused tests passed:

```text
39 passed in 2.48s
```

Focused command:

```text
.venv/bin/python -m pytest \
  tests/test_persistence.py::TestCandidateScores \
  tests/test_analyze_decision_factors.py \
  tests/test_repair_decision_trace_invariants.py \
  tests/test_qp_force_mu_source.py \
  tests/test_qp_grinold_kahn_transform.py \
  tests/test_qp_integration.py \
  tests/test_kelly_sizing.py -q
```

Full-suite status after this patch:

```text
9 failed, 11925 passed, 7955 skipped, 1 xfailed in 242.35s
```

The full-suite failures are not in the P0-1 touched files. They include existing
source-contract drift in monthly calibrator / panel scoring / HF wrapper tests,
one `python` executable PATH assumption, and one sandboxed process-pool semaphore
failure. Treat the repo as not globally green until those are repaired.

Acceptance notes after the patch:

- `scripts/smoke_test_model.py --strategy renquant_104` passed on the prod
  panel artifact.
- A static historical prod sim correctly refused to run because the artifact
  training date would create look-ahead leakage for a February/March 2026
  replay. Treat this as a good fail; historical evaluation must use
  walk-forward manifests.
- `strategy_config.sim_wl200_172_sentiment.json` currently fails static-path
  preflight as a no-op/stale-path config and was not forced through with
  `--skip-preflight`.
- The paper daily run completed end-to-end, but the per-ticker candidate phase
  took about `454s` and emitted pandas fragmentation warnings in
  `kernel/panel_pipeline/feature_matrix.py`. This is a performance/reliability
  item for P1, not a reason to promote.

This is not the end of the repair. It removes a false attribution bug, enforces
the strict μ contract before optimizer emission, and gives us a cleaner base for
P0-3/P0-4 horizon and tax A/B work.
