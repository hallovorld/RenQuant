# RenQuant 104 Decision Tree And Sim Audit — 2026-05-23

## Scope

This note records the clean XGB true-OOS sim rerun after the 2026-05-23
contract and sim hot-path fixes. It is intentionally blunt: the current
decision tree is cleaner, but not yet scientifically sufficient as a
profitable trading system.

## Component I/O Contract Audit — 2026-05-23 Addendum

I audited the production decision path as code, not as documentation:
`SimAdapter/RunnerAdapter/LeanAdapter -> InferenceContext ->
InferencePipeline -> PanelScoringJob -> JointPortfolioQPJob -> Execution/sim
ledger`. The expected contracts below are the contracts that matter for
APY/Sharpe, because any unit mistake here can make a good raw signal lose
money.

### Pipeline Contract Map

| Component | Expected input | Expected output | Numeric contract | Current audit verdict |
|---|---|---|---|---|
| Adapter -> `InferenceContext` | `config`, `today`, OHLCV frames through `today`, `models`, `holdings`, `cash`, `portfolio_value`, `prices`, `corr_matrix`, `spy_returns`, calendar/state maps | Mutable context consumed by jobs | No future bars; all prices and NAV positive; holdings carry entry date/price/shares/lots; sim disables live freshness | Mostly OK in sim path. Historical freshness is disabled deliberately. State/trace path bug existed in WF wrapper and is now fixed. |
| `DataFreshnessGateTask` | Context OHLCV + config freshness knob | Pass or exception | Live only; historical sim should not reject old bars | OK for sim after `run_sim_104` disables freshness by default. |
| `RegimeJob` | SPY/history, GMM/regime state, thresholds | `ctx.regime`, `ctx.confidence`, `ctx.regime_state` | Regime in `{BULL_CALM,BULL_VOLATILE,CHOPPY,BEAR}`; confidence in `[0,1]`; every trade needs a recorded regime thesis | Mechanically OK, but economic quality is not OK: closed entries are dominated by `BULL_CALM`, where score vs realized P&L is negative. |
| `BuyGatesJob` | Regime, drawdown, transition, confidence, SPY velocity/EMA | `ctx.buy_blocked`, `ctx.bear_only` | Buy block must be explainable and must not prevent candidate scoring audit when configured | OK as a gate. Not the main source of low APY; the system still trades. |
| `TickerSellJob` | Holding state, current price, features, model, exit params from current and entry regime | `(ticker, ExitSignal)` in `ctx.exits` | Path exits before model exits; exit reason, params, regime, quantity, score snapshot recorded | Mechanically OK. Forensics show stop-loss/qp-sell losses dominate; current exit logic often realizes BULL_CALM entries after regime deterioration. |
| `TickerCandidateJob` | Candidate ticker OHLCV/features/model/wash/earnings/sector data | `CandidateResult(ticker, raw_score, rank_score, rs_score, ...)` | Candidate score finite; rs_score diagnostic only; no held/pending ticker in buy universe | Mostly OK. Candidate admission is not the primary failure; post-score selection is. |
| `PanelScoringJob` | Candidate/holding set, panel scorer, feature matrix, global calibrator, optional NGBoost | Writes `panel_score`, calibrated `rank_score`, `expected_return`, `mu`, `sigma`, `kelly_target_pct` | `rank_score` probability-like `[0,1]`; `panel_score` raw rank model score; `expected_return/mu` same horizon; `sigma` same horizon or explicitly converted | Contract is only partly scientific. Calibrator expected-return metadata says `lookahead_days=60`, while realized-vol fallback is annualized 60d vol. The code treats them together in Kelly/QP without an explicit horizon conversion. This is a live suspect. |
| `VetoWeakBuysTask` / quality floors | Scored candidates | Filtered candidates | Weak-buy threshold must be regime-conditional and strong enough to matter | Current adaptive cap is toothless for this trace: admitted trade `rank_score` range is `0.4836..0.6762`, while `buy_floor` cap is effectively `0.30`. Gate B is disabled in the sim config. |
| `RankingJob` | Candidates with calibrated scores | `ctx.ranked` sorted candidates | Sorting must preserve the signal actually used for execution | Mechanically OK. The problem is that the executed slice is compressed and not discriminative in BULL_CALM. |
| `JointPortfolioQPJob` | Holdings + candidates, `mu`, `sigma`, covariance/correlation, sector caps, no-trade bands, cash, constraints | `ctx.orders` and extra soft exits | `mu` must be expected-return-like; `sigma` same horizon/unit as `mu`; caps/sector/corr constraints finite; buys must not be forced when edge is weak | Strict μ contract passes, but scientific unit contract is incomplete. `qp_min_invested_pct=0.7` can force deployment even when BULL_CALM score is anti-predictive. QP target weight is mildly anti-correlated with signal in the audited trace. |
| Execution/sim ledger | Orders/exits and price/cash state | Trade log, equity curve, round trips, tax report | Raw events immutable; round-trip FIFO; event-level tax is cash stress; annual-net tax is reporting/gate metric | Fixed two bugs here: annual-net is now first-class in WF metrics; losing lots are no longer mislabeled as `positive_gross`. Event-level tax can still exceed gross by design; it is no longer the main performance metric. |

### Actual Numeric Flow From Latest WF Trace

Trace root audited:
`backtesting/renquant_104/backtesting/renquant_104/artifacts/diagnostics/wf_trade_traces/codex_20260523_annualnet_20260523T170643Z`.
The duplicate path segment is itself a fixed wrapper bug; new `--trace-dir`
resolution will not double-prefix repo-relative paths.

Across the three WF cuts:

| Metric | Value |
|---|---:|
| Closed round trips | 229 |
| Closed gross P&L | +$9,975.26 |
| Event-level tax debited | $20,645.68 |
| Event-level net P&L | -$10,670.42 |
| Closed win rate | 31.88% |
| Average / median hold | 67.2d / 40.0d |

Annual-net reporting changes the interpretation but not the strategy verdict:

| Cut | Event APY | Event Sharpe | Annual-net APY | Annual-net Sharpe | Event tax | Annual-net tax |
|---|---:|---:|---:|---:|---:|---:|
| 2024-01-02 -> 2024-12-31 | -0.14% | +0.033 | +4.74% | +0.578 | $5,240.55 | $387.53 |
| 2024-07-01 -> 2025-06-30 | -0.64% | -0.017 | +4.96% | +0.624 | $6,474.69 | $936.21 |
| 2025-04-01 -> 2026-03-28 | +3.80% | +0.399 | +8.88% | +0.905 | $9,049.17 | $4,056.12 |

WF still fails because it lags SPY and fails regime/monotonicity gates.
The corrected mean annual-net Sharpe is about `+0.702`, but
`strategy_minus_spy_sharpe_mean=-0.379`, `n_cuts_beat_spy_sharpe=1`, and
`n_cuts_beat_spy_apy=0`.

### Weird Numeric Findings

1. BULL_CALM is the main broken area.

| Entry regime | n | Gross P&L | Event tax | Event net | Win rate | Mean P&L pct | Median hold |
|---|---:|---:|---:|---:|---:|---:|---:|
| BULL_CALM | 199 | +$11,125.92 | $19,434.96 | -$8,309.05 | 30.65% | +0.72% | 35d |
| BULL_VOLATILE | 30 | -$1,150.66 | $1,210.72 | -$2,361.38 | 40.00% | +3.07% | 111d |

Per-regime score IC on closed trades:

| Regime | rank_score rho vs P&L | panel_score rho | mu rho | sigma rho | kelly_target rho |
|---|---:|---:|---:|---:|---:|
| BULL_CALM | -0.118 | -0.120 | -0.072 | +0.001 | -0.282 |
| BULL_VOLATILE | +0.518 | +0.440 | +0.350 | +0.355 | n/a |

This is not a small cosmetic problem. In the regime where the system trades
most, the entry score is anti-monotonic. That means the model/decision tree is
not selecting the right names under the BULL_CALM thesis.

2. The QP input bands look numerically valid but economically compressed.

Executed closed-trade inputs:

| Field | min | p10 | median | p90 | max |
|---|---:|---:|---:|---:|---:|
| `entry_rank_score` | 0.4836 | 0.5246 | 0.5374 | 0.6216 | 0.6762 |
| `entry_panel_score` | -0.2895 | -0.1627 | -0.0991 | +0.2291 | +0.4305 |
| `entry_mu` | +0.0018 | +0.0146 | +0.0187 | +0.0452 | +0.0653 |
| `entry_sigma` | 0.1251 | 0.1307 | 0.1455 | 0.2134 | 0.3711 |
| `entry_kelly_target_pct` | 0.0750 | 0.0750 | 0.0750 | 0.0997 | 0.1132 |

The values are finite and within their intended ranges. The weird part is
selection: many Kelly targets sit at the lower cap (`0.075`), so sizing does
not carry much cross-sectional information. In the audited buys, QP target
weight is mildly anti-correlated with `rank_score`, `mu`, and `sigma`. That is
a portfolio-construction smell even if the solver is numerically optimal.

3. The weak-buy veto is not actually weak-buy protection in this trace.

The configured `buy_floor="adaptive_mean_std_cap"` with cap `0.30` allows all
executed trade scores because executed `rank_score` never comes close to that
floor. Gate B (`quality_floor.edge_sharpe_floor`) is disabled even though
earlier full-OOS ablation suggested a BULL_CALM risk-adjusted admission floor
improves APY/Sharpe. This is not promotable yet, but the current live-like
path lacks an active pre-QP edge floor.

4. Tax accounting is now separated correctly, but it revealed the real issue.

Event-level tax is a cash-stress path. Annual netting is a reporting/gate
metric. The IRS classifies gains/losses as short-term or long-term and nets
capital gains/losses in that framework; losses beyond gains have deduction and
carryforward rules. Reference: IRS Topic 409
<https://www.irs.gov/taxtopics/tc409>. The simulator now reports both views:
event-level for conservative cash stress, annual-net for economic performance.

After the fix, annual-net Sharpe improves materially, but the strategy still
loses to SPY. Therefore "tax was the bug" is false. Tax was one bug in the
reporting/gating layer; the core BULL_CALM selection problem remains.

5. QP formulation is structurally reasonable but unit completeness is suspect.

Using an optimization policy with return forecast, risk model, costs, and
constraints follows the cvxportfolio/convex portfolio-control pattern:
<https://www.cvxportfolio.com/en/stable/optimization_policies.html>. Cost and
constraint support are also mature concepts in cvxportfolio:
<https://www.cvxportfolio.com/en/1.1.1/costs.html>. The code follows that
shape, but the project-specific μ/σ inputs are not yet scientifically stamped
enough:

- Calibrator expected return metadata says `lookahead_days=60`.
- Realized-vol fallback uses a 60-day rolling volatility and is annualized.
- Kelly/QP consumes `mu / sigma^2` style inputs without a clear horizon
  conversion in the live trace.

If μ is a 60-trading-day return and σ is annualized volatility, either σ must
be converted to the 60-day horizon or μ must be annualized before optimizer
use. The current code has a "strict μ source" contract, but not a complete
"same horizon μ/σ" contract. This needs a regression test before changing
behavior.

### Bugs Fixed In This Addendum

1. WF trace path resolution.
   `scripts/run_wf_gate.py --trace-dir backtesting/renquant_104/...` used to
   write under
   `backtesting/renquant_104/backtesting/renquant_104/...`. Added
   `_resolve_trace_dir_arg()` and tests for repo-relative and strategy-relative
   trace dirs.

2. Round-trip tax allocation labels.
   `scripts/sim_trade_ledger.py` used to label every matched closed lot as
   `positive_gross`, including 156 losing rows. Tax dollars were not assigned
   to those losing rows, but the audit label was wrong. It now emits:
   `loss_no_tax`, `positive_gross_prorata`, or `event_tax_zero`.

3. WF tax metric metadata.
   `run_wf_gate.py` now imports exact annual-net APY/Sharpe from equity JSON
   and keeps event-level APY/Sharpe/tax plus annual-net tax in metadata. The
   promotion gate no longer depends on rounded console strings or event-level
   tax Sharpe when an annual-net trace exists.

### Not Fixed Yet

1. BULL_CALM entry score monotonicity is broken. This must block promotion.

2. QP `qp_min_invested_pct=0.7` may be forcing capital into a weak/anti-
   predictive BULL_CALM score slice. Needs regime-conditional A/B, not a global
   flip.

3. μ/σ horizon units are insufficiently stamped. Need a contract test that
   proves QP consumes either both 60-day units or both annualized units.

4. Gate B should be re-tested as BULL_CALM-only, not globally. The earlier
   full-OOS improvement is useful evidence, but CLAUDE.md requires
   regime-stratified WF before golden/live changes.

Artifacts:

- Equity: `backtesting/renquant_104/artifacts/diagnostics/resim_20260523_contractfix_full/xgb_full.equity.json`
- Trade report: `backtesting/renquant_104/artifacts/diagnostics/resim_20260523_contractfix_full/xgb_full.report.md`
- Round trips: `backtesting/renquant_104/artifacts/diagnostics/resim_20260523_contractfix_full/xgb_full.round_trips.csv`
- Raw trade log: `backtesting/renquant_104/artifacts/diagnostics/resim_20260523_contractfix_full/xgb_full.trades.json`
- Score diagnostic JSON: `backtesting/renquant_104/artifacts/diagnostics/resim_20260523_contractfix_full/xgb_full.score_diagnostics.json`
- Score diagnostic report: `backtesting/renquant_104/artifacts/diagnostics/resim_20260523_contractfix_full/xgb_full.score_diagnostics.md`

Relevant fixes pushed before this rerun:

- `00ed66b` — speed up and test the 104 sim feature cache path.
- `473e781` — disable missing NGBoost sim overlays and add an artifact contract test.

## Clean XGB True-OOS Result

Window: `2024-07-02` to `2026-02-10`, 404 trading days.

| metric | XGB sim | SPY same window |
|---|---:|---:|
| Total return | +4.65% | +26.07% |
| APY | +2.88% | +15.59% |
| Sharpe | +0.39 | +0.91 |
| Max drawdown | 10.26% | 19.00% |
| Annual vol | 8.14% | 17.53% |

Interpretation: the system reduced beta and drawdown, but did not harvest
enough alpha to compensate for low exposure. The sim reports beta `+0.1377`
and alpha `+0.97%/yr`; the information ratio is negative. This is not a
production-grade result.

## Trade Forensics

Closed round trips:

- Closed gross P&L: `+$756.77`
- Event-level tax: `+$7,227.68`
- Closed net after event-level tax: `-$6,470.91`
- Closed win rate: `32.14%`
- Average closed hold: `75.6d`, median `55.5d`

Open lots at sim end:

- Open gross P&L: `+$11,128.11`
- Mean open P&L: `+15.0%`
- Open lots dominate the positive final NAV.

Tax conclusion:

- Tax is still a major reporting drag under event-level accounting.
- Annual-net tax estimate is much smaller (`~$1.96k` in the report), lifting
  annual-net APY estimate to `~6.3%`.
- Even using annual-net tax, the system remains far below SPY APY/Sharpe, so
  tax is not the only root cause.

## Decision Tree Observations

Gate and optimizer counts from the clean log:

- `404` decision days.
- Regimes: `BULL_CALM=325`, `BEAR=50`, `CHOPPY=20`, `BULL_VOLATILE=9`.
- EMA50 buy block: `40` days.
- Calibrator saturation abstain: `2` days.
- Realized-vol gate: `4,341 / 42,473` candidate checks dropped (`10.2%`).
- Adaptive weak-buy veto: `30,194 / 38,136` candidate checks dropped (`79.2%`).
- Wash-sale drops: `320`.
- QP buys logged: `40`; round-trip report counts `60` buy fills/events.
- QP soft sells suppressed by tax/horizon guards: `442`.
- Missing NGBoost artifact warnings after fix: `0`.

These numbers show a highly restrictive decision tree, but not a complete
buy-kill. It buys, but it buys from a compressed score band and carries low
market exposure.

## Main Root Cause

The most important finding is not simply "tax" or "decision tree too strict."
It is weak realized discrimination at entry.

Closed winners vs closed losers:

| group | n | mean entry rank_score | mean entry μ | mean entry σ | mean pnl_pct |
|---|---:|---:|---:|---:|---:|
| gross winners | 18 | 0.6130 | 0.0297 | 0.1950 | +13.19% |
| gross losers | 38 | 0.6131 | 0.0297 | 0.2542 | -7.24% |

The entry score and μ are almost identical for winners and losers. Sigma is
meaningfully higher for losers. That means the ranking/decision stack is not
converting risk-adjusted quality into entry selection. The decision tree is
filtering many candidates, but among admitted names the score is not
discriminative enough.

The one-off observation is now a repeatable diagnostic:

```bash
python scripts/analyze_trade_score_diagnostics.py \
  --round-trips-csv backtesting/renquant_104/artifacts/diagnostics/resim_20260523_contractfix_full/xgb_full.round_trips.csv \
  --output-json backtesting/renquant_104/artifacts/diagnostics/resim_20260523_contractfix_full/xgb_full.score_diagnostics.json \
  --output-md backtesting/renquant_104/artifacts/diagnostics/resim_20260523_contractfix_full/xgb_full.score_diagnostics.md
```

Closed-trade diagnostics:

| score | n | Spearman vs P&L | top-bottom P&L spread | winner mean | loser mean |
|---|---:|---:|---:|---:|---:|
| entry_rank_score | 56 | +0.0152 | -0.30% | +0.6130 | +0.6131 |
| entry_mu | 56 | +0.0152 | -0.30% | +0.0297 | +0.0297 |
| entry_sigma | 56 | -0.4043 | -13.43% | +0.1950 | +0.2542 |
| entry_mu_over_sigma | 56 | +0.2646 | +9.56% | +0.1597 | +0.1241 |
| entry_panel_score | 56 | +0.0152 | -0.30% | +0.6130 | +0.6131 |
| entry_kelly_target_pct | 36 | -0.0651 | +2.01% | +0.1165 | +0.1065 |

This is a decision-tree problem, not only a model problem: the model score
slice that actually reaches executed buys is almost flat, while the risk
dimension remains informative. The next controlled fix must therefore act
before QP solve, not only in post-hoc reporting or end-of-sim tax accounting.

## Gate B Full-OOS Ablation

Thesis: the production entry path should not admit a candidate on calibrated
rank alone when calibrated μ/σ says the edge is too weak. This is not an
NGBoost claim in this sim. NGBoost is off/missing; μ is from the global
calibrator and σ is the realized-vol fallback. The tested mechanism is
risk-adjusted admission before QP.

Commands used the same true-OOS window as baseline (`2024-07-02` to
`2026-02-10`) and wrote artifacts under:

- `backtesting/renquant_104/artifacts/diagnostics/gateb_full_20260523/`
- configs:
  `backtesting/renquant_104/strategy_config.sim_xgb_gateb_tau12_20260523.json`,
  `backtesting/renquant_104/strategy_config.sim_xgb_gateb_tau14_20260523.json`,
  `backtesting/renquant_104/strategy_config.sim_xgb_gateb_tau16_20260523.json`

| config | APY | Sharpe | MaxDD | event tax | annual-net APY est | buys/sells | win rate | avg hold | longest no-trade |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | +2.88% | +0.39 | 10.26% | $7,228 | +6.3% | 60/40 | 32% | 84d | n/a |
| Gate B τ=0.12 | +7.71% | +0.83 | 13.85% | $9,895 | +10.7% | 59/41 | 39% | 91d | 37d |
| Gate B τ=0.14 | +4.69% | +0.53 | 13.24% | $9,567 | +7.7% | 61/44 | 36% | 90d | 22d |
| Gate B τ=0.16 | +7.02% | +0.84 | 10.24% | $7,260 | +9.5% | 52/38 | 45% | 79d | 58d |

Per CLAUDE.md, the regime view matters more than the pooled number. Every
closed entry in the Gate B sims was `BULL_CALM`, so this is a BULL_CALM result,
not a proof for BEAR/CHOPPY/BULL_VOLATILE deployment.

Closed-trade mechanics:

| config | closed gross | closed event tax | closed net | stop-loss net | trailing-stop net |
|---|---:|---:|---:|---:|---:|
| baseline | +$757 | $7,228 | -$6,471 | -$11,098 | +$5,587 |
| Gate B τ=0.12 | +$6,944 | $9,895 | -$2,951 | -$10,147 | +$7,800 |
| Gate B τ=0.14 | +$6,552 | $9,567 | -$3,015 | -$9,567 | +$7,847 |
| Gate B τ=0.16 | +$4,886 | $7,260 | -$2,374 | -$8,370 | +$6,681 |

Interpretation:

- Gate B materially improves APY/Sharpe versus the clean XGB baseline.
- The gain comes from reducing the worst stop-loss drag and preserving more
  trailing-stop winners.
- τ=0.12 has the best APY, but τ=0.16 has similar Sharpe with lower MaxDD and
  lower tax. τ=0.16 also creates a 58-day no-trade streak, which is a candidate
  starvation warning.
- No threshold is live-promotable from this single true-OOS run. It is Tier-2
  evidence for a BULL_CALM-only risk-adjusted admission gate and requires
  walk-forward/regime-stratified acceptance before touching golden.

Score diagnostics after Gate B are sobering:

| config | n closed | entry rank Spearman | entry μ/σ Spearman | top-bottom μ/σ P&L spread |
|---|---:|---:|---:|---:|
| baseline | 56 | +0.015 | +0.265 | +9.56% |
| Gate B τ=0.12 | 57 | -0.183 | -0.096 | -2.89% |
| Gate B τ=0.14 | 58 | -0.171 | +0.004 | +1.84% |
| Gate B τ=0.16 | 50 | -0.468 | -0.148 | -2.38% |

This means the ablation improves portfolio behavior, but it does not prove the
executed alpha score is discriminative inside the admitted slice. Gate B is a
damage-control admission filter, not a solved model-quality proof.

Code fix from this finding:

- `QualityFloorTask` now resolves static Gate B τ from
  `regime_params.<REGIME>.edge_sharpe_floor_threshold` or nested
  `regime_params.<REGIME>.quality_floor.edge_sharpe_floor.threshold` before
  falling back to the global threshold. This makes the knob regime-conditional
  as required by CLAUDE.md.
- Regression guard:
  `tests/test_quality_floor_gate_b.py::TestGateBRegimeConditionalRegressionGuard::test_regime_threshold_overrides_global_threshold`.
- Guard test failed before the code change and passes after it.

## Bugs Fixed During This Pass

1. Sim feature cache duplicated SPY indicator/regime work per ticker.
   Fixed by extracting `assemble_feature_frame_from_indicators()` and
   precomputing SPY indicators/context once. Tests assert cached assembly
   matches the public `build_feature_frame()` path.

2. Alpha158 frame construction was fragmenting pandas DataFrames by repeated
   column insertion. Fixed by collecting Series and constructing the frame once.
   Tests assert full-frame alpha158 cache equals single-bar inference.

3. XGB true-OOS sim had per-regime NGBoost overlays enabled while the side
   NGBoost artifact path did not exist. Fixed by disabling those overlays in
   the sim config and adding a contract test: if any sim can activate NGBoost,
   its artifact must exist.

4. Gate B admission threshold was implemented as a global static scalar when
   the architecture requires regime-conditional decision knobs. Fixed by
   resolving static Gate B τ through `regime_params` first, with global fallback
   only for unset regimes.

5. Walk-forward acceptance still used pooled absolute Sharpe as the final pass
   criterion, even though the report already computed SPY context. Fixed by
   requiring SPY-relative Sharpe/APY evidence and emitting
   `benchmark_by_dominant_regime` plus `regime_benchmark_failures`, so a
   positive-Sharpe model that loses to SPY in every cut cannot be stamped pass.

6. Walk-forward scorer/calibrator matching could still be bypassed by a static
   or config-fingerprint-only calibrator. Fixed by validating per-fold
   `calibrator_uri` against the selected scorer artifact fingerprint, including
   cached/preloaded calibrators, and by making the calibrator fitting script
   stamp scorer file identity instead of shared strategy config identity.

## Not Yet Scientifically Solved

1. Trade-level score quality is not adequate. Entry `rank_score` and μ do not
   separate winners from losers.

2. The current tax model still supports both event-level and annual-net views.
   Event-level tax is conservative but can misrepresent strategy economics
   when open gains dominate closed losses.

3. QP soft-sell suppression is very frequent (`442`), mostly because tax/horizon
   guards block sells with `expected_loss=0`. This may be correct for tax-aware
   no-trade bands, but it needs an ablation: same μ, same entries, tax soft-sell
   guard off vs on.

4. PatchTST APY/Sharpe is not scientifically available from the static seed44
   artifact for 2024/2025 historical sim. Its strict sidecar says the effective
   selection cutoff plus 60-day label horizon leaks into that period. PatchTST
   should be evaluated by strict IC now, and APY/Sharpe only through a true
   walk-forward PatchTST artifact.

## Literature Anchor

- No-trade bands and transaction-cost-aware portfolio control are grounded in
  Davis and Norman (1990), "Portfolio Selection with Transaction Costs":
  https://pubsonline.informs.org/doi/pdf/10.1287/moor.15.4.676
- Volatility-managed exposure is empirically motivated by Moreira and Muir
  (2017), "Volatility-Managed Portfolios":
  https://ideas.repec.org/a/bla/jfinan/v72y2017i4p1611-1644.html
- The literature on volatility management is mixed out of sample, so any σ
  penalty must be validated regime-by-regime, not assumed:
  https://www.sciencedirect.com/science/article/pii/S0304405X2030132X
- Alpha158-style feature construction follows Qlib benchmark conventions:
  https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md

## Next Scientific Fix Plan

1. Add acceptance gating around the trade-level score diagnostic: promotion
   should require positive separation for the executed slice, not just pooled
   panel IC. The diagnostic exists now; the promotion gate is still pending.

2. Convert the Gate B evidence into a regime-conditional WF test: enable only
   for BULL_CALM, run strict walk-forward acceptance, and reject if the
   no-trade streak / candidate starvation rises materially.

3. Run a controlled ablation panel:
   - baseline clean XGB
   - BULL_CALM Gate B τ ∈ {0.12, 0.14, 0.16}
   - QP tax soft-sell guard off
   - annual-net tax reporting only
   - combinations only after single-factor evidence is positive

4. For PatchTST, do not run static APY/Sharpe. Train or assemble true
   walk-forward PatchTST folds, then evaluate IC and trade simulation using the
   same leakage guards as XGB.
