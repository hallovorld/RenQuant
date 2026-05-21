# RenQuant 104 Failed-Experiment Re-study and Refactor Plan

Date: 2026-05-21

Author: Codex

Purpose: re-audit Claude Code's failed or inconclusive experiments, decide what is worth redoing, and define the first large refactor plan under `CLAUDE.md` principles.

## Executive Verdict

Do not throw away every failed experiment. Also do not trust the old reported wins.

The useful split is:

1. **True falsifications**: the idea was tested cleanly enough and failed the signal, placebo, or economic gate. These should stay closed unless the data universe changes.
2. **Contaminated falsifications**: the experiment was run before Bug C, wrote to dead config paths, used stale artifacts, used pooled metrics, or had no same-window baseline. These are not evidence.
3. **Signal-positive but portfolio-unproven**: IC, top-decile return, or shadow scores look positive, but APY/Sharpe after calibration, sizing, turnover, taxes, and SPY comparison is not proven.
4. **Architecture smells**: a useful hypothesis was implemented in the wrong layer, for example using `sigma` to override alpha instead of making it a risk input.

My recommendation: start the refactor at the **experiment and acceptance layer**, not by immediately changing production trading logic. The next code change should make bad experiments harder to launch and impossible to promote, while letting the current PatchTST/PatchTXT background jobs finish as an architecture screen.

## Source Policy

I used docs as evidence but treated code and current artifacts as source of truth, per `CLAUDE.md`.

Local source files read:

- `CLAUDE.md`
- `doc/research/failed-experiments-log.md`
- `doc/research/promotion-methodology.md`
- `doc/research/2026-05-11-risk-management-experiments.md`
- `doc/research/2026-05-13-final-verdict.md`
- `doc/research/2026-05-15-conditional-shorts-verdict.md`
- `doc/research/2026-05-16-regime-reeval-clean-verdicts.md`
- `doc/research/2026-05-18-patchtst-verdict.md`
- `doc/audits/patchtst_xgb_experiment_audit_2026-05-21.md`
- `backtesting/renquant_104/kernel/pipeline/pp_inference.py`
- `backtesting/renquant_104/kernel/panel_pipeline/job_panel_scoring.py`
- `scripts/patchtst_hf.py`
- `scripts/compare_arch_5cut_5seed.py`

External theory references are listed at the end.

## What Is Worth Redoing

| Priority | Topic | Redo? | Reason | Required redesign |
|---|---:|---|---|---|
| P0 | PatchTST / PatchTXT | Yes | Current run is useful, but it is not a production proof: no same-window XGB comparator in the final comparator and `winsorize_label()` uses full-panel quantiles. PatchTST theory is relevant for sequence structure, but the live question is portfolio APY/Sharpe after the real pipeline. | Let current run finish. Then run a clean resumable 5-cut x 5-seed architecture screen with train-only label transforms, same-window XGB, and portfolio WF acceptance. |
| P0 | XGB production acceptance | Yes, but as a benchmark, not as a new claim | Corrected WF retest shows XGB-only is red: strategy Sharpe/APY are negative or weak in 2/3 cuts and lose badly to SPY. This makes XGB the incumbent benchmark to beat, not a trusted winner. | Every challenger report must include strategy Sharpe/APY, SPY Sharpe/APY, and regime attribution. |
| P1 | LightGBM / CatBoost with sector categorical | Yes | Tree ensembles are well supported in cross-sectional return prediction. LightGBM's categorical/GBDT design may be a fairer sector-aware baseline than ad hoc XGB variants. Current evidence is not decisive. | Implement as a faithful Qlib-style baseline with same split, label, universe, feature set, and acceptance harness. No single IC headline. |
| P1 | NGBoost `mu,sigma` | Yes, but not as alpha override | NGBoost has useful probabilistic theory and observed `sigma` calibration signal, but historical `mu - lambda*sigma` and Kelly integration caused churn and drag. | Treat `sigma` as risk/size/uncertainty input only until calibrated by regime. Test Student-t or CQR calibration before any sigma sizing. |
| P1 | CVaR / vol targeting / DD-Kelly | Yes, selectively | Several old risk tests were pre-Bug-C or dead-path contaminated. Theory supports risk overlays, but always-on overlays can cut return more than drawdown. | Re-run as portfolio-level risk overlays with per-regime activation, same-day baseline, and SPY-relative Sharpe/MaxDD tradeoff. |
| P1 | News / sentiment | Continue, not redo from scratch | The pooled verdict was wrong by `CLAUDE.md`; regime-stratified result matches attention/recession sentiment theory. Already wired. | Accumulate live/shadow regime evidence; re-evaluate only by active regimes, not pooled mean. |
| P1 | Watchlist breadth | Re-audit | `wl200` breadth has theory support, but transfer coefficient can collapse. A breadth claim without old-vs-new WF and per-regime capacity is incomplete. | Run old-universe vs wl200 same-window WF with turnover, sector concentration, and transfer coefficient estimate. |
| P2 | Long-short / shorts | Cheap gate only | Global bottom decile was positive, so full short infra is not justified. But the prior regime logic was repeatedly contaminated by pooled analysis. | Redo only the empirical pre-req gate by regime. Full implementation only if bottom decile is materially negative in BEAR/CHOPPY/BULL_VOLATILE after borrow/tax cost. |
| P2 | Multi-horizon labels | Yes, but split by decision role | `fwd20d` was marginal versus `fwd60d`, but the system has mixed holding horizons and exit horizons. A single label may be structurally wrong. | Assign roles: 5d for exit/churn risk, 20d for rotation, 60d for entry. Do not average horizons blindly. |
| P2 | Meta-label exits | Maybe later | AUC around 0.55 is weak; exits are already complicated. But meta-labeling is theoretically valid after the base decision ledger is stable. | Reopen only after decision ledger captures path-rule exits cleanly. Evaluate false-positive exit cost by regime. |
| P3 | Insider features | No, unless universe changes | Large-cap coverage was sparse and placebo persistence was 100%+. This is a true falsification for current universe. | Reopen only for small-cap universe or a much richer Form 4 feature design. |
| P3 | Vol-adjusted label in same form | No | The direct `y / vol_60d` form was strongly negative and widened tails. | Reopen only as robust target engineering with winsorization and explicit tail model. |

## What Should Not Be Redone Now

1. **Pre-Bug-C APY claims**: not evidence. They can be used only as postmortems.
2. **Global XGB promotion claims**: corrected WF says the strategy does not beat SPY.
3. **Global NGB sigma-on replacement**: the theory supports uncertainty, not blindly subtracting sigma from alpha.
4. **Single-window "beat SPY" reports**: rejected by `CLAUDE.md` and by Bailey/Lopez de Prado style multiple-testing logic.
5. **Any experiment without artifact identity**: if the evaluated artifact is not proven to be the candidate artifact, the result is invalid.

## Old vs New Architecture

| Layer | Current/old behavior | New design |
|---|---|---|
| Model output | `rank_score`, `panel_score`, `mu`, `sigma`, calibrated probability, and expected return can overwrite or stand in for one another. | Typed signal contract: `raw_score`, `rank_percentile`, `prob_outperform`, `expected_excess_return`, `sigma`, `risk_adjusted_score` are separate fields. |
| Calibration | Calibrator can drift from scorer; some scripts fit transforms after split but with full-panel preprocessing. | Calibrator is part of the model bundle. All transforms stamp train-only scope, cut, label horizon, feature schema, and support bounds. |
| Portfolio decision | Candidate ranking, Kelly, QP, top-up, trim, sell gates, and post-QP logic can all emit or mutate trades. | One allocator owns target weights. Other components may propose signals, apply constraints, or veto, but only allocator emits final deltas. |
| Regime design | Some knobs are still global or evaluated pooled-first. | Every hypothesis starts with a regime thesis and every report prints per-regime first. Pooled mean is secondary. |
| Experiment runner | Loose scripts, mixed logs, restarts overwrite outputs, no universal run manifest. | Experiment registry with `manifest.json`, immutable run id, config hash, data fingerprint, process list, output paths, and resume/skip semantics. |
| Acceptance | IC sometimes treated as enough; APY/Sharpe sometimes reported without benchmark, DSR/PBO, or sanity triad. | Acceptance harness requires same-window baseline, SPY comparator, per-regime stats, A/A, shuffled-label, time-shift placebo, DSR/PBO, and artifact identity. |
| Daily decision tree | Logs are useful but not consistently reconstructable as a DAG with inputs and outputs for every node. | Decision ledger records each Task's inputs, outputs, skip reason, and downstream effect per ticker/bar. |

Target flow:

```text
Data snapshot
  -> Feature contract
  -> Signal models
  -> Calibration bundle
  -> Candidate/risk constraints
  -> Single allocator
  -> Execution translator
  -> Decision ledger + acceptance evidence
```

## Refactor Tracks

```text
[now]                     [+1-2d]                  [+3-5d]                   [+1-2w]
Experiment registry  ---> PatchTST clean screen ---> Portfolio WF harness ---> Promotion gate v2
      |                         |                         |
      v                         v                         v
Decision ledger schema ---> train-only transforms ---> allocator ownership audit
```

### Track A: Experiment Registry

Add a small no-production-risk module/script before changing strategy logic.

Required manifest fields:

- `experiment_id`
- `hypothesis`
- `regime_thesis`
- `theory_reference`
- `baseline_artifact`
- `candidate_artifact`
- `candidate_artifact_used`
- `strategy_config_hash`
- `data_fingerprint`
- `feature_schema_hash`
- `label_schema`
- `split_contract`
- `sanity_triad_status`
- `benchmark_set`
- `process_command`
- `output_paths`
- `status`

Invariant: any run missing `regime_thesis`, `baseline_artifact`, `candidate_artifact`, or `sanity_triad_status` cannot be promoted.

### Track B: Clean Architecture Screen for PatchTST/PatchTXT

Keep the current jobs alive because they answer a narrow question: which PatchTST variant is least bad or most promising.

Then build a clean screen:

1. Fix `scripts/patchtst_hf.py::winsorize_label()` to compute label clipping bounds on `split_label == "train"` only.
2. Stamp `label_winsor_source=train_only`, `label_winsor_lo`, and `label_winsor_hi`.
3. Add same-window XGB arm to `scripts/compare_arch_5cut_5seed.py`.
4. Add resume/skip: skip completed `(cut, seed)` only if config hash matches.
5. Summarize by min-regime IC, then run portfolio WF for the best PatchTST arm versus XGB and SPY.

Promotion bar:

- PatchTST must be no worse than XGB on min-regime IC beyond a predeclared tolerance.
- Strategy APY and Sharpe must be benchmark-relative: compare to XGB and SPY in the same windows.
- No promotion from IC alone.

### Track C: Portfolio WF Harness

The current core mistake is allowing signal screens to masquerade as trading-system proof.

The harness must produce this table for every candidate:

| Metric | Required |
|---|---|
| Strategy APY / Sharpe / Sortino / MaxDD | yes |
| SPY APY / Sharpe over same dates | yes |
| Delta strategy vs incumbent | yes |
| Delta strategy vs SPY | yes |
| Per-regime APY / Sharpe / trade count | yes |
| Turnover, tax, slippage proxy | yes |
| No-trade streak and QP feasibility | yes |
| DSR / PBO / trial count | yes |
| A/A, shuffled label, time-shift placebo | yes |

### Track D: Decision Ledger

The daily decision tree should become inspectable without reverse-engineering logs.

For every bar and ticker, persist:

- `task_name`
- `input_snapshot_hash`
- `input_summary`
- `output_summary`
- `decision`
- `skip_reason`
- `regime`
- `candidate_before`
- `candidate_after`
- `holding_before`
- `holding_after`
- `order_delta_before_allocator`
- `final_order_delta`

This supports the user's repeated daily-run audit question: "why did it buy, sell, or disable buy?"

### Track E: One Allocator Rule

Current `InferencePipeline` still runs post-selection `TopUpHeldTask` and `TrimHeldTask` after the phase-3 action jobs. That may be fine mechanically, but architecturally it violates the clean ownership rule unless those tasks are explicitly part of the allocator.

Refactor direction:

1. Candidate and sell jobs propose intents.
2. Risk tasks convert hard constraints to constraints.
3. Allocator solves target weights once.
4. Execution converts target deltas to broker orders.
5. Top-up/trim become allocator sub-tasks or are removed from order emission.

No production change until the decision ledger proves parity.

## Redo Design by Experiment Family

### PatchTST/PatchTXT

Theory says PatchTST can help time-series forecasting by patching local temporal structure and sharing weights across channels. But the paper's channel-independence assumption is not automatically right for cross-sectional equity ranking. This repo's cross-stock attention and FiLM variants are therefore theoretically reasonable, not decoration.

Evidence problem: current PatchTST evidence is IC/top-pick/shadow, not positive APY/Sharpe. The current run also has a label preprocessing leak.

Action: redo after current run finishes. Treat current run as architecture screening only.

### LightGBM / CatBoost

Theory says boosted trees are competitive in empirical asset pricing and LightGBM is designed for efficient high-dimensional GBDT. This is a high-quality baseline, not a random model swap.

Action: implement a faithful baseline with sector categorical support and compare against XGB under the same splits and regimes. This is a better use of time than inventing another custom transformer before PatchTST proves portfolio value.

### NGBoost and Sigma

Theory says NGBoost is for probabilistic prediction and uncertainty estimation. The previous implementation used sigma as a score override and sizing driver, which is not the same thing.

Action: redo only as calibrated uncertainty:

- regime-specific sigma reliability;
- Student-t or conformal intervals;
- sigma affects risk budget and concentration caps before it affects ranking;
- no `mu - lambda*sigma` promotion unless portfolio WF shows benchmark-relative win.

### CVaR / Vol Target / DD-Kelly

Theory supports volatility and tail-risk overlays, but only if they are portfolio-level and cost-aware. Old results were often pre-Bug-C, dead-path, or pooled.

Action: redo as per-regime risk overlays:

- BEAR/CHOPPY first;
- BULL_CALM only if theory predicts it;
- report APY paid per MaxDD point saved;
- reject if Sharpe improves only by lowering exposure while SPY dominates.

### Long-Short

Theory supports long-short only if the model can identify losers after costs. The global bottom decile being positive is a hard warning.

Action: run a cheap regime-stratified bottom-decile gate:

- If bottom decile is not negative by regime after borrow/tax/slippage assumptions, stop.
- If negative only in BEAR/CHOPPY, design shorting only for those regimes.
- Do not build full short execution until this gate passes.

### Multi-Horizon

The old "fwd20 vs fwd60" framing is too blunt. Different decisions have different natural horizons.

Action:

- entry rank: 60d;
- rotation: 20d;
- exit/churn: 5d or path-rule classifier;
- evaluate each horizon at the decision point it controls.

## Acceptance Rules

No candidate is live-promotable unless all are true:

1. Regime-first report exists.
2. Same-window incumbent and SPY comparisons exist.
3. `candidate_artifact_used=true`.
4. A/A, shuffled-label, and time-shift placebo are green or explicitly explained.
5. DSR/PBO or n>=30/t>3 style confirmation is present.
6. Portfolio WF passes, not just IC.
7. Decision ledger replay shows no new pathological buy/sell behavior.
8. Runtime preflight proves scorer/calibrator/config/data fingerprints match.

## Immediate Next Implementation Order

1. Commit this report so Claude Code has a clean target.
2. Let the current PatchTST/PatchTXT jobs continue; do not kill them.
3. Add experiment-registry manifest tooling.
4. Fix train-only PatchTST label winsorization and add regression test.
5. Add same-window XGB arm to the architecture comparator.
6. Build portfolio WF acceptance report that always includes SPY and regime metrics.
7. Start one cheap regime-stratified long-short bottom-decile gate in parallel with PatchTST cleanup.

## Theory References

- Nie et al. 2023, PatchTST, "A Time Series is Worth 64 Words": https://arxiv.org/abs/2211.14730
- Gu, Kelly, Xiu 2020, "Empirical Asset Pricing via Machine Learning": https://academic.oup.com/rfs/article/33/5/2223/5758276
- Ke et al. 2017, LightGBM: https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision
- Duan et al. 2020, NGBoost: https://proceedings.mlr.press/v119/duan20a.html
- Kelly 1956, "A New Interpretation of Information Rate": https://www.princeton.edu/~wbialek/rome/refs/kelly_56.pdf
- Markowitz 1952, "Portfolio Selection": https://doi.org/10.1111/j.1540-6261.1952.tb01525.x
- Boyd et al., CVXPortfolio / portfolio optimization with costs and constraints: https://www.cvxportfolio.com/en/1.5.0/_static/cvx_portfolio.pdf
- Bailey and Lopez de Prado 2014, Deflated Sharpe Ratio: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
- Bailey, Borwein, Lopez de Prado, Zhu 2015, Probability of Backtest Overfitting: https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf
- Romano, Patterson, Candes 2019, Conformalized Quantile Regression: https://papers.nips.cc/paper/8613-conformalized-quantile-regression
- Moreira and Muir 2017, "Volatility-Managed Portfolios": https://ideas.repec.org/a/bla/jfinan/v72y2017i4p1611-1644.html
- Garcia 2013, "Sentiment during Recessions": https://ideas.repec.org/a/bla/jfinan/v68y2013i3p1267-1300.html

