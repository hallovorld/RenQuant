# 2026-05-23 Pending Research + PatchTST/XGB Style Handoff

This handoff is for the next RenQuant 104 session. It does not trust prior
claims unless backed by artifacts in this repo or by explicitly cited external
work. Current production code was not changed by this note.

## What I Was Doing

Before the latest instruction I was checking the `daily_104.sh` live path for a
full live-account daily run. I did not start that real-money path after the
instruction changed to PatchTST/XGB comparison and pending-task research.

In this turn I ran:

- XGB walk-forward short-window sim, leak-safe after the strict PatchTST
  artifact's label horizon:
  - config:
    `backtesting/renquant_104/strategy_config.codex_xgb_wf_safe_compare_20260523.json`
  - window: 2026-05-06 to 2026-05-22
  - result: final `$100,692`, total return `+0.69%`, APY `+15.6%`,
    Sharpe `+12.10`, MaxDD `0.04%`, buys/sells `3/0`
  - open lots: ABBV, MCD, PH, all positive at end of window
  - tax cash debited: `$0`
- SPY same short window:
  - total return `+1.61%`, APY `+39.8%`, Sharpe `+3.28`, MaxDD `-1.93%`
  - important caveat: 13 trading days is too short for a stable annualized
    APY/Sharpe comparison.
- XGB vs PatchTST descriptive style diagnostic:
  - command output: `/tmp/renquant_prod_vs_patchtst_style_eval.json`
  - sampled 45 dates from 2025-02-06 to 2026-02-10
  - this is a style/score-quality diagnostic, not a clean XGB true-OOS proof,
    because the production XGB artifact was trained later.
- PatchTST primary short-window sim:
  - config:
    `backtesting/renquant_104/strategy_config.codex_patchtst_safe_primary_20260523.json`
  - same window: 2026-05-06 to 2026-05-22
  - result: final `$103,214`, total return `+3.21%`, APY `+94.3%`,
    Sharpe `+6.61`, MaxDD `0.70%`, buys/sells `7/0`
  - open lots: ORCL, SPOT, HON, GM, LLY, DUK, IBM; six of seven positive at
    end of window.
  - tax cash debited: `$0`
  - caveat: only 13 trading days and no exits, so this is useful for
    decision-tree behavior/style but not for long-run APY/Sharpe.
- Raw TopK signal sims for XGB and strict PatchTST:
  - commands wrote `/tmp/renquant_raw_topk_xgb_prod_12d.json` and
    `/tmp/renquant_raw_topk_patchtst_strict_12d.json`
  - window: 2025-02-06 to 2026-02-10, 12 rebalance dates, hold 60 trading
    days, rebalance every 20 trading days, Top10/Bottom10.
  - this is a model-style diagnostic before QP, exits, rotation, and broker/tax
    lot handling. Because events overlap, APY/Sharpe are diagnostic only and
    should not be read as self-financing portfolio performance.

Existing prior PatchTST full-sim sidecar:

- `backtesting/renquant_104/artifacts/diagnostics/resim_20260522_apy_sharpe/patchtst_clean/`
- Reported APY `+1.49%`, Sharpe `+0.23`, MaxDD `7.39%`,
  buys/sells `147/139`.
- This is not a clean promotion-quality number because it predates the latest
  tax-cash semantics and uses a static strict PatchTST artifact over a window
  that is not a true walk-forward PatchTST acceptance run.

## PatchTST vs XGB Style

The two models are not redundant.

Short leak-safe full-sim window, 2026-05-06 to 2026-05-22:

| run | return | APY | Sharpe | MaxDD | buys/sells | end-state read |
|---|---:|---:|---:|---:|---:|---|
| PatchTST primary | `+3.21%` | `+94.3%` | `+6.61` | `0.70%` | `7/0` | More aggressive, higher volatility, picked SPOT/IBM/LLY winners. |
| XGB WF primary | `+0.69%` | `+15.6%` | `+12.10` | `0.04%` | `3/0` | More conservative, lower volatility, picked ABBV/MCD/PH. |
| SPY | `+1.61%` | `+39.8%` | `+3.28` | `1.93%` | n/a | Benchmark rally over the same very short window. |

Do not annualize this into a model verdict. With only 13 trading days and zero
exits, the meaningful information is style: PatchTST took more risk and found
different winners; XGB was more selective and smoother.

| diagnostic | XGB | PatchTST | read |
|---|---:|---:|---|
| Mean IC vs `fwd_60d_excess` | `+0.1288` | `+0.0335` | XGB is stronger in the sampled score-quality diagnostic. |
| Positive-IC days | `38/45` | `31/45` | PatchTST has a weaker but still nonzero signal. |
| Score Spearman XGB vs PatchTST | n/a | `+0.2269` | Signals are only mildly correlated. |
| Top-10 overlap | n/a | `1.24 / 10` | Top picks are mostly different. |
| Top-30 overlap | n/a | `6.67 / 30` | Broader buy lists still differ. |
| Bottom-10 overlap | n/a | `0.56 / 10` | Short/avoid lists differ even more. |

Regime split from the style diagnostic:

| regime | n dates | XGB IC | PatchTST IC | XGB/PatchTST rho | interpretation |
|---|---:|---:|---:|---:|---|
| BEAR | 4 | `+0.4207` | `+0.1249` | `+0.097` | Both positive; PatchTST adds different stress-regime ranking. |
| BULL_CALM | 22 | `+0.0469` | `-0.0313` | `+0.234` | PatchTST is not a good bull-calm primary in this sample. |
| BULL_STRONG | 5 | `+0.1634` | `+0.0661` | `+0.269` | XGB stronger, PatchTST positive. |
| BULL_VOLATILE | 5 | `+0.1417` | `+0.0797` | `+0.232` | PatchTST may help in volatile upside/stress transitions. |
| CHOPPY | 9 | `+0.1729` | `+0.1075` | `+0.240` | PatchTST looks most useful as a non-bull-calm diversifier/router arm. |

Raw TopK signal sim:

| diagnostic | XGB | strict PatchTST | read |
|---|---:|---:|---|
| pooled mean IC | `+0.1244` | `+0.0179` | XGB is materially stronger. |
| positive IC rate | `83.3%` | `58.3%` | PatchTST has signal but less consistency. |
| actual pooled APY | `+476%` | `+202%` | Overlapping events; directionally useful, not a portfolio APY. |
| actual after-tax APY stress | `+145%` | `+66%` | Same caveat; XGB has stronger raw picks. |
| alpha vs SPY | `+10.77%` | `+4.53%` | XGB raw Top10 beats SPY by more. |
| shuffle APY | `+45%` | `+180%` | PatchTST actual does not beat shuffle cleanly enough. |
| reverse APY | `+24%` | `+64%` | Both models have correct directionality vs reverse. |
| time-shift APY | `+623%` | `+307%` | Time-shift control is too strong; do not overclaim APY. |

Raw TopK regime IC:

| regime label | XGB IC | PatchTST IC | interpretation |
|---|---:|---:|---|
| HIGH_CALM | `+0.119` | `+0.056` | Both positive; XGB stronger. |
| HIGH_NORMAL | `+0.053` | `-0.017` | PatchTST weak. |
| LOW_SPIKED | `+0.275` | `+0.133` | PatchTST has useful stress/transition signal. |
| MED_CALM | `-0.120` | `-0.167` | Both weak in this small slice. |
| MED_NORMAL | `+0.024` | `-0.143` | XGB marginal, PatchTST weak. |

Working model hypothesis:

- XGB is the better default cross-sectional tabular ranker. It benefits from
  alpha158, fundamentals, PEAD/SUE, sentiment, and current cross-sectional
  normalization. This is the right primary until a challenger passes a strict
  walk-forward acceptance gate.
- PatchTST is a sequence-shape model. It can detect temporal trajectories,
  volatility clusters, and regime-transition shapes that the tabular ranker may
  compress away. Its current evidence says "shadow/router candidate", not
  "replacement primary".
- The immediate PatchTST improvement path is not to chase one high IC seed. It
  is to build a true walk-forward PatchTST manifest with causal calibrators,
  then compare per-regime IC and portfolio behavior against the XGB manifest.

## Why The PatchTST Sim Was Slow

The PatchTST primary sim triggers the same inference feature builder used by
live/LEAN:

`prepare_inference_panel_frames(...)` in
`backtesting/renquant_104/training_panel/pipeline.py`

That function rebuilds neutralized feature frames, raw factor frames, macro
betas, hourly/minute data, asset embeddings, and cross-sectional z-scores for
the full watchlist history. It is scientifically correct for train/inference
parity, but it took about 11 minutes before the first daily inference log.
After that, each daily pipeline was about 2 seconds.

Engineering follow-up:

1. Add a point-in-time inference-frame cache keyed by:
   `as_of_date`, watchlist hash, config feature-contract hash, and raw data
   max dates.
2. Store neutralized/factor/macro/embedding outputs under
   `backtesting/renquant_104/artifacts/sim/inference_frame_cache/`.
3. Add TDD:
   - cache hit returns byte-identical frames for a fixed fixture;
   - cache miss occurs when feature contract or as-of date changes;
   - live path can bypass or invalidate cache.

This is a performance bug, not a trading-signal conclusion.

## Pending Tasks And Research-Backed Solutions

### 1. PatchTST Acceptance

Current state:

- Long-running PatchTST/PatchTXT experiment families have finished and were
  reconciled in `doc/research/2026-05-23-patchtst-status.md`.
- Strict current shadow seed44 has positive but modest min-regime IC.
- Static long-window APY/Sharpe from the current artifact would be selection
  leaky because the artifact was selected on validation ending 2026-02-10 with
  a 60-business-day label horizon.

Plan:

- Build `walkforward_manifest_hf_patchtst.strict.json` analogous to the XGB
  manifest.
- For every cut: train on past-only data, train-only winsorization, embargo
  equal to label horizon, cut-local calibrator, artifact fingerprint.
- Acceptance reports must show:
  - per-cut, per-seed, per-regime IC;
  - PBO/DSR;
  - TopK/dropout-style raw signal return;
  - full sim APY/Sharpe/MaxDD/tax/turnover;
  - paired daily return comparison against XGB and SPY.

Research basis:

- PatchTST uses channel-independent patch tokens to preserve local time-series
  semantics while reducing attention cost and allowing longer history
  ([Nie et al., ICLR 2023](https://openreview.net/forum?id=Jbdc0vTOcol)).
- Purged/embargoed financial CV is needed because labels depend on future
  returns; standard K-fold is invalid for overlapping financial labels
  ([Lopez de Prado, Advances in Financial Machine Learning](https://philpapers.org/rec/LPEAIF)).
- Multiple trials require DSR/PBO-style correction before treating a best seed
  or best configuration as real
  ([Bailey and Lopez de Prado, 2014](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)).

### 2. XGB Trust And Role

Current state:

- Strict XGB walk-forward rerun over 2024-07-02 to 2026-02-10 is positive:
  event APY about `+15%`, Sharpe about `+1.7`, with much lower drawdown than
  SPY. Annual-net APY trails SPY because realized gains create tax drag.
- Score monotonicity at individual trade level is weak. The signal is
  portfolio-useful but noisy.

Plan:

- Keep XGB as primary.
- Make XGB acceptance reports always include:
  - per-regime IC;
  - score decile realized return monotonicity;
  - same-window SPY and XGB raw TopK baselines;
  - paired daily return tests;
  - transaction-cost/tax-adjusted metrics.

Research basis:

- XGBoost is a strong sparse tabular baseline with regularized boosted trees
  and efficient handling of sparse features
  ([Chen and Guestrin, 2016](https://arxiv.org/abs/1603.02754)).
- Qlib separates forecast model, portfolio strategy, recorder, and analysis;
  RenQuant should keep model score evaluation separate from execution-tree
  evaluation
  ([Qlib paper/project](https://www.microsoft.com/en-us/research/publication/qlib-an-ai-oriented-quantitative-investment-platform/),
  [Qlib strategy docs](https://qlib.readthedocs.io/en/latest/component/strategy.html)).

### 3. After-Tax APY And Turnover

Current state:

- Tax cash accounting bug is fixed: sims report tax separately and do not
  debit cash inside the event-level path.
- Tax is still economically material. Pure XGB improves event APY but can
  increase turnover and tax drag.

Plan:

- Keep event-level and annual-net metrics side by side.
- Move from "trade whenever rank improves" toward a tax/cost-aware no-trade
  region:
  - require alpha advantage to exceed estimated fee/slippage/tax drag;
  - reduce rebalances for fast-decaying weak signals;
  - pin near-long-term-gain holdings unless replacement alpha is large;
  - include tax-adjusted rotation margin in QP objective.

Research basis:

- With transaction costs, optimal policies contain a no-trade region rather
  than continuous churn
  ([Davis and Norman, 1990](https://pubsonline.informs.org/doi/abs/10.1287/moor.15.4.676)).
- With predictable returns and costs, optimal trading moves partially toward an
  aim portfolio and weights slower-decay signals more heavily
  ([Garleanu and Pedersen, 2009/2013](https://www.nber.org/papers/w15205)).

### 4. Stop-Loss Loss Bucket And Exit Science

Current state:

- In the strict WF rerun, stop-loss exits are the main gross loss bucket.
- This does not prove stops are bad; it proves the current fixed exit tree is
  doing much of the damage and needs decomposition by entry thesis.

Plan:

- Add a trade-exit attribution report by:
  - entry regime;
  - score decile;
  - holding age;
  - realized volatility;
  - drawdown path before exit;
  - exit reason.
- Test meta-label exits only as a secondary filter/sizer on top of the primary
  alpha, never as a leakage-prone direct PnL optimizer.
- Prefer volatility-scaled exits and survival/hazard-style diagnostics before
  changing fixed stop thresholds.

Research basis:

- Triple-barrier/meta-labeling ties labels to profit-taking, stop-loss, and
  time horizon rather than naive fixed-horizon labels
  ([Lopez de Prado book summary](https://philpapers.org/rec/LPEAIF),
  [Joubert meta-labeling framework](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4032018)).

### 5. Decision Tree Data Contract

Current state:

- Decision inputs now appear in trade logs for QP buys (`score_snapshot`,
  `decision_inputs`, source job/task/order type).
- User question remains valid: every component needs explicit expected input
  and output ranges, not just "it ran".

Plan:

- Write a `decision_tree_contract.md` that covers each phase:
  data freshness, regime, gates, candidates, panel scoring, calibration,
  weak-buy veto, vol gate, Kelly sizing, QP, rotation, order emission,
  persistence, ntfy.
- Add contract tests:
  - sector coverage is complete for all buyable names;
  - `rank_score` in `[0,1]`, `mu` in expected return units, `sigma > 0`;
  - QP never consumes raw uncalibrated alpha as expected return;
  - every emitted order has `event_id`, `score_snapshot`, `decision_inputs`,
    source job/task, and solver evidence;
  - sim/live/LEAN use the same universe floor and sector metadata.

Research/open-source basis:

- LEAN is an event-driven engine used for research, backtesting, and live
  trading; RenQuant should keep live and backtest semantics aligned
  ([QuantConnect LEAN docs](https://www.quantconnect.com/docs/v2/writing-algorithms/key-concepts/algorithm-engine)).
- Zipline's event-driven design and Pipeline abstraction are useful references
  for separating factor computation from trading logic
  ([Zipline docs](https://zipline.ml4trading.io/)).

### 6. NGBoost Overlay

Current state:

- The rejected NGBoost artifact with negative `val_mu_ic` should stay rejected.
- Config semantics remain confusing: global `ngboost.enabled=false` but
  regime overlays and hysteresis can still activate NGBoost behavior.

Plan:

- Rename config fields so the distinction is explicit:
  - `ngboost.global_default_enabled`
  - `ngboost.regime_overlay_enabled`
  - `ngboost.hysteresis_enabled`
- Require every promoted NGBoost artifact to pass:
  - positive mu IC;
  - sigma calibration/dispersion check;
  - no worse paired daily returns than XGB-only in the relevant regime;
  - clear shadow-to-promote decision record.

### 7. Noisy ntfy / Reopen-Cancel Alerts

Current state:

- User continues receiving noisy/error ntfy alerts.
- This has not been fully repaired in this turn.

Plan:

- Add alert taxonomy:
  - `ACTION_REQUIRED`: live order/broker state mismatch.
  - `INFO`: successful run, no orders.
  - `SUPPRESSED_DUPLICATE`: same issue within cooldown window.
  - `STALE_STATE`: state repair advisory, not emergency.
- Add idempotency key per alert:
  `date + run_type + broker + strategy + normalized_error`.
- Add tests for duplicate suppression and reopen/cancel false positives.

### 8. Full Live Daily Run

Current state:

- `scripts/daily_104.sh` uses live Alpaca for the final trade step.
- I did not run it after the latest instruction changed focus.

Plan:

- Run only when explicitly resumed, because it is real money.
- Before running:
  - confirm NYSE open/closed behavior;
  - dry-check broker account and open orders;
  - verify today's config drift guard passes;
  - tail ntfy/log output live.

## Suggested Next-Session Order

1. Implement or at least spec the inference-frame cache, because PatchTST
   primary full sim spends most of its wall time before the first bar.
2. Implement PatchTST walk-forward manifest generation and causal calibrator
   training.
3. Run PatchTST-vs-XGB paired acceptance: per-regime IC, raw TopK, full sim,
   SPY comparison, DSR/PBO.
4. Add decision-tree contract doc + tests.
5. Add tax/no-trade objective improvements to QP and rotation.
6. Fix ntfy alert idempotency.
7. Only then consider a live daily full run.

## Bottom Line

XGB remains the primary model. PatchTST should remain shadow, but it is useful:
the low top-pick overlap and positive non-bull-calm IC suggest it may become a
regime router or ensemble member after a strict walk-forward acceptance path.
The largest engineering debt is not another random model sweep; it is a
scientific acceptance harness, cached point-in-time feature frames, and
after-tax/no-trade-aware portfolio construction.
