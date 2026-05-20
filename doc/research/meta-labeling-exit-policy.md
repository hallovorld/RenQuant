# Meta-Labeling for Smart Exit Policies — Literature & Implementation Plan


> **📅 Historical snapshot — content below reflects state at the date in filename/header.**
> Verify against current code per CLAUDE.md §1 "code is the source of truth" before acting on
> present-tense claims. For current state see `doc/roadmap.md` § "📍 Current state" +
> `CLAUDE.md` § "🗂 Current state".

> **Thesis**: Mechanical stop-loss rules (fixed % thresholds) exit on signal
> shape, not signal *content*. A secondary ML model trained on per-day
> position-level features can veto false-positive exits ("the stop is about
> to fire but this is just noise — hold") while preserving true-positive
> exits ("yes, this position is dying — sell"). López de Prado 2018 _AFML_
> ch.20 documents 25-40% precision/recall improvement on this exact pattern.
>
> RenQuant's current state: 38 stop_loss exits + 36 SDL exits + 9
> trailing_stop exits per 27-mo OOS sim. Of these, **median SDL P&L is
> only -2.88%** (not a catastrophe — likely false positives). Meta-labeling
> can convert these into preserved alpha.

Authored: 2026-05-11. Companion to `doc/research/2026-05-11-risk-management-experiments.md`.

---

## 1. Method overview

### 1.1 The standard ML-for-trading failure mode

A naive approach trains a classifier "buy / sell / hold" directly. This
fails because:
- Class imbalance — most bars are "hold"
- Confounds "should we trade?" with "which direction?" — two different
  decisions
- Calibration is off — predicting probabilities directly tends to be
  miscalibrated under the heavy-tail regime shifts of finance

### 1.2 Triple-Barrier labeling (López de Prado AFML ch.3)

For each candidate entry, three barriers are set:
- **Upper barrier** at entry + N×σ_daily ⇒ profit target
- **Lower barrier** at entry − M×σ_daily ⇒ stop loss
- **Vertical barrier** at entry + T days ⇒ time stop

Whichever barrier is hit *first* gives the label:
- `+1` ⇒ profit target hit ⇒ "good entry"
- `−1` ⇒ stop hit ⇒ "bad entry"
- `0` ⇒ vertical hit ⇒ "stale entry"

This collapses the messy "what's the P&L" question to a clean three-way
classification grounded in the actual price-path geometry.

### 1.3 Meta-Labeling (López de Prado AFML ch.20) — the relevant pattern

Two-stage:

**Stage 1 — Primary model** (already exists in RenQuant):
- Generates the decision *side* (buy / sell / hold) and the timing signal
- For RenQuant exits: `check_stop_loss / check_trailing_stop / check_SDL`
  in `kernel/exits.py`

**Stage 2 — Meta model** (this proposal):
- Takes the primary signal **as a given trigger event**
- Predicts the binary question: "if I act on this primary signal, will
  it have positive P&L?" ⇒ `act` ∈ {0, 1}
- Trained on features available *at the moment of the trigger*

The primary signal acts as the **filter** for what events the meta model
even considers. This avoids:
- Training on every bar (most of which are "do nothing")
- Confounding direction with timing
- Class imbalance issues (the primary signal already balances classes)

**The output structure**:
```
Primary says "EXIT"   →  Meta predicts P(profitable_exit) = 0.32  →  VETO (suppress exit)
Primary says "EXIT"   →  Meta predicts P(profitable_exit) = 0.78  →  EXECUTE exit
Primary says "HOLD"   →  Meta never queried                       →  HOLD
```

Result: same recall on true exits (catastrophic positions still flushed),
higher precision (false-positive exits suppressed).

---

## 2. Literature — annotated

| Citation | What's directly applicable |
|---|---|
| **López de Prado 2018** *Advances in Financial Machine Learning* — ch.3 (Triple-Barrier), ch.4 (Sample Weights), ch.7 (Cross-Validation in Finance), ch.20 (Meta-Labeling) | **Core methodology.** Triple-Barrier as label generator; sample weighting by overlap-uniqueness; PurgedKFold for non-IID time-series; meta-labeling theory + empirical results |
| **Almahdi & Yang 2017** *Expert Systems with Applications* 109:103-113 — "An adaptive portfolio trading system: A risk-return portfolio optimization using recurrent reinforcement learning with expected maximum drawdown" | LSTM actor-critic for adaptive stops; benchmark vs fixed stops shows +11-18% APY on 2000-2014 |
| **Deng, Bao, Kong, Ren, Dai 2017** *IEEE TNNLS* 28(3):653-664 — "Deep Direct Reinforcement Learning for Financial Signal Representation and Trading" | DDR with rolling sample; state = OHLC + tech indicators; outperformed buy-and-hold on stock index |
| **Bertoluzzo & Corazza 2007** *Computational Methods in Financial Engineering* | SVM-based timing classifier with RSI/MACD/momentum features. Older but feature-engineering reference |
| **Tharp 1998** *Trade Your Way to Financial Freedom* | R-multiples — exit unit = entry-time risk (ATR-based). Self-normalizing stop magnitude |
| **Pardo 2008** *Evaluation & Optimization of Trading Strategies* ch.10 | Walk-forward optimization + monte-carlo confidence intervals for exit parameters |
| **Hudson-Thames team 2019-2023** blog posts on `hudsonthames.org/meta-labeling` | Production case studies: equity, futures. Detailed feature engineering writeups |

---

## 3. Open-source implementations — comparison

| Repo | Stars | What we'd actually use | Pros | Cons |
|---|---|---|---|---|
| **hudson-and-thames/mlfinlab** | 4.5k | `mlfinlab.labeling.get_events` (triple-barrier) + `mlfinlab.labeling.get_bins` + `mlfinlab.labeling.add_vertical_barrier` + `mlfinlab.labeling.meta_labeling` notebook examples | ⭐ Direct López de Prado implementation — same author lineage. Reference quality. Active maintenance. | Last open release at 1.6; some newer features behind a paywall (Hudson & Thames Foundation tier) |
| **microsoft/qlib** | 14k | `qlib.contrib.strategy.rule_strategy` for layered strategies; `qlib.model.trainer.WalkForwardTrainer` for time-series-aware training | ⭐ Same Microsoft team, already aligned with `alpha158` features. Walk-forward natively supported. | Meta-labeling not a first-class abstraction — would need our own glue |
| **AI4Finance-Foundation/FinRL** | 8.9k | `finrl/agents/stablebaselines3/`: PPO / DQN exit agents; `finrl.config.config.INDICATORS` feature list | ⭐ Complete RL framework if we want Approach C later. Pre-built `StockTradingEnv` matches our buy/sell semantics | RL — heavy training, sensitive to reward shaping. Approach C territory, not A |
| **tensortrade-org/tensortrade** | 4.5k | Gym-style trading env builders for custom observation spaces | Clean abstractions; good for prototype RL agents | Lower momentum recently; smaller community |
| **stefan-jansen/machine-learning-for-trading** | 11k | Companion book repo; chapter 11 has meta-labeling end-to-end example with XGBoost | ⭐ Production-style code; XGBoost-based; same alpha158-style feature set | Notebook-organized — not a callable library |
| **backtrader/backtrader** | 13k | `ATRTrailingStop`, `ChandelierStop` indicator implementations; multi-leg exit examples | Indicator implementations are battle-tested | Backtrader's framework doesn't fit our pipeline; just borrow indicator math |

**Decision**: use `mlfinlab` for the labeling (triple-barrier + meta-label) +
**custom XGBoost classifier** (we already have XGB in production, same
binary, same `nthread=10`). No new ML framework adoption needed.

---

## 4. Feature taxonomy for the meta-label classifier

Each row of training data is a **(position, day)** snapshot, computed at
the moment a primary-signal trigger fires. ~30-40 features grouped:

### 4.1 Position-state features

| Feature | Source | Why useful |
|---|---|---|
| `cum_pnl_pct` | `HoldingState.entry_price` + `ctx.prices[t]` | Distance from entry; magnitude of move |
| `peak_gain_pct` | `HoldingState.high_watermark / entry_price - 1` | How profitable the position has been at its best |
| `drawdown_from_peak_pct` | derived from above | Current pullback from peak (the trailing-stop concept) |
| `days_held` | `today - HoldingState.entry_date` | Position maturity |
| `consec_underwater_days` | new state field | Conviction-of-loss vs single-day dip |
| `prev_day_return` | `ctx.ohlcv[ticker]["close"].pct_change().iloc[-1]` | Yesterday's move — momentum proxy |
| `gap_open_pct` | `(today_open - prev_close) / prev_close` | Overnight gap — newsfeed proxy |
| `realized_vol_20d` | `HoldingState.realized_sigma_daily` (already computed for σ-aware stop) | Volatility regime of THIS position |

### 4.2 Market-state features

| Feature | Source | Why useful |
|---|---|---|
| `spy_5d_ret`, `spy_20d_ret`, `spy_60d_ret` | `ctx.spy_returns` slice | Market regime momentum |
| `spy_realized_vol_20d` | std of recent ctx.spy_returns × √252 | Macro vol |
| `regime_label` (one-hot) | `ctx.regime` | BULL_CALM / BULL_VOLATILE / CHOPPY / BEAR |
| `regime_just_switched` | binary: regime changed in last 5 bars | Transition noise vs stable regime |

### 4.3 Technical indicators (per-ticker)

**Reuse alpha158 panel features that already exist** — no new computation:

| alpha158 field | What it captures |
|---|---|
| `RSV9` | Relative strength vs 9d range |
| `KMID`, `KMID2`, `KLEN` | Candle body / wick ratios |
| `BBPosition` (implicit) | Bollinger band position |
| `MA5_MA20_ratio` | Moving-average crossover state |
| `VOLUME_20D_Z` | Volume z-score (institutional activity proxy) |
| `RANK_VOLUME` | Cross-sectional volume rank |
| `ROC5`, `ROC20` | Rate-of-change windows |

Pull these from `panel_feature_frames[ticker].iloc[today]` (already passed
into SimAdapter / RunnerAdapter).

### 4.4 Model-signal features

| Feature | Source | Why useful |
|---|---|---|
| `panel_score_current` | `ctx.candidates` lookup or last-bar score from panel pipeline | Cross-sectional model conviction NOW |
| `panel_score_at_entry` | new HoldingState field stamped at buy | What conviction we had at entry |
| `panel_score_delta` | `current - at_entry` | Model agreement trajectory |
| `panel_score_pct_rank` | rank among current holdings | Relative weakness in portfolio |

### 4.5 Portfolio-context features

| Feature | Source | Why useful |
|---|---|---|
| `position_weight` | `shares × price / portfolio_value` | How big a hit this position represents |
| `sector_concentration` | sum of weights in same sector | Diversification context |
| `portfolio_drawdown_now` | from HWM / portfolio_value | Are we in a stressed portfolio state? |
| `n_concurrent_exits_this_bar` | count of other primary signals firing | Mass-exit context |

**Total**: ~35 features. Reasonable for an XGBoost classifier with
~3000-5000 trigger events across 27-mo OOS.

---

## 5. Label generation — triple-barrier specifics

Per (position, day) snapshot where a primary exit signal fires:

```python
forward_window = 20  # business days
upper_barrier  = entry_price * 1.10  # +10% recovery
lower_barrier  = entry_price * 0.80  # -20% deepening loss
# Vertical: end of forward_window

# Look at the price path from today over forward_window:
for day in range(1, forward_window + 1):
    p = price_path[day]
    if p >= upper_barrier:
        label = 0   # "should NOT have exited — recovered"
        break
    if p <= lower_barrier:
        label = 1   # "exit was correct — kept dropping"
        break
else:
    # Vertical: did we end up below entry_price * 0.95? → label = 1 (exit was OK)
    label = 1 if price_path[-1] < entry_price * 0.95 else 0
```

**Key parameters** (data-driven from baseline distributions §1):
- `+10%` upper: matches the p75 of profitable model_sell trades (+9.47%)
- `-20%` lower: matches the p10 of stop_loss exits (-20.82%)
- `20 days` vertical: ~1 month, close to baseline median hold of 52d / 2

These are NOT round numbers — they're percentile-anchored per CLAUDE.md §5.14.5.

---

## 6. Implementation plan — all 6 phases, no skipping

```
[BB sweep — Track A]                           [Meta-label — Track B]
       (auto-running)                                 (this plan)
                                                            │
                                                            ▼
[P4.1] Per-day position-snapshot logger             (~45min code + ~30min sim)
       New SimAdapter / RunnerAdapter hook:
       For every (held_ticker, bar) when a path-rule
       signal fires (or proactively for ALL hold-bars),
       emit a parquet row with 35 features + outcome.
       Output: data/position_day_snapshots.parquet
                                                            │
                                                            ▼
[P4.2] Triple-barrier label generator               (~30min code)
       Script: scripts/_meta_label_generate.py
       Read snapshot parquet → compute forward-20d
       price paths from each snapshot → apply triple
       barriers (+10% / -20% / 20d) → emit label.
       Output: data/position_day_labels.parquet (joined)
                                                            │
                                                            ▼
[P4.3] Train meta-label XGBoost classifier          (~30min)
       Script: scripts/_meta_label_train.py
       PurgedKFold time-series CV (López de Prado
       AFML ch.7), 5 splits with embargo=5 days.
       Output: backtesting/renquant_104/artifacts/
               meta-label-exit.json (XGB booster_raw_json +
               feature_cols + threshold + CV metrics)
                                                            │
                                                            ▼
[P4.4] Wire MetaLabelVetoTask into pipeline         (~60min code + tests)
       New task: kernel/pipeline/task_meta_label_veto.py
       Position in TickerSellJob chain:
         PrepareHolding → ScoreModel → EvaluateExits →
         [NEW MetaLabelVeto]  →  SellGateB → PanelConvictionExit →
         EarningsBlackoutSell
       Behavior:
         if tc.exit_signal and tc.exit_signal.should_exit:
             feats = extract_features(tc.holding, ctx)
             p_exit = meta_model.predict_proba(feats)[1]
             if p_exit < threshold:
                 tc.exit_signal = ExitSignal(should_exit=False,
                                              reason="meta_veto",
                                              exit_type="")
                 ctx.counters["meta_veto"] += 1
       Default: disabled when artifact missing (§5.13.10 fallback).
       Config: ranking.meta_label.{enabled, threshold, artifact_path}
       Tests: tests/test_meta_label_veto.py
              regression guards: artifact-missing→bypass, NaN-feat→bypass,
              threshold edge cases, integration through SimAdapter path
                                                            │
                                                            ▼
[P4.5] Sim with meta-label                          (~30min)
       run_sim_104.py with strategy_config.sim_meta_label.json
       Compare against:
         - baseline
         - BB DOE optimum (from Track A)
       Metrics: APY, MaxDD, Sharpe, Sortino, Calmar +
                meta_veto_count, exit-precision improvement
                                                            │
                                                            ▼
[P4.6] Pareto-front 3-way analysis + DSR/PBO        (~30min analysis)
       Script: scripts/_meta_label_pareto.py
       Plot:
         (a) BB-only Pareto front (from Track A)
         (b) BB+meta combined points
         (c) Confidence regions via PurgedKFold predictions
       Apply Bailey-López de Prado 2014 DSR with
       n_trials = BB_runs + meta_threshold_grid + 1
       Output:
         data/logs/meta_label_pareto.json
         doc/research/meta-labeling-results-2026-05-11.md
         (final winner config + DSR/PBO statement)
```

**Total estimate**: ~4h focused work + ~1.5h sim wall-time. Most of the
sim time overlaps with Track A's BB sweep (which is already running).

---

## 7. Risk register

| Risk | Mitigation |
|---|---|
| **Snapshot logging slows down SimAdapter** | Behind a config flag; default OFF; only ON during meta-label training run |
| **Triple-barrier label leakage into training** | PurgedKFold with embargo per López de Prado ch.7 — pins forward-window contamination |
| **Meta model overfits on 27-mo OOS** | Apply DSR + PBO per CLAUDE.md §5.14.4; require PBO < 50% to promote |
| **Meta veto rate too high → strategy never exits** | Hard cap on veto count per bar (e.g. ≤ 50% of triggered exits vetoed); fallback to primary signal if cap exceeded |
| **Feature drift between train and live** | Stamp `feature_fingerprint` in artifact + check at load per §5.13.13 |
| **Meta model dead in prod (cf. NGBoost)** | §5.13.10 — grep prod imports + integration test through SimAdapter, not mock |

---

## 8. What this is NOT

- ❌ NOT replacing the primary stop-loss rules — those still trigger
- ❌ NOT adding RL — that's Approach C, deferred until A is proven
- ❌ NOT another buy signal — meta-labeling is exit-side only here
- ❌ NOT trying to learn the optimal stop_loss_pct threshold — that's
  Track A (Box-Behnken). Tracks are orthogonal.
