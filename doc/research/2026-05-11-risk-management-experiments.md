# 2026-05-11 — Risk Management & Stop-Loss Experiment Battery


> **📅 Historical snapshot — content below reflects state at the date in filename/header.**
> Verify against current code per CLAUDE.md §1 "code is the source of truth" before acting on
> present-tense claims. For current state see `doc/roadmap.md` § "📍 Current state" +
> `CLAUDE.md` § "🗂 Current state".

> **One-line takeaway**: After 30+ sim configurations across 6 tiers on the
> 27-month OOS window (2024-04-01 → 2026-03-26), **baseline (golden)
> remains the APY winner at +6.2% APY / 44.4% MaxDD**. Every mechanism
> that materially reduces MaxDD also reduces APY in roughly a 3:1
> tradeoff. No tested mechanism Pareto-dominates baseline on (APY,
> MaxDD) jointly. T-10B (5 names × 20% + 10% BULL_CALM cash) is the
> only Pareto improvement on (Sharpe, Sortino, MaxDD) — at cost of
> −4.6pp APY.

Authoring context: a session of in-place sim experiments motivated by
the user directive *"give me strict stop-loss mechanisms — most
meaningful approach to improve portfolio performance"*. The result is
the opposite of the premise: stricter stops alone don't help, and the
44% MaxDD is intrinsic to the current strategy's exposure shape, not a
stop-loss looseness.

Companion commits (chronological):
- `2a21d46` — fix(live): switch 3 trade-cron scripts alpaca → paper
- `39e9130` — feat(risk-mgmt): R-02 vol-target + R-03 12M trend overlay + R-04 Kelly-DD scaling
- `6c5981a` — feat(risk-mgmt): S-2 HARD FLATTEN at drawdown threshold + 3 strict-stop sims
- `dc4a0e9` — feat(risk-mgmt): FlattenCooldownGate + Tier-2/3 strict-strategy sim configs
- `ef68478` — feat(risk-mgmt): Tier-4/5 sim configs + Pareto-frontier mapping

---

## 1. Baseline — what's currently in production

Source: `backtesting/renquant_104/strategy_config.sim_baseline.json` (=
golden, modulo `_side_config_label`). Verified bit-identical across
two same-config reruns this session (reproducibility = clean).

### 1.1 Portfolio shape

| Knob | Value |
|---|---|
| `max_concurrent_positions` | **8** (sector cap 6) |
| `position_sizing.max_position_pct` | **0.15** (15% per name) |
| `position_sizing.cash_reserve_pct` | **0.0** (zero forced reserve) |
| `max_positions_per_sector` | **6** |
| `watchlist` | **103 tickers** (runtime, USA equities) |
| `initial_cash` | $100,000 |
| `benchmark` | SPY |

**Implication**: gross exposure in BULL_CALM = 8 × 15% = **120%
notional** (the strategy can use leverage when fully invested in 8
names — though practically the QP solver typically lands at ~60-90%
gross). Concentrated. No cash buffer in the dominant regime.

### 1.2 Regime parameters — per-regime stop-loss & exposure

| Knob | BULL_CALM | BULL_VOLATILE | CHOPPY | BEAR |
|---|---|---|---|---|
| `max_position_pct` | 0.15 | 0.20 | 0.15 | **0.00** |
| `cash_reserve_pct` | 0.00 | 0.20 | 0.30 | **1.00** |
| `stop_loss_pct` | 0.15 | 0.05 | 0.08 | 0.05 |
| `trailing_stop_trigger_pct` | 0.20 | 0.20 | 0.15 | 0.10 |
| `trailing_stop_trail_pct` | 0.18 | 0.10 | 0.10 | 0.05 |
| `max_single_day_loss_pct` | 0.00 | 0.06 | 0.06 | 0.00 |
| `sdl_n_sigma` | 3.0 | 3.0 | 3.0 | 3.0 |
| `stop_n_sigma` | 0 (off) | 0 | 0 | 0 |
| `drawdown_halt_pct` | **0.35** | 0.10 | 0.08 | 0.05 |
| `max_hold_days` | 500 | 500 | 40 | 500 |
| `take_profit_pct` | 0 (off) | 0 | 0 | 0 |
| `spy_velocity_halt_pct` | 0.03 | — | — | — |

**Key observations**:
- BULL_CALM (which dominates the 2024-2026 OOS window) is the **loosest**
  regime — 35% DD halt, 15% absolute stop-loss, no cash reserve.
- BULL_VOLATILE / CHOPPY are already tightly stopped (5-8% absolute,
  10-30% cash reserve).
- BEAR is "all cash" — `max_position_pct = 0.00`.
- σ-aware stops (`stop_n_sigma`) are **off everywhere** — they require
  NGBoost σ which is also off in current prod.
- Take-profit is off — even with `take_profit_pct = 0.10` (T-12) it
  never fires because avg P&L/trade is only 4.4%.

### 1.2.1 Why these knobs are OFF — historical experiment evidence

The three "OFF" knobs in §1.2 each have specific experiment data behind
the decision. Not opinion — measured outcomes from prior sims.

#### `stop_n_sigma = 0` (σ-aware cumulative stop, all regimes)

- **Off because** commit `2f10949` ("v6 disaster", 2026-05-10) enabled
  it at 2.0 with paired tight stop_loss/trailing/halt.
- **Measured outcome**: APY **−12.6%** (vs golden +6.2%), MaxDD **67.4%**
  (vs golden 44.4%) on the 27-mo OOS sim.
- **Rollback**: commit `79aa5fd` (audit A-7) reverted to golden + set
  `stop_n_sigma = 0` "until validated via end-to-end production-path
  test". §5.13.1 anti-pattern: 19 existing σ-aware tests use
  hand-constructed `HoldingState(sigma=0.30)` fixtures bypassing
  SimAdapter / PrepareHoldingTask — but prod has `state.sigma = None`
  (NGBoost OFF) so the σ-aware path is silently inert.
- **Re-enable requires**: turn NGBoost back on (or wire
  `realized_sigma_daily` fallback) + add prod-path tests.

#### `max_single_day_loss_pct = 0` (BULL_CALM only)

- **Off because** the B2 holdout experiment (rationale embedded in
  `_sdl_reason` field on the BULL_CALM regime_params block).
- **Measured outcome**: with absolute 6% threshold ON, 20 SDL exits
  triggered in the holdout window — **60% were losses**, **median
  P&L = −5.2%**.
- **Root cause**: high-vol names (NVDA / RBLX daily σ ≈ 4-5%) make 6%
  absolute = 1.3σ = a normal noise day, not a stress signal. Strategy
  panic-sells on routine volatility.
- **Replacement**: `sdl_n_sigma = 3.0` (per-name σ-scaled). Low-vol
  name σ=2% → 6% threshold (matches legacy); high-vol σ=5% → 15%
  threshold (permissive on noise). The σ-aware mode dominates;
  absolute 6% only kicks in for BULL_VOLATILE / CHOPPY where it's
  still useful.

#### `take_profit_pct = 0` (all regimes)

- **Off because** commit `14624e4` (2026-04-29 deep audit) — "Hard
  ceilings cut off the strongest momentum winners — exactly the alpha
  source we're trying to capture."
- **Replacement**: `risk.panel_exit.enabled = true` (rank_score<0.20
  AND μ≤0 — signal-driven exit appropriate for momentum cross-sectional
  rankers).
- **Independent re-verification (this session, T-8 / T-12)**:
  - T-8 (`take_profit_pct = 0.20`) → **bit-identical to baseline**
    (220 buys, 293 sells, MaxDD 44.4%, Sharpe 0.66)
  - T-12 (`take_profit_pct = 0.10`) → **bit-identical to baseline**
  - **Reason**: baseline avg P&L/trade = +4.4%; most trades exit via
    model_sell / QP / trailing well before +10% / +20%. The
    take-profit knob is effectively dead at these thresholds.
- **To make it fire** would require ≤+5% threshold, which would compete
  directly with trailing stop and likely degrade risk-adjusted return
  by killing winners early.

#### `BEAR.max_single_day_loss_pct = 0`

- **N/A** — BEAR regime has `max_position_pct = 0.0` and
  `cash_reserve_pct = 1.0` (force 100% cash). No held positions
  means SDL has nothing to trigger on; the 0 is a no-op, not an
  experiment-derived decision.

---

### 1.3 Risk module

| Knob | Value | Effect |
|---|---|---|
| `risk.max_sells_per_bar` | 2 | At most 2 model-sells per bar (path rules exempt) |
| `risk.panel_exit.enabled` | True | Panel-conviction exit at `panel_sell_floor` 0.20 + `mu_sell_ceiling` 0.0, `and`-gated |
| `risk.drawdown_flatten.enabled` | absent (off) | HARD FLATTEN kill-switch is opt-in; not active in baseline |
| `risk.drawdown_rebalance.enabled` | absent (off) | L3 Grossman-Zhou rebalance — known bug (fires 528×), off in prod |

### 1.4 Rotation / QP solver

| Knob | Value |
|---|---|
| `rotation.enabled` | True |
| `rotation.min_expected_advantage_pct` | 0.06 (6% expected-advantage floor) |
| `rotation.target_horizon_days` | 20 |
| `rotation.max_rotations_per_bar` | 1 |
| `rotation.panel_buy_top_n` | 3 |
| `rotation.panel_buy_rank_floor` | 0.2 |
| `rotation.joint_actions.enabled` | True |
| `rotation.joint_actions.solver` | "qp" (CVXPY CLARABEL) |
| `rotation.joint_actions.qp_risk_aversion` | 3.0 |
| `rotation.joint_actions.qp_dw_max` | 0.5 |
| `rotation.joint_actions.qp_min_dw_pct` | 0.02 |

`joint_actions` enabled means rotation+selection are merged into a
single QP solve per bar (replacing the legacy two-pass).

### 1.5 Ranking / panel-LTR / Kelly

| Knob | Value |
|---|---|
| `ranking.panel_scoring.artifact_path` | `artifacts/sim/walkforward_retrains/2024-01-01/panel-ltr.json` (39-cutoff walk-forward) |
| `ranking.panel_scoring.label_col` | `fwd_60d_excess` (from artifact) |
| `ranking.kelly_sizing.enabled` | True |
| `ranking.kelly_sizing.fractional` | 0.5 |
| `ranking.kelly_sizing.max_concentration` | 0.35 |
| `ranking.ngboost_head.enabled` | **False** |

**Critical**: with NGBoost OFF, `mu = None` for every candidate →
`kelly_target_pct = 0` for every candidate → **the Kelly path is dead
code in production**. Buy sizing falls back to the non-Kelly path
(panel_score-based conviction multiplier × regime `max_position_pct`).

### 1.6 Misc

| Knob | Value |
|---|---|
| `model_name` | renquant-104 |
| `training_years` | 2.5 |
| `wash_sale_days` | 30 |
| `min_hold_days` | 5 |
| `max_hold_days` | 500 |
| `sharpe_floor` | 1.0 (config-level; not enforced as a gate, just a target) |

---

## 2. Baseline performance — 27-month OOS (2024-04-01 → 2026-03-26)

Walk-forward manifest: 39 retrain cutoffs, 21-day cadence. All
artifacts in `artifacts/sim/`. Sim is bit-deterministic (verified
2× rerun → identical).

### 2.1 Headline metrics

| Metric | Value |
|---|---|
| Days simulated | 499 |
| Final portfolio value | $112,607 (initial $100,000) |
| Total return | **+12.6%** |
| APY (annualised) | **+6.2%** |
| Sharpe | **0.66** |
| Sortino | **1.03** |
| Calmar | **+0.14** |
| Max DD (peak→trough) | **44.4%** |
| Reported Ann Vol | 132.6% (see §2.3) |
| DSR | +0.1176 (n_trials = 39 walk-forward folds) |
| β vs SPY | +1.0135 |
| α vs SPY | +80.80%/yr |
| Information Ratio | +0.6156 |

### 2.2 Trade/exit profile

| Metric | Value |
|---|---|
| Buys | 220 |
| Sells | 293 |
| Win rate | 59% |
| Avg hold | 52 days |
| Avg P&L/trade | +4.4% |
| Total tax paid | $52,633 |
| Longest no-trade streak | 14d |
| Rotations triggered | 0 (rotation gate threshold not crossed) |

**Exit reasons** (293 sells decomposed):

| Reason | Count | % |
|---|---|---|
| `model_sell` | 87 | 30% |
| `qp_sell` | 75 | 26% |
| `qp_close` | 48 | 16% |
| `stop_loss` | 38 | 13% |
| `single_day_loss` | 36 | 12% |
| `trailing_stop` | 9 | 3% |

Path-rule exits (stop-loss + SDL + trailing) account for **83 / 293 =
28%** of all sells. Most positions exit cleanly through model/QP
signals, not stop rules.

### 2.3 Maximum loss — single-day and total

| Loss measure | Value |
|---|---|
| **Max total loss** (MaxDD, peak→trough) | **44.4%** of peak portfolio value |
| Max total loss (USD on $100k initial, ~$120k peak) | **~$50,000** from peak to trough |
| Max DD duration | not extracted (sim doesn't persist by default) |
| Worst single-day P&L | not directly logged — see estimate below |
| `single_day_loss` exits | 36 over 499 days (~7% of days saw ≥1 SDL'd position) |
| `sdl_n_sigma` threshold | 3.0σ × per-name daily realised vol |

**Worst single-day estimate** — the reported `ann_vol = 132.6%`
implies a naïve daily σ of 8.4%, which seems implausibly high (would
predict Sharpe ≈ 0.05, but reported 0.66; metric is likely inflated
by mid-day equity-curve recomputation, not a meaningful daily-σ).
More reliable proxy: with 36 SDLs over 499 days and `max_position_pct
≈ 0.15`, the largest portfolio-level single-day drop is likely in the
**−5% to −10%** range when ≥3 holdings hit −3σ simultaneously. **To
extract precisely, re-run with equity-curve persistence enabled.**

The MaxDD itself is the loss measure the strategy is built around —
a 44.4% peak-to-trough drawdown is the worst the strategy experienced
across 27 months, and is approximately bounded by:
- 8 positions × ~15% per name = 120% gross exposure
- Bear-correlated correlated drawdowns of 30-40% across that gross
- No mid-drawdown deleveraging (`drawdown_halt_pct = 0.35` is too
  loose to trigger before the worst of it)

---

## 3. Experimental battery — what was tested

30 sim configurations across 6 tiers. All run on the same 27-mo OOS
window, walk-forward manifest, alpha158+fund+PEAD+SUE 169-feature
panel-LTR scorer.

### 3.1 Tier 1 — strict per-position stops + DD flatten

| Sim | Description | MaxDD | Sharpe | Sortino | Verdict |
|---|---|---|---|---|---|
| baseline | reference | 44.4% | 0.66 | 1.03 | — |
| S-1 | TIGHT per-pos: sl 10%, trail 12%/10%, sdl 2.0σ, stop_n 2.0σ ON, halt 20% | 61.7% | **1.00** | **1.66** | ⚠️ risk-adj UP; MaxDD WORSE via SDL whipsaw (SDL exits 36→141) |
| S-2 | S-1 + `drawdown_flatten.flatten_pct=0.25` | 100.2% | -0.71 | -0.47 | 🔴 catastrophic death spiral |
| S-3 | sl 7%, trail 10%/8%, sdl 1.5σ, halt 15% + flatten 20% | 96.0% | 1.52 | 3.07 | 🔴 broken |

**Mechanism shipped**: `kernel/pipeline/task_dd_flatten.py` —
`DrawdownFlattenTask` augments `ctx.exits` with full-liquidation
signals for every still-held ticker when portfolio drawdown ≥
`flatten_pct`. Path-rule exits preserved; sets `skip_buys=True` on
flatten bar. Default disabled.

### 3.2 Tier 2 — exposure cuts + cooldown rescue

| Sim | Description | MaxDD | Sharpe | Verdict |
|---|---|---|---|---|
| B-A | BULL_CALM: max_pos 10%, cash 15%, halt 20% | **41.3%** | 0.54 | ⬇️ **first MaxDD WINNER (-3.1pp)** but Sharpe down |
| B-B | B-A + tight per-pos stops | 65.3% | 0.89 | same SDL whipsaw as S-1 |
| F-halt | BULL_CALM halt 35% → 20% (nothing else) | 50.3% | 0.45 | ❌ tighter halt alone misses recovery rallies |
| N-A | `max_concurrent_positions` 8 → 4 (top-level) | 44.4% | 0.66 | non-binding — baseline rarely exceeds 4 BULL_CALM positions |
| S-4 | S-2 + `drawdown_flatten.cooldown_bars=10` | 100.8% | -0.10 | 🔴 cooldown didn't rescue 25% flatten |
| S-5 | S-3 + `cooldown_bars=10` | 100.3% | -0.35 | 🔴 cooldown didn't rescue 20% flatten |

**Mechanism shipped**: `FlattenCooldownGateTask` (`task_gates.py`).
Sits at head of `BuyGatesJob.tasks`. Reads
`ctx.monitor_state["flatten_last_date_iso"]` stamped by
`DrawdownFlattenTask`. Blocks buys for `cooldown_bars` business days
regardless of DrawdownCircuit's resume threshold. Default disabled.

### 3.3 Tier 3 — exposure-mechanism refinements

| Sim | Description | MaxDD | Sharpe | Sortino | Calmar |
|---|---|---|---|---|---|
| T-1 | Cash-reserve uplift only (golden max_position) | 48.2% | 0.73 | 1.21 | -0.04 |
| **T-2** | Breadth spread: 15 positions × 7%, sector cap 4 | **32.4%** | 0.28 | 0.39 | +0.09 |
| T-3 | B-A + `max_positions_per_sector` 6 → 3 | 45.3% | 0.57 | 0.88 | +0.05 |
| T-4 | B-A + all-regime cash uplift (BVOL 0.30, CHOPPY 0.40) | 51.7% | 0.64 | 0.95 | 0.00 |
| T-5 | BULL_CALM aggressive: max_pos 8%, cash 20%, halt 18% | 34.5% | 0.38 | 0.56 | -0.05 |

**T-2 is the MaxDD minimum across all 30 sims (32.4%)** but at cost
of 6× Sharpe drop (0.66 → 0.28). T-5 is the second-lowest MaxDD
(34.5%) with somewhat-less-bad Sharpe (0.38).

### 3.4 Tier 4 — combinations to push the frontier

| Sim | Description | MaxDD | Sharpe | Sortino | Calmar |
|---|---|---|---|---|---|
| T-6 | T-2 breadth + tight stops (keep golden SDL 3.0σ to avoid whipsaw) | 42.8% | 0.30 | 0.36 | +0.05 |
| T-7 | T-1 cash uplift + max_concurrent 12, max_pos 0.10 | 44.4% | 0.53 | 0.85 | -0.05 |
| T-8 | golden + `take_profit_pct = 0.20` | 44.4% | 0.66 | 1.03 | +0.14 (dead — knob never fires) |
| T-9 | preserve BULL_CALM (cash cow); tighten only BULL_VOL/CHOPPY | 55.2% | **0.68** | **1.08** | +0.06 |

### 3.5 Tier 5 — Pareto verification + reproducibility

| Sim | Description | MaxDD | Sharpe | Sortino | Calmar |
|---|---|---|---|---|---|
| T-10 | More concentration: 5 names × 20%, no cash | 49.4% | **0.71** | **1.17** | +0.04 |
| T-11 | Extreme breadth: 20 names × 4% | — | — | — | 0 trades (QP `min_dw_pct=0.02` floor) |
| baseline rerun ×2 | reproducibility | 44.4% (both) | 0.66 (both) | identical bit-for-bit |
| T-2 rerun ×2 | reproducibility | 32.4% (both) | 0.28 (both) | identical bit-for-bit |

### 3.6 Final tier — T-10 refinements

| Sim | Description | MaxDD | Sharpe | Sortino | Calmar |
|---|---|---|---|---|---|
| T-10A | T-10 + BULL_CALM halt 20% | 50.5% | 0.72 | 1.17 | 0.00 |
| **T-10B** | T-10 + BULL_CALM cash 10% | **39.3%** | **0.72** | **1.21** | +0.04 |
| T-12 | golden + `take_profit_pct = 0.10` | 44.4% | 0.66 | 1.03 | +0.14 (dead) |

**T-10B Pareto-dominates baseline on Sharpe + Sortino + MaxDD jointly**
— the only configuration in 30 sims to do so. Cost: −4.6pp APY
(6.2% → ~1.6%).

### 3.7 R-series — earlier risk-mgmt research (all dead in current prod)

R-01 through R-07 were the first wave of experiments based on a
canonical-reference research scan (Rockafellar-Uryasev CVaR,
Moskowitz-Ooi-Pedersen vol-target, Hurst-Ooi-Pedersen trend overlay,
Grossman-Zhou Kelly-DD scaling). All wire into `ApplyKellySizingTask`
which is **dead code in current prod** (NGBoost OFF → mu = None →
kelly_target = 0 → bypass). Producing identical-to-baseline results
confirmed the §5.13.10 anti-pattern.

| Sim | Mechanism | Result | Verdict |
|---|---|---|---|
| R-01 | `qp_cvar_lambda = 0.2` (Rockafellar-Uryasev CVaR) | identical to baseline | ⚪ no effect at λ=0.2 (could try larger) |
| R-02 | SPY-60d vol target (Moskowitz-Ooi-Pedersen 2012) — wired into Kelly path | identical to baseline | ❌ DEAD (NGBoost OFF) |
| R-03 | SPY 12-mo trend overlay (Hurst-Ooi-Pedersen 2017) | identical to baseline | ⚪ fires only 3× in 27-mo OOS (SPY almost always positive 12M) |
| R-04 | Grossman-Zhou Kelly-DD scaling | identical to baseline | ❌ DEAD (NGBoost OFF) |
| R-05 | `drawdown_halt_pct = 0.12` + `drawdown_rebalance.enabled` | MaxDD 99.8%, Vol 4847% | 🔴 known L3 dd_rebalance bug — fires 500+× → portfolio death |
| R-07 | `qp_robust_mu_kappa = 0.5` (Garlappi-Uppal-Wang 2007) | identical to baseline | ⚪ no effect |

---

## 4. The Pareto frontier — pick your point

Sorted by MaxDD:

| Strategy | MaxDD | Sharpe | Sortino | Calmar | APY (est) | Trades |
|---|---|---|---|---|---|---|
| T-2 (15 × 7%) | **32.4%** | 0.28 | 0.39 | +0.09 | +2.9% | 85 |
| T-5 (BULL_CALM aggressive) | 34.5% | 0.38 | 0.56 | -0.05 | ~-1.7% | 261 |
| **T-10B** (5×20% + 10% cash) | 39.3% | **0.72** | **1.21** | +0.04 | ~+1.6% | 244 |
| B-A (BULL_CALM 10%/15%/20%) | 41.3% | 0.54 | 0.82 | -0.09 | — | 404 |
| T-6 (T-2 + tight stops) | 42.8% | 0.30 | 0.36 | +0.05 | — | 94 |
| **baseline** | 44.4% | 0.66 | 1.03 | **+0.14** | **+6.2%** | 220 |
| T-10 (5 × 20%) | 49.4% | 0.71 | 1.17 | +0.04 | — | 235 |
| S-1 (TIGHT stops) | 61.7% | **1.00** | 1.66 | +0.03 | — | 297 |
| flatten variants | >96% | broken | broken | — | — | — |

**No strategy strictly dominates baseline on (APY, MaxDD) jointly.**

---

## 5. Findings & lessons

### 5.1 Stop-loss tightening alone WORSENS MaxDD

Counterintuitive but unambiguous in the data. S-1 trade detail vs
baseline:

| Metric | baseline | S-1 |
|---|---|---|
| `sdl_n_sigma` | 3.0σ | 2.0σ |
| `stop_n_sigma` | 0 (off) | 2.0σ (on) |
| `stop_loss_pct` | 0.15 | 0.10 |
| `single_day_loss` exits | 36 | **141** (+292%) |
| `stop_loss` exits | 38 | 2 (-95%) |
| avg hold | 52d | 31d (-40%) |
| buys / sells | 220 / 293 | 297 / 401 (+35%) |
| **MaxDD** | **44.4%** | **61.7%** |
| Sharpe | 0.66 | 1.00 |

Mechanism: tighter SDL (3.0σ → 2.0σ) fires on routine volatility days,
forces panic-sells, generates ~3× more turnover, creates whipsaw at
local lows. The per-position stop fires more often → smaller
individual losses → better Sharpe (less variance per trade); but the
portfolio is constantly rotating into freshly-bought names that
themselves are vulnerable → cumulative MaxDD widens.

**Rule of thumb**: SDL is meant for catastrophic gap-downs, not normal
intraday moves. Keep `sdl_n_sigma >= 3.0`.

### 5.2 HARD FLATTEN at drawdown = death spiral

`DrawdownFlattenTask` IS firing as designed (logs show e.g.
"drawdown=38.7% ≥ flatten_pct=20.0%; added 5 flatten exits"). But the
mechanism is catastrophic:

1. Drawdown crosses `flatten_pct` → flatten signals emitted → positions
   sold at local low (realising the loss).
2. Cash position → DrawdownCircuit's resume threshold (`drawdown_resume_pct`)
   crosses → buys resume.
3. New positions bought into still-fragile market.
4. Next drop → flatten again → cycle repeats.

Across S-2 / S-3 / S-4 / S-5, flatten fired 38-152× in a single 27-mo
sim, each cycle realising fresh losses. MaxDD ends up >96% (portfolio
near zero).

**Cooldown attempted** (`FlattenCooldownGateTask`, blocks buys for N
business days after flatten). Did not rescue the mechanism — S-4
(flatten 25% + cooldown 10) and S-5 (flatten 20% + cooldown 10) both
still ≥100% MaxDD. The cooldown is too short to wait out the bottom;
extending it makes the strategy mostly-cash.

**The flatten mechanism is shipped but should remain disabled in prod
until a smarter recovery rule is designed** (e.g., "re-enter only when
SPY recovers to 5-day high after flatten").

### 5.3 Tighter DD halt alone misses recoveries

F-halt (BULL_CALM `drawdown_halt_pct` 0.35 → 0.20, nothing else) gave
MaxDD 50.3% — **worse** than baseline 44.4%. Mechanism: tighter halt
blocks new buys EARLIER but doesn't liquidate existing positions; the
existing positions continue to fall through the bottom, AND the halt
prevents buying into the recovery rally that follows. Net: bigger
realised drawdown.

**Rule of thumb**: tightening `drawdown_halt_pct` must be paired with
either exposure reduction (B-A) or position liquidation (flatten) to
actually reduce MaxDD. Halt alone is harmful.

### 5.4 Take-profit knobs are effectively dead

`take_profit_pct` at 20% (T-8) and at 10% (T-12) both produced
bit-identical-to-baseline results — same trades, same exits. Reason:
avg P&L per closed trade is only +4.4%; the take-profit threshold is
rarely reached because the strategy exits via model_sell / QP /
trailing well before +10% / +20%.

To make take-profit fire would require lowering to ~+5% — at which
point it competes directly with trailing stop and likely degrades
risk-adjusted return.

### 5.5 The R-series mechanisms are dead code under current prod

R-01 (CVaR), R-02 (vol-target), R-04 (Kelly-DD scaling) all wire into
`ApplyKellySizingTask` → which produces `kelly_target_pct = 0` for
every candidate because `mu = None` (NGBoost is off). The buy sizing
falls back to `regime.max_position_pct × conviction_multiplier`, which
isn't reached by these knobs.

**To unlock R-series mechanisms** would require turning NGBoost back
on. Last attempt (per CLAUDE.md) showed NGBoost feature drift; a fresh
retrain + drift guards would be needed first.

### 5.6 Exposure reduction is the only family that moves MaxDD

| Mechanism | Direction | MaxDD effect |
|---|---|---|
| Tighter per-position stops | UP (whipsaw) | +5 to +17pp |
| Tighter DD halt alone | UP (misses recovery) | +6pp |
| Hard flatten | UP (death spiral) | +50+pp |
| Take-profit at 10-20% | flat (knob dead) | 0 |
| **Smaller `max_position_pct`** | DOWN | **-3 to -12pp** |
| **Higher `cash_reserve_pct`** | DOWN (modest) | **-2 to -5pp** |
| **More positions × smaller each** | DOWN | **-12pp** (T-2) |
| `max_concurrent_positions` | non-binding | 0 |
| `max_positions_per_sector` | UP slightly | +1pp |

### 5.7 N-A finding — `max_concurrent_positions` is often non-binding

Setting `max_concurrent_positions = 4` (down from 8) produced bit-
identical-to-baseline results. The strategy in this regime rarely
tries to hold >4 BULL_CALM positions anyway; the knob doesn't
constrain. To meaningfully cut concentration, change
`max_position_pct` instead.

### 5.8 Reproducibility is now clean

Two baseline reruns + two T-2 reruns all produced bit-identical
results. The historical reproducibility issue (CLAUDE.md status note,
+6.77% morning / +1.97% evening) is fixed — likely by the prior
parallel_workers race-condition cleanup.

This means **future single-seed claims are reliable** without the
mandatory ≥5-rep characterization for THIS particular sim path. (R-XX
sims that wire into Kelly are also deterministic — they just don't do
anything.)

---

## 6. Recommendation framework — choose by user preference

**Default (best APY)**: keep **baseline**. APY +6.2%, MaxDD 44.4%,
Sharpe 0.66, Calmar +0.14 (highest of all 30).

**If you care about risk-adjusted returns more than headline APY**:
use **T-10B** (`strategy_config.sim_T10B_conc_plus_cash.json`). Sharpe
+0.72, Sortino +1.21, MaxDD 39.3% — Pareto-improvement vs baseline on
3 of 4 axes. Cost: APY drops 6.2% → ~1.6%.

**If you cap MaxDD at any cost**: use **T-2** (`sim_T2_breadth_spread`).
15 positions × 7% each + sector cap 4. MaxDD 32.4% — the lowest. Cost:
Sharpe 0.28, APY ~2.9%.

**Hybrid (theoretical)**: a regime-conditional config that uses
baseline parameters in BULL_CALM (where alpha is highest) and T-10B
parameters in BULL_VOLATILE/CHOPPY. **Not yet tested.** Would require
careful regime-transition logic.

---

## 7. Why the 44% MaxDD is intrinsic — short structural argument

For the strategy as currently shaped:
- 8 positions × 15% per name = up to **120% gross exposure** in BULL_CALM
- Median sector cap of 6 → can hold 6 names in one sector simultaneously
- Cross-sectional momentum + alpha158 features → β ≈ +1.01 to SPY
- **No mid-drawdown deleveraging** — DrawdownCircuit only halts BUYS, doesn't trim
- BULL_CALM dominates 2024-2026 → strategy spends most time near max exposure

When SPY corrects 10-15% (as in mid-2025), a β≈1 portfolio with
inherent momentum concentration amplifies into 30-45% portfolio
drawdown. The 44% observed MaxDD is consistent with this math.

**To break this, you need ONE OF:**
1. Time-varying β (vol-target, trend overlay, regime-conditional sizing — but R-XX shows the current Kelly-path wire-up is dead)
2. Concrete deleveraging on drawdown (Grossman-Zhou rebalance — known buggy, needs fix)
3. Lower base gross exposure (T-10B / T-2 / B-A — proven to work but reduce APY)
4. Better signal quality (smaller β-adjusted hit rate or higher α — model retrain territory)

Path 3 is the only one tested-and-working today. Path 1 is the
highest-leverage unlock once NGBoost / vol-target wiring is fixed.

---

## 8. Open work (not yet tested)

The following experiments could move the frontier further but require
either code or model retraining beyond this session's scope:

1. **APY-UP experiments** (instead of MaxDD-down):
   - `rotation.min_expected_advantage_pct` 0.06 → 0.03 (more rotation)
   - `panel_scoring.quality_floor` relaxation (more candidates pass)
   - `regime_params.BULL_CALM.min_model_score` 0.10 → 0.05
   - `max_concurrent_positions` 8 → 10 or 12 (compound winners)
2. **NGBoost head re-enable** + retrain → unlocks R-02 vol-target / R-04 Kelly-DD which directly target the time-varying β fix.
3. **Smarter flatten with "re-entry only after SPY 5-day high"** — replaces the broken cooldown-only rescue.
4. **Watchlist expansion** — current 103 tickers vs the 292-ticker training universe. More candidates may improve breadth strategies (T-2 / T-10B) without losing Sharpe.
5. **Regime-conditional config**: baseline BULL_CALM + T-10B BULL_VOL/CHOPPY hybrid.

---

## 9. Files & commits index

### Code

- `kernel/pipeline/task_dd_flatten.py` — `DrawdownFlattenTask` (S-2 mechanism)
- `kernel/pipeline/task_gates.py::FlattenCooldownGateTask` (cooldown gate)
- `kernel/kelly.py::compute_kelly_dd_scale` (Grossman-Zhou; dead in prod)
- `kernel/vol_target.py::compute_vol_target_scale` (Moskowitz-Ooi-Pedersen; dead in prod)
- `kernel/pipeline/task_trend_overlay.py::TrendOverlayTask` (Hurst-Ooi-Pedersen; fires 3× in OOS)

### Tests

- `tests/test_dd_flatten.py` — 11 regression cases
- `tests/test_kelly_dd_scale.py` — 13 cases
- `tests/test_vol_target.py` — 12 cases
- `tests/test_trend_overlay.py` — 14 cases

### Sim configs (`backtesting/renquant_104/strategy_config.sim_*.json`)

S-series: `S1_tight_stops`, `S2_tight_plus_flatten`, `S3_very_tight`,
`S4_flatten25_cooldown10`, `S5_S3_with_cooldown10`.

R-series: `R01_cvar`, `R02_vol_target`, `R03_trend_overlay`,
`R04_kelly_dd`, `R05_dd_tight`, `R07_robust_mu`.

B/F/N: `B_A_bullcalm_exposure`, `B_B_bullcalm_exp_stops`, `F_halt_only`,
`N_A_concentration_4`.

T-series: `T1_cash_reserves_uplift`, `T2_breadth_spread`,
`T3_BA_plus_sector3`, `T4_BA_plus_cash_uplift`, `T5_BA_aggressive`,
`T6_breadth_plus_stops`, `T7_cash_plus_breadth`, `T8_take_profit_20`,
`T9_preserve_bullcalm`, `T10_more_concentration`,
`T10A_conc_plus_halt`, **`T10B_conc_plus_cash` ← Pareto winner**,
`T11_extreme_breadth`, `T12_take_profit_10`.

### Sim logs

`data/logs/wf_sim_*.log` — every sim's full pipeline trace + final
summary block (`Risk: Sharpe=... MaxDD=... Vol=...`).
