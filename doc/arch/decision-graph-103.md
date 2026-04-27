# renquant_103 Decision Logic Graph
*Covers notebook simulation (cell 657a4a6c, streaming detect_regime), LEAN (main.py OnData), and live runner.*
*Every branch, condition, and policy. Use this as the ground truth for alignment.*
*Last updated: 2026-04-21 — 802 tests collected (800 pass, 2 skipped). Parallel pipeline: per-ticker ThreadPoolExecutor for sell and candidate phases. TrainingPipeline also parallel per-ticker.*

> **Shared 103/104 trunk.** This decision graph is the inference trunk for both strategies.
> renquant_104 (active) inserts a single `PanelScoringJob` node between `CandidateJob` and
> `RankingJob` and adds three panel-driven policy knobs (`buy_floor`, sizing conviction,
> `rotation_advantage`). See `doc/arch/strategy-104.md` for the diff. Everything else below
> applies verbatim to both strategies.

---

## Pipeline Architecture

LEAN `OnData` and the live runner both execute via the same `InferencePipeline`. The pipeline runs in 3 phases: global sequential → per-ticker parallel → global sequential.

```
LeanAdapter.make_context(data)         RunnerAdapter.make_context()
         │                                      │
         ▼                                      ▼
 InferenceContext (populated with LEAN Portfolio / broker state + OHLCV)
         │
         ▼
 InferencePipeline.run(ctx)
 ┌────────────────────────────────────────────────────────────────────────┐
 │  Phase 1: Global sequential                                            │
 │    RegimeJob    — detect_regime() → ctx.regime, ctx.confidence         │
 │    DrawdownJob  — circuit breaker → ctx.hwm, ctx.skip_buys             │
 │    BuyGatesJob  — market gates    → ctx.buy_blocked / ctx.bear_only    │
 │                                                                        │
 │  Phase 2a: Parallel (ThreadPoolExecutor, per held ticker)              │
 │    TickerSellJob [AAPL] ──┐                                            │
 │    TickerSellJob [GOOG] ──┤─→ collect ctx.exits                        │
 │    TickerSellJob [TSLA] ──┘   (compute_exits() per ticker)             │
 │                                                                        │
 │  Phase 2b: Parallel (ThreadPoolExecutor, per candidate ticker)         │
 │    TickerCandidateJob [AMD] ──┐                                        │
 │    TickerCandidateJob [CAT] ──┤─→ collect ctx.candidates               │
 │    ...                        ┘   (score_artifact() per ticker)        │
 │                                                                        │
 │  Phase 3: Global sequential                                            │
 │    RankingJob   — blend rank_score + rs_score → ctx.ranked             │
 │    RotationJob  — held vs candidates on rank_score; tax-adj swap_margin│
 │                  emits "rotation" exits + buys → ctx.rotations         │
 │    SelectionJob — run_selection_loop() → ctx.orders                    │
 └────────────────────────────────────────────────────────────────────────┘
         │
         ▼
 LeanAdapter.commit(ctx)               RunnerAdapter.commit(ctx)
 (Liquidate / SetHoldings)             (broker.place_order / save live_state.json)
```

`SellOnlyPipeline` (intraday sell-only): Phase 1 (RegimeJob → DrawdownJob) + parallel `TickerSellJob` — no buy phase.

**Key files** (flat layout — all pipeline modules at the same level):
| File | Purpose |
|------|---------|
| `kernel/pipeline/context.py` | `InferenceContext` (~50 fields), `TickerInferenceContext` |
| `kernel/pipeline/pipeline.py` | `Task`, `Job`, `TickerJob` ABCs + `run_parallel()` |
| `kernel/pipeline/pp_inference.py` | `InferencePipeline`, `SellOnlyPipeline` (+ ticker-context builders) |
| `kernel/pipeline/pp_training.py` | `TrainingPipeline` + all training jobs/tasks |
| `kernel/pipeline/job_regime.py` | `RegimeJob` |
| `kernel/pipeline/job_drawdown.py` | `DrawdownJob` |
| `kernel/pipeline/job_gates.py` | `BuyGatesJob` |
| `kernel/pipeline/job_sell.py` | `TickerSellJob` (per-ticker `TickerJob`) |
| `kernel/pipeline/job_candidates.py` | `TickerCandidateJob` (per-ticker `TickerJob`) |
| `kernel/pipeline/job_ranking.py` | `RankingJob` |
| `kernel/pipeline/job_rotation.py` | `RotationJob` |
| `kernel/pipeline/job_selection.py` | `SelectionJob` |
| `kernel/rotation.py` | `find_rotation_pairs`, `tax_drag`, `effective_swap_margin` (pure swap-pair selector) |
| `kernel/pipeline/task_*.py` | Atomic tasks per concern (regime, drawdown, gates, sell, candidates, ranking, selection) |
| `adapters/lean.py` | `LeanAdapter` — LEAN ↔ `InferenceContext` bridge |
| `adapters/runner.py` | `RunnerAdapter` — live runner ↔ `InferenceContext` bridge |
| `main.py` | ~200-line LEAN entry point (Initialize + OnData) |

**Isolation rules:**
- `kernel/` — no `common/` imports; stdlib + numpy + pandas only (Docker-safe)
- `adapters/` — can import `kernel/` and broker libs; not used inside LEAN Docker
- `main.py` — imports `kernel/` and `adapters/lean.py` only

---

## Symbols

```
◆ = decision / condition       ► = action / side-effect
○ = loop iteration              ✗ = skip / block / continue
✓ = pass / proceed
```

---

## TOP-LEVEL LOOP

```
for each TRADING DAY  (bt_dates in notebook / OnData call in LEAN)
│
├─ [LEAN only] IsWarmingUp? ──YES──► return
│
├─ Update SPY return buffer
│
├─ ══════════════════════════════════════════════════════════
│  REGIME DETECTION (3 layers)
│  ══════════════════════════════════════════════════════════
│  │
│  ├─ Layer 1 — Hurst exponent on last 63 SPY daily returns (configurable)
│  │     H > hurst_trending_threshold (0.65)  → MOMENTUM
│  │     H < hurst_reversion_threshold (0.52) → REVERSION
│  │     else                                  → AMBIGUOUS
│  │
│  ├─ Layer 2 — CUSUM changepoint on last N SPY returns
│  │     threshold=5.5, drift=0.5, lookback=20
│  │     ◆ break detected?
│  │     YES → regime_state.countdown = TRANSITION_BARS (3)
│  │
│  ├─ Layer 3 — GMM on [10d_ret, 20d_vol, SPY_ADX, autocorr]
│  │     → P(BULL_CALM), P(BULL_VOLATILE), P(CHOPPY), P(BEAR)
│  │
│  ├─ BEAR hard override  (checked before GMM threshold)
│  │     spy_20d_ann_vol = std(spy[-20:]) × √252
│  │     spy_20d_ret     = sum(spy[-20:])
│  │     ◆ ann_vol > bear_vol_threshold (0.35)
│  │         OR spy_20d_ret < bear_return_threshold (-0.08)?
│  │       YES → hard_bear = True
│  │
│  └─ Resolve regime:
│        hard_bear OR P(BEAR) > 0.5  → BEAR
│        hurst=MOMENTUM              → BULL_CALM
│        hurst=REVERSION             → CHOPPY
│        hurst=AMBIGUOUS             → dominant_GMM (unless BEAR → BULL_VOLATILE)
│
│     regime_confidence:
│        CHOPPY → Hurst distance: (hurst_rev − H) / (hurst_rev − hurst_floor)
│                 hurst_rev=0.52, hurst_floor=0.20 (both from config)
│        BULL_CALM / BULL_VOLATILE / BEAR → GMM P(current_regime)
│        transition window → 0.5 (flat)
│     rp = regime_params[regime]   ← all thresholds read from here
│
├─ ══════════════════════════════════════════════════════════
│  PORTFOLIO MARK-TO-MARKET
│  ══════════════════════════════════════════════════════════
│  port_val = cash + Σ (shares[t] × close[t])
│  hwm      = max(hwm, port_val)
│
├─ ══════════════════════════════════════════════════════════
│  DRAWDOWN CIRCUIT BREAKER
│  ══════════════════════════════════════════════════════════
│  dd = (hwm - port_val) / hwm
│  ◆ dd >= rp["drawdown_halt_pct"]?
│     YES → skip_buys = True     (new buys blocked this day)
│     NO  → skip_buys = False
│
├─ ══════════════════════════════════════════════════════════
│  SELL LOOP  (for each held ticker)
│  ══════════════════════════════════════════════════════════
│  │
│  ○ for ticker in holdings:
│  │  ◆ price data available?  NO → continue (skip)
│  │  │
│  │  ├─ Update position HWM:
│  │  │     pos_hwm[ticker] = max(pos_hwm[ticker], current_price)
│  │  │
│  │  ├─ [EXIT 1] TRAILING STOP  (BULL_CALM: trigger=20%, trail=18%)
│  │  │     peak_gain = (pos_hwm - entry_price) / entry_price
│  │  │     ◆ ts_trigger > 0 AND ts_trail > 0?
│  │  │        YES → ◆ peak_gain >= ts_trigger?
│  │  │               YES → trail_floor = pos_hwm × (1 − ts_trail)
│  │  │                     ◆ current_price <= trail_floor?
│  │  │                        YES ►SELL  reason=trailing_stop  → continue
│  │  │
│  │  ├─ [EXIT 2] CUMULATIVE STOP-LOSS  (BULL_CALM=15%, others=5%)
│  │  │     loss = (entry_price − current_price) / entry_price
│  │  │     ◆ loss >= rp["stop_loss_pct"]?
│  │  │        YES ►SELL  reason=stop_loss  → continue
│  │  │
│  │  ├─ [EXIT 2b] SINGLE-DAY LOSS GATE  (BULL_CALM=10%, others=0%)
│  │  │     sdl_pct = rp["max_single_day_loss_pct"]
│  │  │     ◆ sdl_pct > 0?
│  │  │        YES → daily_drop = (prev_close − current_price) / prev_close
│  │  │              ◆ daily_drop >= sdl_pct?
│  │  │                 YES ►SELL  reason=single_day_loss  → continue
│  │  │
│  │  ├─ [EXIT 3] MAX HOLD
│  │  │     days_held = today − entry_date
│  │  │     ◆ days_held >= rp["max_hold_days"]?  (500 BULL/BEAR, 40 CHOPPY)
│  │  │        YES ►SELL  reason=max_hold  → continue
│  │  │
│  │  ├─ [EXIT 4a] TAX-AWARE HOLD GATE  (suppress model-sell near 1-year LT threshold)
│  │  │     lt_gate = lt_hold_gate_days (330)
│  │  │     ◆ lt_gate > 0 AND lt_gate <= days_held < 365 AND gain >= lt_hold_min_gain (10%)?
│  │  │        YES → still update sell_streak (ready when gate opens)
│  │  │              ✗ return _NO_EXIT  (suppresses model-sell; hard stops above still fire)
│  │  │
│  │  ├─ [EXIT 4b] MODEL SELL  (gated by min_hold + streak)
│  │  │     ◆ min_hold_days > 0 AND days_held < min_hold_days (30)?
│  │  │        YES → sell_streak[ticker] = 0  (streak cannot build during hold period)
│  │  │              ✗ continue  (skip model check entirely)
│  │  │        │
│  │  │        NO → run model on today's features
│  │  │              ◆ model signal == "sell"?
│  │  │                 YES → sell_streak[ticker] += 1
│  │  │                       ◆ sell_streak[ticker] >= CONSECUTIVE_SELLS (3)?
│  │  │                          YES ►SELL  reason=model_sell
│  │  │                               sell_streak[ticker] = 0
│  │  │                          NO  → wait (streak accumulating)
│  │  │                 NO  → sell_streak[ticker] = 0  (streak resets)
│  │  │
│  │  └─ [NO EXIT] → hold position
│  │
│  ► After sell loop: record last_sell_date[ticker] = today for each sold ticker
│                     update prev_closes for all tickers  (LEAN: done here, pre-buy)
│
├─ ══════════════════════════════════════════════════════════
│  BUY GATE CHECKS  (ordered, each can short-circuit to next day)
│  ══════════════════════════════════════════════════════════
│
│  ◆ open_slots <= 0  OR  skip_buys?
│     YES ✗ return / continue (no buys)
│
│  [GATE 1] TRANSITION UNCERTAINTY WINDOW
│  ◆ transition_countdown > 0?
│     YES → transition_countdown -= 1  ✗ return / continue (no buys this bar)
│
│  [GATE 2] BEAR REGIME BRANCH  ────────────────────────────────────────
│  ◆ regime == BEAR?
│     YES → ◆ skip_buys  OR  defensive_held >= BEAR_DEFENSIVE_SLOTS (1)?
│               YES ✗ return / continue
│               NO  → DEFENSIVE SCAN:
│                       for ticker in DEFENSIVE_TICKERS (GLD/TLT/XLV/XLU):
│                         ◆ already held?           YES ✗ skip
│                         ◆ wash-sale blocked?      YES ✗ skip
│                         ◆ model action == "buy"?  NO  ✗ skip
│                         ► compute model_score
│                         → add to bear_candidates
│                       sort bear_candidates by model_score DESC
│                       ► BUY best_ticker  (size = min(cash, port_val × 15%))
│               ✗ return / continue (no offensive buys in BEAR)
│
│  [GATE 3] SPY VELOCITY CRASH FILTER
│  spy_nday = cumulative SPY return over last SPY_VEL_LOOKBACK_DAYS (3)
│  ◆ spy_nday < −SPY_VEL_HALT_PCT (−3%)?
│     YES ✗ return / continue (blocks all new buys)
│
│  [GATE 4] SPY EMA50 TREND GATE
│  spy_ema50 = ewm(span=50) of SPY close
│  ◆ SPY_close < spy_ema50?
│     YES ✗ return / continue (macro downtrend — blocks all new buys)
│
├─ ══════════════════════════════════════════════════════════
│  CANDIDATE SCAN  (for each ticker in watchlist, not held)
│  ══════════════════════════════════════════════════════════
│  candidates = []
│
│  ○ for ticker in watchlist:
│  │  ◆ already held?  YES ✗ skip
│  │  ◆ price data available?  NO ✗ skip
│  │
│  │  [FILTER 1] WASH-SALE GUARD
│  │  ◆ last_sell_date[ticker] exists AND today − last_sell_date < WASH_SALE_DAYS (30)?
│  │     YES ✗ skip
│  │
│  │  [FILTER 2] EARNINGS FILTER
│  │  ◆ |today − any earnings date| <= EARNINGS_BUFFER (3 days)?
│  │     YES ✗ skip
│  │
│  │  [FILTER 3] MODEL BUY SIGNAL
│  │  run model.predict() on today's features
│  │  ◆ action == "buy"?  NO ✗ skip
│  │
│  │  [FILTER 4] MIN MODEL SCORE  (regime-aware)
│  │  model_score = model.predict_score_bulk() for today
│  │  min_score   = rp["min_model_score"]   ← per-regime, NOT hardcoded
│  │     BULL_CALM=0.10, BULL_VOLATILE=0.15, CHOPPY=0.15, BEAR=0.0
│  │  ◆ model_score < min_score?  YES ✗ skip
│  │
│  │  [SCORE] RELATIVE STRENGTH vs SECTOR ETF
│  │  rs_score = stock_20d_return − sector_ETF_20d_return
│  │
│  │  ► add (ticker, model_score, rs_score) to candidates
│
├─ ══════════════════════════════════════════════════════════
│  RANKING  (w_rank × rank score + w_rs × relative strength)
│  ══════════════════════════════════════════════════════════
│  ◆ len(candidates) == 0?  YES ✗ return / continue
│
│  Normalize each dimension to [0, 1]:
│     norm_rank = (rank_score − min_ms) / (max_ms − min_ms)     [0.5 if range=0]
│     norm_rs   = (rs_score − min_rs) / (max_rs − min_rs)       [0.5 if range=0]
│
│  Blend weights: read from strategy_config.json ranking.blend_weights
│     Default [0.5, 0.5]; updated daily by scripts/recalibrate_scores.py
│     via Pearson correlation of each signal vs actual forward outperformance
│
│  combined_rank = w_rank × norm_rank + w_rs × norm_rs
│  sort candidates by combined_rank DESC
│
├─ ══════════════════════════════════════════════════════════
│  ROTATION  (held vs candidates on calibrated rank_score)
│  ══════════════════════════════════════════════════════════
│  ◆ rotation.enabled? (config) ── NO ✗ skip block
│  ◆ regime == BEAR?               ── YES ✗ skip block
│
│  build held_scores: {ticker → hs.rank_score} for each holding
│         (skip those exiting today and those with score=None)
│  build held_meta:   {ticker → {entry_date, entry_price, current_price}}
│
│  pairs = find_rotation_pairs(held_scores, held_meta, ranked,
│                              today, rotation_cfg, tax_cfg)
│         per-position effective margin =
│            base swap_margin
│            + tax_drag(unrealized_pnl%, hold_days, ST/LT rate, threshold)
│         positions within lt_protection_days of LT threshold w/ gain → +inf
│         greedy: walk candidates in rank order; pair each with the weakest
│         eligible held that it beats by ≥ effective margin (max 2/bar)
│
│  ○ for pair in pairs:                                    [Validate guards]
│  │  ◆ wash-sale on pair.buy_ticker?     YES ✗ drop
│  │  ◆ sector cap fails on virtual set?  YES ✗ drop
│  │  ◆ correlation > 0.70 on virtual?    YES ✗ drop
│  │  → append to validated
│
│  ○ for pair in validated:                                [Emit]
│  │  ► append ExitSignal(exit_type="rotation") for sell_ticker → ctx.exits
│  │  ► size buy_ticker via compute_position_size(max_pct, reserve_pct)
│  │  ► append buy order → ctx.orders, increment counters["rotations"]
│  │  ► remove buy_ticker from ctx.ranked (avoid double-buy)
│
├─ ══════════════════════════════════════════════════════════
│  SELECTION LOOP  (greedy, fills slots in rank order)
│  ══════════════════════════════════════════════════════════
│  slots_filled = 0
│
│  ○ for (ticker, rank_score, rs_score) in ranked:
│  │  ◆ open_slots <= 0?  YES ✗ break
│  │
│  │  [CHECK 1] TIERED THRESHOLD ESCALATION
│  │  tier_idx = min(slots_filled, len(TIERED_THRESHOLDS) − 1)
│  │  tier_min = TIERED_THRESHOLDS[tier_idx]["min_model_score"]
│  │     tier 0 (1st slot today) = 0.10
│  │     tier 1 (2nd slot today) = 0.30
│  │     tier 2 (3rd+ slot today)= 0.50
│  │  ◆ rank_score < tier_min?  YES ✗ skip ticker
│  │
│  │  [CHECK 2] WASH-SALE GUARD  (second check — LEAN re-checks here after ranking)
│  │  ◆ wash-sale blocked?  YES ✗ skip
│  │
│  │  [CHECK 3] SECTOR GUARD
│  │  sector = sector_map[ticker]
│  │  ◆ ticker is DEFENSIVE?  YES → exempt (defensives can stack)
│  │  sector_count = count of (held + already_selected_today) with same sector
│  │  ◆ sector_count >= MAX_POSITIONS_PER_SECTOR (3)?  YES ✗ skip
│  │
│  │  [CHECK 4] CORRELATION GUARD
│  │  for each held_ticker and already_selected_today:
│  │     corr = corr_dict[ticker][held_ticker]
│  │     ◆ |corr| >= CORR_THRESHOLD (0.70)?  YES ✗ skip ticker (break inner loop)
│  │  ◆ all correlations below threshold?  YES → proceed
│  │
│  │  ► POSITION SIZING
│  │     cash_reserve = port_val × rp["cash_reserve_pct"] × regime_confidence
│  │        BULL_CALM=0%, BULL_VOLATILE=20%, CHOPPY=30%, BEAR=100%
│  │     max_pos_pct  = rp["max_position_pct"] × regime_confidence
│  │        BULL_CALM=15%, others vary
│  │     invest = min(cash − cash_reserve,  port_val × max_pos_pct)
│  │     [OVERSIZE FALLBACK] shares = invest / current_price
│  │     ◆ shares == 0 AND price ≤ portfolio × 25%?
│  │        YES → invest = min(portfolio × 25%, available_cash)  [oversize fallback for high-priced stocks]
│  │     ◆ invest < 100?  YES ✗ skip (insufficient capital)
│  │
│  │  ► BUY EXECUTION
│  │     shares = invest / current_price
│  │     cash  -= invest
│  │     entry_dates[ticker] = today
│  │     pos_hwm[ticker]     = current_price
│  │     sell_streak[ticker] = 0
│  │     last_sell_date.pop(ticker)   (clear wash-sale clock on re-entry)
│  │     held_tickers.append(ticker)
│  │     slots_filled += 1;  open_slots -= 1
│
└─ END OF DAY
```

---

## EXIT ACTION DETAIL

When any sell fires:
```
► SELL  ticker at current_price
   proceeds  = shares × price
   hold_days = today − entry_date
   gain      = proceeds − cost
   tax       = gain × rate  if gain > 0  (LT if hold_days >= 365, else ST)
   cash     += proceeds − tax
   last_sell_date[ticker] = today
   sell_streak[ticker]    = 0
   del holdings[ticker]
   trade_log.append({action, ticker, date, pnl_pct, hold_days, tax, exit_reason})
```

---

## REGIME PARAMETER TABLE

| Param | BULL_CALM | BULL_VOLATILE | CHOPPY | BEAR |
|-------|-----------|---------------|--------|------|
| `stop_loss_pct` | 0.15 | 0.05 | 0.05 | 0.05 |
| `max_hold_days` | 500 | 500 | 40 | 500 |
| `max_position_pct` | 0.15 | 0.20 | 0.15 | 0.0 |
| `drawdown_halt_pct` | 0.35 | 0.10 | 0.08 | 0.05 |
| `trailing_stop_trigger_pct` | 0.20 | 0.0 | 0.0 | 0.0 |
| `trailing_stop_trail_pct` | 0.18 | 0.0 | 0.0 | 0.0 |
| `max_single_day_loss_pct` | 0.10 | 0.0 | 0.0 | 0.0 |
| `min_model_score` | 0.10 | 0.15 | 0.15 | 0.0 |
| `cash_reserve_pct` | 0.0 | 0.20 | 0.30 | 1.0 |
| `spy_velocity_halt_pct` | 0.03 | 0.03 | 0.03 | N/A |
| `spy_velocity_lookback_days` | 3 | 3 | 3 | N/A |

---

## GLOBAL CONSTANTS (not regime-dependent)

| Constant | Value | Where used |
|----------|-------|------------|
| `CONSECUTIVE_SELLS` | 3 | model sell streak trigger |
| `MIN_HOLD_DAYS` | 30 | min days before model-sell allowed |
| `WASH_SALE_DAYS` | 30 | cool-off after any exit |
| `EARNINGS_BUFFER` | 3 | days around earnings blocked |
| `CORR_THRESHOLD` | 0.70 | correlation guard |
| `MAX_POSITIONS` | 8 | max concurrent positions |
| `MAX_POSITIONS_PER_SECTOR` | 3 | sector concentration cap |
| `BEAR_DEFENSIVE_SLOTS` | 1 | max defensive positions in BEAR |
| `BEAR_DEFENSIVE_PCT` | 0.15 | defensive position size in BEAR |
| `TRANSITION_BARS` | 3 | post-CUSUM no-buy window |
| `LT_HOLD_GATE_DAYS` | 330 | suppress model-sell near 1-year threshold |
| `LT_HOLD_MIN_GAIN` | 10% | min unrealized gain to activate tax gate |
| `TIERED_THRESHOLDS` | [0.10, 0.30, 0.50] | slot-N escalation |

---

## NOTEBOOK vs LEAN vs LIVE RUNNER: KEY ALIGNMENT POINTS

All three components now share the same `kernel/pipeline/` job implementations.
LEAN uses `LeanAdapter` + `InferencePipeline`; live runner uses `RunnerAdapter` + `InferencePipeline`.

| # | Policy | Notebook | LEAN/Live (via kernel.pipeline) | Verified |
|---|--------|----------|---------------------------------|----------|
| 1 | Trailing stop uses peak_gain (HWM) | ✓ pos["high_price"] HWM | ✓ SellJob → compute_exits EXIT 1 | ✓ |
| 2 | Stop-loss from entry price (not HWM) | ✓ (entry_price − price) / entry_price | ✓ SellJob → compute_exits EXIT 2 | ✓ |
| 3 | Single-day loss gate uses prev_close | ✓ ohlcv.iloc[_idx-1] | ✓ SellJob sets hs.prev_close = ohlcv[-2] | ✓ |
| 4 | min_hold resets streak to 0 | ✓ sell_streak[t]=0 under hold | ✓ SellJob → compute_exits EXIT 4b | ✓ |
| 5 | Consecutive sell streak (3 required) | ✓ CONSECUTIVE_SELLS=3 | ✓ SellJob exit_params["consecutive_sell_signals"] | ✓ |
| 6 | Transition window before BEAR branch | ✓ countdown check first | ✓ BuyGatesJob Gate 1 before Gate 2 | ✓ |
| 7 | BEAR: 1 defensive slot | ✓ BEAR_DEFENSIVE_SLOTS=1 | ✓ SelectionJob limits open_slots=1 in bear_only | ✓ |
| 8 | Velocity crash filter uses cumulative return | ✓ spy_now/spy_prev-1 | ✓ BuyGatesJob → check_spy_velocity_crash | ✓ |
| 9 | SPY EMA50 uses ewm(span=50) | ✓ .ewm(span=50, adjust=False) | ✓ BuyGatesJob → check_spy_ema_trend | ✓ |
| 10 | min_model_score is regime-aware rank-score filter | ✓ rp.get("min_model_score") on calibrated rank score | ✓ CandidateJob reads regime_p["min_model_score"] | ✓ |
| 11 | RS score = stock_20d − etf_20d | ✓ pct_change(20) | ✓ CandidateJob → compute_relative_strength | ✓ |
| 12 | Ranking: data-driven blend on rank_score + RS | ✓ w_rank/w_rs from config | ✓ RankingJob reads ranking.blend_weights | ✓ |
| 13 | Tiered thresholds: tier_idx = min(slots_filled, N-1) | ✓ | ✓ SelectionJob → run_selection_loop | ✓ |
| 14 | Wash-sale checked in candidate scan AND selection | ✓ scan + selection | ✓ CandidateJob + SelectionJob both check | ✓ |
| 15 | Sector guard counts held + already_selected_today | ✓ held+selected | ✓ SelectionContext.held_tickers (appended) | ✓ |
| 16 | Correlation guard checks held + already_selected | ✓ | ✓ SelectionJob → passes_correlation_guard | ✓ |
| 17 | cash_reserve_pct scaled by regime_confidence | ✓ port_val × reserve × confidence | ✓ SelectionJob: reserve_pct × ctx.confidence | ✓ |
| 18 | max_position_pct scaled by regime_confidence | ✓ rp["max_position_pct"] × confidence | ✓ SelectionJob: max_pct × ctx.confidence | ✓ |
| 19 | last_sell_date.pop() on re-buy | ✓ pop() clears clock | ✓ RunnerAdapter.commit() pops on buy | ✓ |
| 20 | entry_dates recorded on buy | ✓ | ✓ commit() in both adapters sets entry_dates | ✓ |
| 21 | EXIT 3 max_hold_days enforced | ✓ days_held >= max_hold_days in sell loop | ✓ SellJob → compute_exits EXIT 3 | ✓ |
| 22 | CHOPPY regime_confidence uses Hurst distance (not GMM) | ✓ (hurst_rev−H)/(hurst_rev−floor) | ✓ RegimeJob → detect_regime → compute_regime_confidence | ✓ |
| 23 | Streaming detect_regime() per bar (RegimeState persists) | ✓ RegimeState across bars | ✓ RegimeJob reads/writes ctx.regime_state | ✓ |
| 24 | BEAR hard override (vol/return threshold) | ✓ ann_vol > 0.35 or 20d_ret < -0.08 → BEAR | ✓ RegimeJob → detect_regime in kernel/regime.py | ✓ |
| 25 | Artifacts in artifacts/ subdir | ✓ STRATEGY_DIR/artifacts/ | ✓ RunnerAdapter.make_context() / lean adapter | ✓ |
| 26 | LT tax-aware hold gate | ✓ lt_hold_gate_days=330, lt_hold_min_gain=0.10 | ✓ SellJob → _build_exit_params passes both keys | ✓ |
| 27 | Cross-sectional rotation (held vs candidates) | ✓ cell calls find_rotation_pairs + guards | ✓ RotationJob (BuildPairs → ValidatePairs → EmitRotations) | ✓ |
| 28 | Tax-adjusted swap_margin (ST/LT drag) | ✓ kernel.rotation.tax_drag | ✓ same — shared kernel primitive | ✓ |
| 29 | LT-discount protection window pins position | ✓ effective_swap_margin → +inf when within lt_protection_days w/ gain | ✓ same — shared | ✓ |
| 30 | Rotation skipped in BEAR regime | ✓ BEAR `continue` before rotation block | ✓ RotationJob.should_skip on bear_only | ✓ |

**Post-migration note (2026-04-20):** LEAN `main.py` is now ~160 lines (down from 576). All decision
logic lives in `kernel/pipeline/job_*.py` + `kernel/pipeline/task_*.py` (flat layout) and is shared by LEAN and the live runner via adapters.
Notebook remains independent Python simulation code that mirrors the same kernel functions.

**CHOPPY max_hold_days raised 23 → 40** to accommodate min_hold_days=30 + 3 consecutive sell signals (otherwise model-sell is structurally unreachable in CHOPPY with short max_hold).

---

## DECISION PRIORITY (exit priority strictly ordered)

```
1. Trailing stop      — BULL_CALM only; peak-gain triggers, then trails HWM
2. Cumulative stop    — regime-dependent width (15% vs 5%)
2b. Single-day gate   — BULL_CALM only; catches gap-downs before cumulative fires
3. Max hold           — hard time limit (500d / 40d CHOPPY)
4a. Tax-aware hold gate — suppress model-sell at days 330-364 if gain ≥ 10%
4b. Model sell streak  — N=3 consecutive signals; gated by min_hold=30d
```

Stop-loss and single-day gate are immediate (no hold guard).
Tax gate suppresses model-sell near the 1-year LT threshold to protect unrealized gains.
Model sell requires 30-day seasoning AND 3 consecutive signals.
