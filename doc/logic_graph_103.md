# renquant_103 Decision Logic Graph
*Covers notebook simulation (cell 657a4a6c) and LEAN (main.py OnData).*
*Every branch, condition, and policy. Use this as the ground truth for alignment.*

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
│  ├─ Layer 1 — Hurst exponent on last 63 SPY daily returns
│  │     H > 0.55  → MOMENTUM
│  │     H < 0.45  → REVERSION
│  │     else      → AMBIGUOUS
│  │
│  ├─ Layer 2 — CUSUM changepoint on last N SPY returns
│  │     ◆ break detected?
│  │     YES → transition_countdown = TRANSITION_BARS (3)
│  │
│  ├─ Layer 3 — GMM on [10d_ret, 20d_vol, SPY_ADX, autocorr]
│  │     → P(BULL_CALM), P(BULL_VOLATILE), P(CHOPPY), P(BEAR)
│  │
│  └─ Resolve regime:
│        P(BEAR) > 0.5           → BEAR
│        hurst=MOMENTUM          → BULL_CALM
│        hurst=REVERSION         → CHOPPY
│        hurst=AMBIGUOUS         → dominant_GMM (unless BEAR → BULL_VOLATILE)
│
│     regime_confidence = GMM P(current_regime) [0.5 during transition]
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
│  │  │     ◆ days_held >= rp["max_hold_days"]?  (500 BULL/BEAR, 10 CHOPPY)
│  │  │        YES ►SELL  reason=max_hold  → continue
│  │  │
│  │  ├─ [EXIT 4] MODEL SELL  (gated by min_hold + streak)
│  │  │     ◆ min_hold_days > 0 AND days_held < min_hold_days (20)?
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
│  RANKING  (50% model score + 50% relative strength)
│  ══════════════════════════════════════════════════════════
│  ◆ len(candidates) == 0?  YES ✗ return / continue
│
│  Normalize each dimension to [0, 1]:
│     norm_model = (model_score − min_ms) / (max_ms − min_ms)   [0.5 if range=0]
│     norm_rs    = (rs_score − min_rs) / (max_rs − min_rs)      [0.5 if range=0]
│
│  combined_rank = 0.5 × norm_model + 0.5 × norm_rs
│  sort candidates by combined_rank DESC
│
├─ ══════════════════════════════════════════════════════════
│  SELECTION LOOP  (greedy, fills slots in rank order)
│  ══════════════════════════════════════════════════════════
│  slots_filled = 0
│
│  ○ for (ticker, model_score, rs_score) in ranked:
│  │  ◆ open_slots <= 0?  YES ✗ break
│  │
│  │  [CHECK 1] TIERED THRESHOLD ESCALATION
│  │  tier_idx = min(slots_filled, len(TIERED_THRESHOLDS) − 1)
│  │  tier_min = TIERED_THRESHOLDS[tier_idx]["min_model_score"]
│  │     tier 0 (1st slot today) = 0.10
│  │     tier 1 (2nd slot today) = 0.30
│  │     tier 2 (3rd+ slot today)= 0.50
│  │  ◆ model_score < tier_min?  YES ✗ skip ticker
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
│  │     cash_reserve = port_val × rp["cash_reserve_pct"]   ← regime-aware
│  │        BULL_CALM=0%, BULL_VOLATILE=20%, CHOPPY=30%, BEAR=100%
│  │     max_pos_pct  = rp["max_position_pct"]              ← scaled by confidence
│  │        BULL_CALM=15%, others vary
│  │     invest = min(cash − cash_reserve,  port_val × max_pos_pct)
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
| `max_hold_days` | 500 | 500 | 10 | 500 |
| `max_position_pct` | 0.15 | 0.15 | 0.15 | 0.0 |
| `drawdown_halt_pct` | 0.15 | 0.15 | 0.15 | 0.15 |
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
| `MIN_HOLD_DAYS` | 20 | min days before model-sell allowed |
| `WASH_SALE_DAYS` | 30 | cool-off after any exit |
| `EARNINGS_BUFFER` | 3 | days around earnings blocked |
| `CORR_THRESHOLD` | 0.70 | correlation guard |
| `MAX_POSITIONS` | 8 | max concurrent positions |
| `MAX_POSITIONS_PER_SECTOR` | 3 | sector concentration cap |
| `BEAR_DEFENSIVE_SLOTS` | 1 | max defensive positions in BEAR |
| `BEAR_DEFENSIVE_PCT` | 0.15 | defensive position size in BEAR |
| `TRANSITION_BARS` | 3 | post-CUSUM no-buy window |
| `TIERED_THRESHOLDS` | [0.10, 0.30, 0.50] | slot-N escalation |

---

## NOTEBOOK vs LEAN: KEY ALIGNMENT POINTS

| # | Policy | Notebook | LEAN | Verified |
|---|--------|----------|------|----------|
| 1 | Trailing stop uses peak_gain (HWM) | ✓ pos["high_price"] HWM | ✓ _position_high_watermarks | ✓ |
| 2 | Stop-loss from entry price (not HWM) | ✓ (entry_price − price) / entry_price | ✓ same | ✓ |
| 3 | Single-day loss gate uses prev_close | ✓ ohlcv.iloc[_idx-1] | ✓ _prev_closes[ticker] | ✓ |
| 4 | min_hold resets streak to 0 | ✓ sell_streak[t]=0 under hold | ✓ continue skips streak | ✓ |
| 5 | Consecutive sell streak (3 required) | ✓ CONSECUTIVE_SELLS=3 | ✓ _consecutive_sells_required | ✓ |
| 6 | Transition window before BEAR branch | ✓ countdown check first | ✓ same order | ✓ |
| 7 | BEAR: 1 defensive slot | ✓ BEAR_DEFENSIVE_SLOTS=1 | ✓ defensive_held>=1 → return | ✓ |
| 8 | Velocity crash filter uses cumulative return | ✓ spy_now/spy_prev-1 | ✓ np.prod(1+r)-1 | ✓ |
| 9 | SPY EMA50 uses ewm(span=50) | ✓ .ewm(span=50, adjust=False) | ✓ same | ✓ |
| 10 | min_model_score is regime-aware | ✓ rp.get("min_model_score") | ✓ regime_params.get(...) | ✓ |
| 11 | RS score = stock_20d − etf_20d | ✓ pct_change(20) | ✓ _compute_rs_score | ✓ |
| 12 | Ranking: 50/50 normalize-then-blend | ✓ explicit | ✓ norm() helper | ✓ |
| 13 | Tiered thresholds: tier_idx = min(slots_filled, N-1) | ✓ | ✓ | ✓ |
| 14 | Wash-sale checked in candidate scan AND selection | NB: scan only | LEAN: scan + selection | delta |
| 15 | Sector guard counts held + already_selected_today | ✓ held+selected | ✓ held_tickers (appended) | ✓ |
| 16 | Correlation guard checks held + already_selected | ✓ | ✓ | ✓ |
| 17 | cash_reserve deducted before invest calc | ✓ cash-cash_reserve | ✓ _execute_buy cashReserve | ✓ |
| 18 | max_position_pct scaled by regime_confidence | NB: not scaled | LEAN: scaled via _rp() | delta |
| 19 | last_sell_date.pop() on re-buy | ✓ pop() clears clock | LEAN: does NOT pop; old entry stays but days>=30 makes check pass — functionally equivalent | ✓ |
| 20 | entry_dates recorded on buy | ✓ | ✓ entry_times[ticker] | ✓ |

**Deltas (row 14, 18):** Minor divergences; row 14 is belt-and-suspenders (both block), row 18 means LEAN
sizes slightly smaller during low-confidence periods — this is intentional in LEAN (confidence scaling)
but not replicated in the notebook which uses flat percentages.

---

## DECISION PRIORITY (exit priority strictly ordered)

```
1. Trailing stop      — BULL_CALM only; peak-gain triggers, then trails HWM
2. Cumulative stop    — regime-dependent width (15% vs 5%)
2b. Single-day gate   — BULL_CALM only; catches gap-downs before cumulative fires
3. Max hold           — hard time limit (500d / 10d CHOPPY)
4. Model sell streak  — N=3 consecutive signals; gated by min_hold=20d
```

Stop-loss and single-day gate are immediate (no hold guard).
Model sell requires 20-day seasoning AND 3 consecutive signals.
