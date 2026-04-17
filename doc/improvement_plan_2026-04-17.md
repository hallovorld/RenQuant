# renquant_103 Improvement Plan — 2026-04-17

## Context

Prepared after the 2026-04-17 maintenance pass, which found and fixed:
- Wash-sale re-check missing from runner's selection loop
- Sector ETF data (XLK, XLI) not fetched by runner → RS = 0.0 for all tech/industrial stocks

Current regime: **CHOPPY** (GMM 50% confidence, CUSUM triggered).  
Market backdrop: April 2026 tariff shock; high realized volatility; SPY oscillating around EMA50.  
Active positions: AMZN, PLTR, BA — all bought within last 2 days, already -4% on AMZN/BA.

---

## Confirmed Bugs Fixed This Pass (already done)

| Bug | Fix |
|-----|-----|
| RS = 0.0 for all tech stocks | Runner now fetches sector ETFs (XLK, XLI, etc.) alongside watchlist |
| Wash-sale re-check absent in selection loop | Added re-check in runner's selection loop to match notebook + LEAN |

---

## Improvement Ideas (plan only — no implementation)

### 1. CHOPPY Stop-Loss Is Too Tight vs Min-Hold Window

**Problem**: CHOPPY regime has a 5% stop-loss but a 20-day min-hold. Positions bought in CHOPPY
can only exit via: stop-loss (5%), max-hold (23d), or model-sell (only days 20-23).
In a high-volatility tariff-shock environment, 5% swings happen intraday. AMZN was -4.2% after
1 day, BA -4.1%. These will likely be cut at 5% before getting 20 days to recover.

**Options**:
- Raise CHOPPY stop-loss from 5% → 8% to allow normal volatility room, matching BULL_CALM's
  stop relative to volatility
- OR lower CHOPPY min_hold from 20 → 5 days so the model can exit earlier if it agrees
- OR tighten CHOPPY max_position_pct from 15% → 8% (less capital at risk per trade)

**Recommended**: Raise stop to 8%, keep min_hold=20. Rationale: the 5% stop was calibrated
for normal volatility; CHOPPY in a macro shock context sees larger daily swings. Accept that some
trades will lose more, but reduce the frequency of stop-outs on temporary noise.

---

### 2. CHOPPY Entry Mode Is Too Aggressive During Macro Uncertainty

**Problem**: The "divergence" entry in CHOPPY (stock outperforming SPY) buys stocks that have had
a recent bounce. In a tariff-shock / macro correction environment, today's "outperformer" is often
a stock that bounced from an oversold level but has no fundamental support for continuation.
AMZN/BA were bought after a 1-day bounce on April 16, immediately reversing to -4%.

**Options**:
- Add a sector-momentum filter in CHOPPY: only buy if the sector ETF (e.g. XLK) is also above
  its 10-day EMA, ensuring the sector-level trend hasn't reversed
- Add an additional SPY-momentum condition for CHOPPY entry: require SPY to be up >0.5% on
  the buy day (stronger conviction that the overall market is recovering)
- Reduce max_concurrent_positions to 4 in CHOPPY (from 8) to limit exposure during uncertainty

**Recommended**: Reduce CHOPPY max_concurrent_positions to 4 and add a weaker SPY-day-momentum
condition (SPY >0% on the entry day). Cap the CHOPPY damage without over-engineering.

---

### 3. Blend Weights Currently 100% Model Score, 0% RS

**Problem**: After the last recalibration, `blend_weights = [1.0, 0.0]` — RS is entirely ignored.
Now that the sector ETF bug is fixed, RS scores will be non-zero for all stocks. The next
recalibration (daily, via `scripts/recalibrate_scores.py`) should produce meaningful RS weights.

**Action**: No code change needed. After tomorrow's model retrain + recalibration, verify in logs
that RS scores are non-zero for tech stocks and that `blend_weights[1] > 0.0`.

**Watch for**: If recalibration still produces w_rs = 0.0, investigate whether the logistic
regression can distinguish rank_score vs RS — may need to adjust normalization.

---

### 4. GMM Regime Confidence = 50% Is Too Uncertain to Trade

**Problem**: CHOPPY with 50% confidence means the model is as uncertain as a coin flip. The
transition_countdown then halts buys for 3 bars. Combined: the strategy waits 3 bars, then buys
cautiously in CHOPPY, often right before CUSUM triggers again (reset to 3 bars). This can create
a pattern where buys only happen in tiny windows between CUSUM resets.

**Options**:
- Raise the minimum GMM confidence threshold to allow buys: require confidence ≥ 60% before
  placing any new CHOPPY buy (add to `min_model_score`-style config for entry)
- Shorten transition_uncertainty_bars from 3 → 2 to reduce the "locked-out" period
- Consider a "confidence veto": if regime_confidence < 0.55, treat as BEAR (no offensive buys)
  regardless of the detected regime label

**Recommended**: Add a confidence veto — if GMM confidence < 55%, block all offensive buys
(treat as transition). This prevents the strategy from buying in genuinely ambiguous regimes.

---

### 5. Missing Sector ETF Coverage in Watchlist

**Problem**: XLK (tech sector ETF) and XLI (industrial sector ETF) are not in the watchlist.
This means:
- No direct XLK/XLI positions can be taken as defensive moves in volatile markets
- RS computation now works (bug fixed), but there's no watchlist coverage of the most liquid
  sector ETFs for rotation

**Options**:
- Add XLK, XLI to the watchlist as additional defensive/rotation candidates
- Add them as "neutral" sector positions buyable in CHOPPY/BEAR when the sector is outperforming

**Recommended**: Add XLK to the watchlist (tech sector ETF). It can act as a broad-market hedge
during CHOPPY and provides additional RS signal. XLI less important. Low priority.

---

### 6. BEAR Regime Defensive Handling: 2 Slots Instead of 1

**Problem**: In BEAR regime only 1 defensive slot is allowed (GLD/TLT/XLV/XLU). In a sustained
bear market, this means >85% of the portfolio sits in cash. While conservative, it leaves return
on the table if multiple defensives are outperforming.

**Options**:
- Raise BEAR_DEFENSIVE_SLOTS from 1 → 2, allowing e.g. GLD + TLT simultaneously
- Weight defensive slots by GMM confidence: more slots as BEAR confidence increases

**Recommended**: Raise to 2 defensive slots. Low risk (defensives rarely correlate badly). Can
double the protective return during sustained bear phases. Implement after CHOPPY/volatility fixes.

---

### 7. Min-Hold Asymmetric Code Is Dead Complexity

**Problem**: The notebook has asymmetric min_hold code (different thresholds for winners vs losers)
but comment says "disabled" and both thresholds are set to 20 days in config. The runner uses a
single threshold. This is dead code that adds cognitive overhead.

**Action**: Remove asymmetric min_hold code from notebook simulation cell; use single threshold
`MIN_HOLD_DAYS = CONFIG.get("min_hold_days", 20)`. Makes parity clearer and reduces confusion
in future audits.

---

## Priority Order

| Priority | Improvement | Complexity | Risk |
|----------|-------------|------------|------|
| P1 — Do next | Verify RS scores non-zero after tomorrow's retrain | None (observe) | None |
| P1 — Do next | CHOPPY max_concurrent_positions → 4 | Minimal config change | Low |
| P2 | GMM confidence veto (< 55% → no buys) | Small code change | Medium |
| P2 | Raise CHOPPY stop-loss 5% → 8% | Config change only | Low |
| P3 | Remove asymmetric min_hold dead code from notebook | Notebook cleanup | Low |
| P4 | Add XLK to watchlist | Data + model retrain | Low |
| P4 | Raise BEAR defensive slots 1 → 2 | Config + notebook change | Low |

---

## What NOT to Change

- **Do not change RS timeframe** (20-day in all three components, verified)
- **Do not change consecutive_sell_signals** (3 in all components, verified)
- **Do not change defensive_tickers** (all 4 in config, verified)
- **Do not add volume filter for entry** (removed intentionally — relative features encode it)
- **Do not change scoring calibration method** (sample-size-aware selection is correct)
- **Do not replace GMM with a simpler regime detector** — the 3-layer approach is the core thesis

---

## Next Validation Steps (before implementing any improvement)

1. After tomorrow's daily run: check logs for non-zero RS scores on TSLA, AMZN, GOOG, MSFT
2. After tomorrow's recalibration: check `ranking.blend_weights` in strategy_config.json
   (expect `blend_weights[1] > 0.05` if RS has predictive power)
3. Run a 2-week notebook simulation with CHOPPY max_concurrent_positions=4 vs 8 to compare
   drawdown in the April 2026 period
4. Run a simulation with confidence veto (< 55%) to check if it would have prevented
   the AMZN/BA buys yesterday (April 16)
