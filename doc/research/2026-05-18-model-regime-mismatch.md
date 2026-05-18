# 2026-05-18 — Strategic finding: model is mean-reversion in a momentum market

## Verbatim user critique

> "这行情不买芯片不卖能源就买工业股？这真的有道理吗？"
> (In THIS market we're not buying chips, not selling energy, but buying industrials? Does this really make sense?)

## The data

Recent 30-day total returns (2026-04-18 → 2026-05-18):

| Sector | ETF | 30d | 5d |
|---|---|---|---|
| Technology (rally) | XLK | **+28.9%** | -0.9% |
| Industrials | XLI | +4.1% | -2.1% |
| Energy | XLE | -0.4% | +4.0% |
| Utilities | XLU | **-5.0%** | -2.8% |

Mega-cap tech leaders 30-day:
- NVDA **+26.8%**
- AAPL +16.0%
- MSFT +13.2%

Today's model picks:
- BUY DUK +6 (Utilities, -7.5% over 30d)
- TOPUP HON +2 (Industrials, -6.6% over 30d) — CANCELED
- BUY MCD +3 (Consumer Defensive) — CANCELED (calibrator bug, separate)

## The verdict

The model is **buying the losers and ignoring the winners**. This is **classic mean-reversion** behavior — and it's the WRONG bet in this market regime.

### Why this happens (structural)

Alpha158's 158 features are dominated by mean-reversion-friendly stats:
- **RSV5/10/20/30/60** — Rolling Span (close - low) / (high - low)
- **QTLU/QTLD** — quantile rank within window
- **MIN/MAX** — extreme prices in window
- **ROC** — rate-of-change (looks oversold/overbought)
- **CORR/CORD** — autocorrelation features

These features bias the model toward "this stock is at the LOW of its recent range → it's cheap → buy it". In a trending bull market for tech, NVDA at +26.8% over 30d has its RSV at the TOP of range → model rates it overbought → won't buy. HON at -6.6% has its RSV at the BOTTOM → model rates it underpriced → BUY.

This is the **fundamental signal mismatch**. Alpha158 is calibrated to a balanced/mean-reverting market regime; in a strong sector rotation (tech rally, defensive sell-off), it actively bets against the trend.

### Why the regime detector doesn't help

Today's detector verdict: `regime=BULL_CALM conf=0.63`. BULL_CALM means "trending up with low vol" which IS sort of accurate for SPY broadly. But the **sector dispersion** is dramatic (tech +29 vs utilities -5 = 34pp spread in 30d). The regime detector is a **whole-market** signal, not a sector-rotation signal.

### Why this incident matters more than MCD

The MCD-rebuy was a CALIBRATOR mechanism failure — easy to fix (Platt + flat-region guard + anti-churn). Today's verdict shipped 4 layers of protection (commit 1d0317a).

**This regime-mismatch is a STRATEGY-LEVEL issue** — not a bug, but a fundamental signal-design mismatch with current market regime. No amount of calibrator-fixing will make the model love NVDA when its alpha158 features say "overbought".

## Action

Immediate: HON top-up CANCELED on Alpaca LIVE.

Followup ideas (need to decide priority):

1. **Add momentum overlay** to alpha158 feature set
   - Top-N momentum (12-1 month, weak short-term reversal then strong medium-term momentum) per Jegadeesh-Titman 1993
   - Sector momentum (cross-sectional rank within sector) per Moskowitz-Grinblatt 1999
   - Cost: 1 day eng + retrain. Risk: could over-fit to current regime.

2. **Sector-regime-conditional weighting**
   - When TECH sector cross-sectional momentum > threshold, increase max_position_pct on tech, decrease on defensives
   - Per CLAUDE.md PRIME DIRECTIVE this is the right pattern (regime-conditional knob)
   - Need: sector momentum data (have GICS sectors now) + cross-sectional rank task

3. **Switch to a different model architecture**
   - LightGBM with momentum-aware features
   - PatchTST sequence model (captures momentum)
   - Cost: 1-2 weeks

4. **Accept the trade-off**
   - Document that this model is mean-reversion biased
   - Accept it underperforms in strong-trend regimes
   - Add a "kill switch" that pauses new buys when SPY 1-month return > +10%
   - Cheapest: 1 day add a config knob `pause_new_buys_in_strong_trend_pct`

## Recommendation

Combination of #2 (regime-conditional sizing) + #4 (kill switch).

The kill switch is the IMMEDIATE protection: when monthly tech-vs-rest dispersion is extreme (e.g., XLK/XLU spread > +20%), the model's mean-reversion bias is most dangerous. Pause new buys, let existing winners run, let regime moderate before re-engaging.

#2 is the longer-term proper fix per PRIME DIRECTIVE.

## What I owe the user

I shipped 4 layers of MCD-class fixes. They prevent **mechanism failures**. They do NOT make the model SMART about regime. The user is right to point out: "it's not enough to not buy MCD wrong; the model should be buying NVDA right".

The fix is a STRATEGY question, not a CODE question. Need to discuss whether to:
- Add momentum overlay (changes model behavior)
- Add kill switch (limits exposure when regime is hostile)
- Both

Or accept current model behavior with explicit acknowledgment that it loses in momentum regimes.
