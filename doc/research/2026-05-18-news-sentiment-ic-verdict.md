# 2026-05-18 — News sentiment IC eval verdict (FINAL)

## TL;DR

**SHELVED** — signal evaporates with proper data discipline. Initial 8-month eval looked Tier 2 SCREEN (IC +0.046); honest 6-year eval shows IC +0.006 — well within shuffle-placebo noise. **Do NOT integrate.**

## Setup evolution

| Pass | News range | Merged rows | Panel dates | sentiment_pos_share × fwd_60d IC |
|---|---|---|---|---|
| 1 (this morning) | 1y (2025-05 → 2026-05) | 10,669 | 184 | +0.046 |
| 2 (this afternoon) | 6y (2020-01 → 2026-05) | 78,751 | ~1,500 | **+0.006** |

The first pass triggered a SCREEN verdict. The second pass — **with 8× more data** — shows the signal collapses to placebo level.

## Final 6-year results

| Feature × Label | raw IC | shuffle max | ts+30 (past returns) | ts-30 (far future) | WF sign-consistent | Verdict |
|---|---|---|---|---|---|---|
| sentiment_pos_share × fwd_60d | +0.006 | +0.005 | +0.023 | +0.007 | 2/5 | NULL |
| mean_sentiment × fwd_60d | +0.003 | +0.004 | +0.051 | -0.002 | 2/5 | NULL |
| sentiment_dispersion × fwd_60d | +0.018 | +0.005 | +0.026 | +0.015 | 5/5 | marginal (placebo eats most) |
| n_articles × fwd_60d | +0.025 | +0.004 | +0.034 | +0.018 | 4/5 | marginal (placebo eats most) |
| sentiment_neg_share × fwd_60d | +0.012 | +0.006 | -0.039 | +0.004 | 5/5 | NULL |

**Best surviving features** (sentiment_dispersion + n_articles) show raw IC slightly above shuffle, but the **ts-30 placebo (sentiment shifted 30d into future, i.e. corr with returns 31-90d post-news) ALSO shows ~0.018**, meaning ~80% of the apparent signal is news endogeneity (news lags returns), not forward predictive power.

After subtracting ts-30 placebo: net forward IC ≈ +0.005-0.007. Below the +0.01 minimum required for Tier 2 SCREEN integration.

## Why the first pass was misleading

The 1-year window (2025-05 → 2026-05) coincided with a specific market regime where:
1. Heavy AI-narrative news flow correlated with semiconductor / mega-cap winners
2. News flow was concentrated in tickers that already had strong momentum
3. The cross-sectional rank correlation was inflated by an extreme few outlier events (NVDA, META, AAPL news cycles)

The 6-year sample captures multiple regimes (2020 COVID, 2021 meme stocks, 2022 bear, 2023-24 AI bull, 2025 broadening), washing out the period-specific signal.

This is exactly the failure mode:
- **López de Prado 2018 AFML §11** "Backtest Through Cross-Validation" — single-period IC inflates without out-of-sample purging
- **Hou-Xue-Zhang 2020 RFS** "Replicating Anomalies" — 65% of factor anomalies fail to replicate when extended OOS
- **Bailey-López de Prado 2014** *Notices of AMS* "Pseudo-Mathematics and Financial Charlatanism" — deflated Sharpe ratio adjusts for selection bias

## Decision

**SHELVED. Not promoted to integration.**

Closure rationale:
- 6-year IC well below promotion gate (+0.01)
- ts-30 placebo confirms most signal is news lag, not predictive
- Engineering cost (panel join + retrain + ops) not justified by +0.006 raw IC
- Lower-hanging P0 items remain (wl200 in flight, smart-orders shelved, LGBM with GICS ready after #4)

What we keep:
- ✅ Alpaca news fetcher (14-day chunked) — production-ready infra
- ✅ FinBERT scorer (MPS-batched, sanity-gated) — production-ready infra
- ✅ Per-ticker sentiment parquets (6y × 103 tickers, ~80k daily aggregates) — keep for future research (e.g. event-driven studies, intraday sentiment, news-shock asymmetry)
- ❌ Panel integration — do NOT integrate

What we drop:
- Daily news cron — not needed if not integrated
- IC eval as gating logic — closed, won't re-run unless new feature engineering

## What this saves

Engineering time NOT spent:
- Panel builder integration (1 day)
- panel-LTR retrain (3-4 hours compute)
- side config + inference path (1 day)
- Daily cron (1 day)
- Test suite for new features

Total saved: ~1 week of engineering on a feature that 6-year evidence shows is null.

## Reference for future

If we revisit (e.g. intraday horizon, event-driven model), pre-conditions:
1. Use ≥ 5y data (1y is selection-bias prone)
2. ts-30 placebo MUST be < 30% of raw IC
3. Per-regime stratified IC MUST be positive in ≥ 3 of 4 regimes
4. Subtract a "naive news-lag" baseline (lag-1 news → return correlation) before claiming forward IC
