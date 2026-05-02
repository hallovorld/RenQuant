# PEAD enrichment — Track B

**Date**: 2026-05-02
**Status**: design (not implemented)
**Branch target**: main (orthogonal to sector-arch)

## Current state

`earnings_surprise_cum` is wired as a panel feature column. It is the
trailing-4-quarter cumulative surprise %, ffilled to the daily index.
This captures the *magnitude* of recent surprises but **not the
recency or decay** — a 5% beat 80 days ago looks identical to a 5%
beat 5 days ago.

`earnings_calendar` is loaded by the inference pipeline as a blackout
veto for entry/exit windows (±2 days pre, +5 days post). It is **not**
exposed as a panel feature for the training model.

## What's missing (and why it matters)

PEAD literature (Bernard-Thomas 1989, Chan-Jegadeesh-Lakonishok 1996)
shows that the post-earnings drift:
1. **Strongest in days 1-30** post-announcement
2. **Decays mostly by day 60**
3. **Approximately zero past day 90**
4. **Sign + magnitude scale with surprise quintile** — a top-quintile
   beat drifts further than a small beat

A flat trailing-4-quarter sum doesn't capture (1)-(3), so the model
misses the time-decay structure.

## Three new feature columns

### B-1. `days_since_earnings`
Trading days since most recent announcement, clamped to [0, 90] (past
that, signal is gone). NaN before first announcement in window.

```python
# Pseudo
earnings_dates_ticker = sorted announcement dates from earnings_calendar
for date in panel.index:
    most_recent = max(d for d in earnings_dates_ticker if d <= date)
    days = trading_days_between(most_recent, date)
    return min(90, days) if days >= 0 else NaN
```

### B-2. `pead_decay_weight`
Linear ramp from 1.0 at day 0 to 0.0 at day 60, then 0 past 60.

```python
def pead_decay(days_since: int) -> float:
    if days_since is NaN: return NaN
    if days_since > 60: return 0.0
    return max(0.0, 1.0 - days_since / 60.0)
```

### B-3. `pead_signal` (interaction)
The strongest PEAD feature in literature: most-recent surprise × decay.

```python
pead_signal = most_recent_surprise_pct × pead_decay_weight
```

This single column captures: (a) sign (positive surprise drifts up,
negative down), (b) magnitude (size of surprise), (c) recency (decays
to 0 by day 60+).

Likely the highest-IC of the three new features.

## Why three columns not one

Tree models can use them differently:
- `days_since_earnings` is useful as a regime indicator (blackout periods,
  near-announcement caution).
- `pead_decay_weight` alone (no interaction) lets the tree model
  combine with OTHER features (e.g., volume × pead_decay).
- `pead_signal` is the canonical PEAD alpha; should be the most useful.

XGBoost will pick whichever helps; columns that don't contribute will
get low feature importance.

## Predicted lift

Per literature, PEAD captures ~3-8 bp at quarterly horizons on
fundamental-anchored universes (mid/large cap US equities). At our
fwd_5d horizon with ~100 ticker panel, expect:
- `pead_signal` alone: +2-5 bp CPCV IC
- All three: +3-6 bp (some redundancy)

## Implementation sequence

**B-1**: extend `kernel/earnings_surprise.py` with
`compute_pead_features(surprises, earnings_calendar, ohlcv)` that
returns a dict of three series per ticker, ffilled to daily index.

**B-2**: add the three columns in `pp_panel_training.py::TickerPanelFactorJob`
alongside the existing `earnings_surprise_cum` block. Gated by
`panel_ltr.pead.enabled` (default `false` until validated).

**B-3**: regression tests:
- `test_pead_decay_at_day_0_is_one`
- `test_pead_decay_at_day_60_is_zero`
- `test_pead_signal_sign_matches_surprise`
- `test_no_lookahead_at_announcement_day` — feature value at the
  announcement day must reflect ONLY pre-day data (the surprise
  becomes available after-market).
- `test_days_since_clamps_at_90`

**B-4**: A/A on wl103 baseline.

```bash
python scripts/train_104.py \
  --strategy-config-name strategy_config.pead_off.json \
  --skip-baseline --skip-recalibrate --force
python scripts/train_104.py \
  --strategy-config-name strategy_config.pead_on.json \
  --skip-baseline --skip-recalibrate --force
```

Compare CPCV mean_ic. Acceptance: ≥ baseline + 0.002 (per CLAUDE.md
§2a "rigorously controlled" exception, since only PEAD columns change).

## Open questions

1. **Multi-quarter aggregation**: keep current `earnings_surprise_cum`
   alongside? Or replace? Current is too noisy; replace if `pead_signal`
   ships and beats it.
2. **Pre-announcement run-up**: should we also model the 5-10 day pre-
   announcement period (volume + IV pickup signal)? Probably yes, as
   a follow-up after PEAD validation.
3. **Sector interactions**: PEAD strength varies by sector (tech > consumer staples).
   Could interact with sector indicators if Layer 2 lands. Lower priority.

## Sequencing

Track B is independent of Track A (insider data) and Track C (microstructure).
Can run in parallel as long as compute budget allows. Cheapest of the
high-EV tracks: ~1 day dev, 30 min retrain × 2 for A/A.

If PEAD A/A shows lift > 0.005, ship to main. If 0.000-0.005, keep but
re-evaluate after Track C (microstructure may interact with PEAD).
If < 0, document and shelve in failed-experiments-log.

## Done criteria

- B-1/B-2/B-3 land.
- B-4 A/A measured: pead_on mean_ic ≥ baseline + 0.002.
- Decision: ship to main (default `enabled=true`) OR shelve with
  reproduction recipe.
