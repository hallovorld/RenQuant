# Triple-barrier label — design (Track F)

**Branch**: `exp/wl500-and-sector-arch`
**Date**: 2026-05-02
**Status**: design (not implemented)
**Reference**: Lopez de Prado, *Advances in Financial Machine Learning*, ch. 3

## Why

Current production label: `fwd_5d_return` — the realised 5-trading-day
forward return per ticker-date. The XGBoost rank-pairwise model learns
to order ticker-dates by this fixed-horizon outcome.

**Failure modes of fixed-horizon labels**:

1. **Asymmetric within-window volatility ignored**. A ticker that gains
   3% on day 1, then drifts back to +1% by day 5 is labelled identical
   to a ticker that monotonically grinds up to +1% over 5 days. The
   first signal had a clear directional move; the second is noise. Rank
   labels can't distinguish.

2. **Late-day reversal risk**. A ticker that's up 5% by day 4 then
   gives back 4% on a CPI surprise on day 5 ends up labelled +1%.
   Same outcome as a 5-day grind to +1% — but the model in production
   would have sold on day 4 (intraday stop or news flow), not day 5.
   Labels diverge from realisable outcomes.

3. **Volatility regime blindness**. fwd_5d in a low-vol week (σ ≈ 0.5%)
   vs a high-vol week (σ ≈ 3%) gets the same label scale. The model
   over-weights "big" moves in calm periods that a vol-aware policy
   would normalise.

## Triple-barrier idea

For each ticker-date, define three barriers:

```
upper barrier (profit-take):   p_t × (1 + α × σ_t)
lower barrier (stop-loss):     p_t × (1 − β × σ_t)
time barrier (max horizon):    t + max_horizon_days
```

where `σ_t` = trailing 20-day daily-return volatility, `α`, `β` are
multipliers (typical: α = β = 2.0), `max_horizon_days` ≈ 5-10.

Walk the price path forward; record the FIRST barrier hit. The label
becomes:

| First-hit barrier | Label (rank model) | Sample weight |
|---|---|---|
| Upper (profit) | `+r_realised` (positive) | `1.0` |
| Lower (stop) | `−r_realised` (negative) | `1.0` |
| Time (timeout) | `r_at_time` (any sign) | `0.5` (lower confidence) |

Sample weight discounts timeouts because they carry less information
about the directional thesis (the model "got it wrong" or "it didn't
play out yet").

## Predicted benefits

- **5-10 bp IC lift** on top of current panel (rough estimate from
  AFML examples + typical mid-cap pattern; not a measured number).
- **Cleaner Sharpe ratio**: rejected by the "stop" barrier translates
  to a lower realised loss than fwd_5d that lets the loss run.
- **Volatility-aware**: σ_t scaling makes 1bp move in a calm week
  count as much as 3bp in a vol week — policy attention re-weighted
  to risk-adjusted moves.

## Predicted risks

- **Label distribution shift**. Most labels may become "timeout" if
  α/β are too wide. Acceptable proportion: 20-40% timeouts. If 60%+,
  barriers are too wide; if 5-10%, too tight.
- **Loss of homogeneous treatment**. Tickers with high σ get wider
  barriers; tickers with low σ get tighter. Could over-fit to high-σ
  tickers (more labelled events).
- **Computational cost**. Labels now require a forward walk per
  ticker-date instead of a vectorised lookup. ~10x slower (still
  bounded — 1-2 minutes added to pipeline).

## Falsification design

A/A control:
- Same panel, same features, same model, same CPCV folds.
- Two label variants: fwd_5d (current) vs triple_barrier (new).
- Run both, compare CPCV mean_ic.

**Acceptance criterion**: triple_barrier mean_ic ≥ baseline + 0.005
(5bp lift). Lower than the +0.020 promotion floor because:
- Variables are rigorously controlled (only the label changes).
- Theory anchor is strong (AFML §3).
- A small lift here is real, not noise.

Per CLAUDE.md §2a, this qualifies for the "rigorously-controlled
variables" exception.

## Implementation sequence

**F-1**: write `panel_pipeline/triple_barrier_labels.py` — pure function
`compute_triple_barrier_labels(ohlcv, alpha, beta, max_horizon)`. Returns
DataFrame indexed by (date, ticker) with columns `(label, weight, hit_type)`.

**F-2**: add config flag `panel_ltr.label_mode` ∈ {`fwd_5d`, `triple_barrier`}.
Default `fwd_5d` (production unchanged). Wire into PanelLabelTask.

**F-3**: regression tests:
- `test_triple_barrier_alpha_beta_symmetric`: α=β=2 + symmetric vol
  → expected 50/50 upper/lower hits.
- `test_triple_barrier_volatility_scaling`: high-vol ticker gets wider
  barriers (event count test).
- `test_triple_barrier_no_lookahead`: shifting the label by +1 day and
  retraining should produce IC ≈ 0 (catches sneaky lookahead).

**F-4**: A/A on wl103 baseline. CPCV mean_ic comparison.

**F-5**: if F-4 lifts, also test on wl178 (sector arch) — does triple-
barrier interact with sector heterogeneity? Could amplify or wash out
the L1+L2 effect.

## Open questions

1. **Per-ticker σ_t vs cross-sectional σ_t**: per-ticker σ is more
   accurate but adds heteroscedasticity to the label space (labels
   not comparable across tickers). Cross-sectional σ would normalise
   labels but lose the per-ticker risk profile. AFML uses per-ticker;
   we may want cross-sectional for our rank-pairwise objective. Test
   both.

2. **α and β asymmetric?** If we know the model is going long-only,
   maybe β (stop) should be tighter than α (profit-take) to incentivise
   asymmetric R/R. But that's a policy question, not a label question.
   For now, symmetric α=β.

3. **Compatibility with NGBoost μ/σ head?** NGBoost models fit a
   probability distribution to the label. If labels are sparse-event
   triple-barrier outcomes, the distributional fit may be worse.
   Test NGBoost separately.

## Sequencing relative to other tracks

Track F is independent of Track A (insider) / Track B (PEAD) / Track C
(microstructure) — no feature dependency. **But run AFTER those land**:
the IC lift from triple-barrier should be measured on the richest
feature set, not a stripped-down one. Otherwise the +5bp could be lost
in noise from missing features.

Order:
1. Track A (insider data) — running now
2. Re-baseline wl103 with insider feature
3. Track B (PEAD enrichment) — likely small lift, ~2-5 bp
4. Track C (microstructure) — bigger lift, ~10-30 bp, longer dev
5. Track F (triple-barrier) — ~5-10 bp, runs on richest panel
6. Track D (watchlist curation) — runs on richest panel + best label

## Done criteria

- F-1/F-2/F-3 land + regression tests green.
- F-4 A/A measured: triple_barrier mean_ic ≥ baseline + 0.005.
- Decision: keep as default OR shelve as "dependency on full features
  not yet realised; revisit after Track C".
