# 2026-06-06 — Scoping: fast-moving data for BULL_CALM timing alpha

**Premise (from the placebo diagnosis,
[`../2026-06-06-bull-calm-specialist-placebo-diagnosis/`](../2026-06-06-bull-calm-specialist-placebo-diagnosis/)):**
the current 172-feature set (alpha158 windows + fundamentals + PEAD/SUE) yields
**no reliable, placebo-robust BULL_CALM timing alpha**. Its informative content
in BULL_CALM is slow factor exposure that is (a) largely priced and (b)
non-stationary. To get timing IC that survives a time-shift placebo, the model
needs **information that changes faster than the 60-day horizon** — signals the
current panel structurally cannot contain.

This doc scopes candidate fast data sources, ranks them by expected
signal-per-cost, and defines the integration + validation contract. **No data
is ingested here** — this is the design to critique first (§7.10 / §6.3).

## Why "fast" is the right axis

A 60-day-window feature is ~invariant over a 60-day placebo shift, so any IC it
carries is indistinguishable from persistence. Only features whose
cross-sectional ranking **turns over inside the prediction horizon** can produce
IC that the time-shift placebo cannot replicate. So the selection filter for a
new source is: *does its cross-sectional signal decorrelate from itself within
~20 trading days?* If not, it will just add more persistence.

## Candidate sources (ranked by expected signal/cost)

| Rank | Source | Why it could carry BULL_CALM timing | Turnover | Cost / friction | Leakage risk |
|---|---|---|---|---|---|
| 1 | **News-sentiment velocity** (Δ sentiment, not level) | We already ingest news sentiment (`sentiment_pos_share`, `mean_sentiment`, `n_articles_log`); the *level* is slow but the **day-over-day change / surprise** is fast and currently unused as a ranking feature. Cheapest because the raw data is already on disk. | High | **Low — data already ingested**; needs a velocity/surprise transform + point-in-time discipline | Medium — must use only as-of-close articles; no revision backfill |
| 2 | **Short-interest deltas** | Bi-monthly short-interest *changes* + days-to-cover proxy fast crowding/squeeze signal; known cross-sectional predictor at weeks horizon | Medium | Low-med — FINRA/exchange short-interest feed; semi-monthly cadence | Low — official release dates are point-in-time |
| 3 | **Options-implied signals** (put/call skew, IV term-structure slope, Δ open-interest) | Fast, forward-looking positioning; skew shifts lead cross-sectional returns at days–weeks | High | **High — needs an options data vendor**; nontrivial cleaning | Medium — must align to the option's as-of date |
| 4 | **Analyst-revision momentum** (EPS estimate Δ, up/down-grade flow) | Revision *velocity* is a classic PEAD-adjacent fast signal; complements existing SUE/PEAD (which are event-stamped, not velocity) | Medium | Med — estimates feed | Low-med — revision timestamps are clean |
| 5 | **Intraday microstructure** (overnight gap, close-to-open drift, realized-vol-of-vol) | Fastest, but mostly a risk/vol signal; weak directional cross-sectional alpha at 60d | Very high | Med — we have OHLCV; intraday needs minute bars | Low |

## Recommended first cut — source #1 (news-sentiment velocity)

Highest signal/cost because **the raw data is already ingested** (the placebo
runs even warn about `sentiment_pos_share / mean_sentiment / n_articles_log`
coverage). The gap is that only *levels* exist; the fast signal is the change.

MVP feature set (all point-in-time, as-of close):
- `sentiment_velocity_5d` = mean_sentiment(t) − mean_sentiment(t−5)
- `sentiment_surprise` = (mean_sentiment(t) − trailing-20d mean) / trailing-20d std
- `news_intensity_accel` = Δ n_articles_log over 5d (attention spike)
- `pos_share_velocity_5d` = Δ sentiment_pos_share over 5d

## Validation contract (non-negotiable — §7.2 / R1–R5)

A new source earns a place ONLY if, on BULL_CALM OOS rows:
1. **Placebo-robust:** aligned-60 real IC − 60d model-placebo IC > 0 with the
   placebo ≈ 0 (the test the specialist FAILED in H1). Run through the existing
   `analyze_manifest_sanity_placebo.py`.
2. **Stationary:** positive net IC in BOTH halves of the OOS window (the
   two-window test the specialist failed) — not one-window luck.
3. **Triad clean:** shuffle + time-shift placebo ≈ 0 (R2).
4. **Incremental:** IC *after* orthogonalizing against the existing 172 features
   (must add signal the panel doesn't already have, else it is redundant).
5. **Leakage-audited at write time:** every feature at `t` uses only data
   available by `t`'s close; revision/backfill explicitly excluded (§7.2 R-checks).

Only a source clearing all five is wired into a BULL_CALM specialist. Anything
that passes pooled but fails (2) is the same trap we just diagnosed.

## Build order (range-finding first, §7.11)

1. Engineer the 4 sentiment-velocity MVP features from existing on-disk news
   data (no new ingestion). ~hours, zero data cost.
2. Run the 5-gate validation on BULL_CALM OOS. **Decision point:** if even the
   already-paid-for sentiment-velocity source fails gates (1)+(2), that is
   strong evidence BULL_CALM timing alpha is genuinely scarce at this universe
   /horizon, and we should stop spending on new vendors and instead accept
   BULL_CALM as a low-conviction regime (size down, lean on BEAR/CHOPPY where
   signal exists).
3. Only if #1 clears gates do we pay for sources #2–#4 (short-interest,
   options, revisions), cheapest-first.

**Do NOT** start with a paid options/short-interest vendor — the free
sentiment-velocity experiment is the range-finder that tells us whether fast
data helps at all before any spend.

Agent-Origin: Claude
