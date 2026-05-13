# 2026-05-13 — v2 conditional GK with rigorous LOO methodology

## Setup

After 16-window panel showed GK_conditional v1 = NEITHER (t=+0.20), the
hypothesis was that v1 picked the wrong regimes to disable. The proper
test of "regime-conditional GK works" requires **honest cross-validation**:
for each held-out window, build the disable-set from the OTHER 15
windows' regime-Δ profile. No peek-ahead.

Implementation:
1. Compute per-window daily Δ (GK094 vs baseline) on 16 non-overlapping
   3-month windows (n=990 paired daily obs).
2. For each held-out window i:
   - Pool the 15 other windows' Δ by regime
   - Disable any regime with mean Δ < 0 AND n ≥ 8 (power floor)
   - Build per-window strategy config `sim_GK_v2_loo_{Qi}.json`
3. Run 16 sims, one per held-out window, each using its own LOO config.
4. Pooled analysis on 16-window paired Δ.

## Result: TIER 1 REJECT

| Metric | Value |
|---|---:|
| n_days | 990 |
| n_windows | 16 |
| Mean Δ (annualized) | **−2.92%** |
| Newey-West SE | 0.011% (daily, lag=6) |
| t-statistic | **−1.08** |
| p-value | 0.279 |
| 95% bootstrap CI (ann) | [−8.35%, +2.11%] |
| Cohen's d | −0.031 |
| Consistency | 6/16 positive (38%) |
| **VERDICT** | **TIER 1 REJECT** |

**Worse than blanket GK094** (which itself was Tier 1 REJECT at −2.02%/yr).

## Why LOO failed: disable-set instability

The disable-set varies dramatically across LOO folds:

| Held-out | n_disabled | Notes |
|---|---:|---|
| Q01, Q03, Q05-08, Q10, Q12-13, Q15 | 6 | typical |
| Q02, Q04, Q11, Q16 | 5 | LOW_SPIKED stays enabled |
| **Q09** | 5 | MED_NORMAL stays enabled |
| **Q14** | 6 | even **HIGH_CALM** disabled — the "favorable" regime |

The very fact that HIGH_CALM gets disabled in Q14 (when Q14 itself
is excluded from training) tells us Q14 IS where the HIGH_CALM
positive bias came from. Without Q14 in training, HIGH_CALM looks
negative.

This is overfitting in disguise: the disable-set fits noise in the
training fold, then mispredicts which regimes will be positive in the
held-out fold.

## Interpretation

Three possible explanations for why regime-conditional GK fails:

1. **No stable regime-conditional signal exists.** Most likely. Across
   4 years of OOS, GK's Δ in any given regime is dominated by sampling
   noise. A "regime-conditional optimal" exists in any sample but
   doesn't generalize.

2. **Regime labels are too coarse.** Our SPY-derived 3×3 = 9 cells may
   not capture the relevant micro-regimes. Finer labels (e.g. credit
   spread regime, term structure regime) might surface stable signals.

3. **Path dependence pollutes per-regime Δ.** GK-on days affect later
   positions, so a "MED_CALM regime Δ" reflects not just MED_CALM
   behavior but also state inherited from prior HIGH_CALM days. LOO
   doesn't fix this.

For practical purposes the answer is: **don't deploy regime-conditional
GK from the current data**. Higher-power test would need:
- More OOS years (extending to 2020-2026 = 6 years vs current 4)
- Multi-seed sim with OHLCV bootstrap to amplify n
- Finer regime labels (e.g. add VIX percentile)
- Or accept that GK doesn't have signal and abandon this branch

## Decision

**Auto-promote: NO. TIER 1 REJECT.**

Production unchanged. Live broker remains alpaca LIVE on baseline.

Per `feedback_auto_promote_to_prod.md`, Tier 1 REJECT explicitly
forbids promotion. Per `feedback_audit_when_theory_disagrees.md`,
when canonical theory (Grinold-Kahn α→μ, regime-conditional
deployment) fails rigorous test on real data, the appropriate response
is to abandon, not to keep tuning.

## What we learned (positive)

1. **Methodology works.** The LOO + Newey-West + bootstrap framework
   correctly distinguishes overfit signal (v1 t=+1.46 at 9 windows)
   from no-signal (v2 LOO t=−1.08 at 16 windows under proper CV).

2. **The 8/9-window panel was statistically inadequate.** Both v1 and
   the original GK094 panel at 9 windows showed positive bias from
   2024-2026 favorable regime sampling. 16-window OOS + LOO removes
   that bias.

3. **Regime labels via SPY are computable and stable.** SpyRegimeLabelTask
   produces ≥90% agreement with the offline analyzer; the regime-Δ
   structure visible across 16 windows is real (just not exploitable
   for GK).

## Next research direction

Per `doc/roadmap.md` — exit parameter-tweak space, pursue structural
changes:
1. Universe expansion wl200/wl500/R1K (Grinold-Kahn √breadth)
2. Model class swap (LightGBM with category encoding / PatchTST)
3. New feature blocks (options-implied vol / FinBERT / alt-data)

Each is 1-2 weeks per item, but each has theoretical justification
for larger effect than parameter tweaks.

## Files

- `data/logs/sim_2026-05-13_v2_loo/equity/*.json` (16 LOO sim outputs)
- `data/logs/_reports/v2_loo_final.json` (analyzer JSON)
- `backtesting/renquant_104/strategy_config.sim_GK_v2_loo_Q*.json` (16 LOO configs)
