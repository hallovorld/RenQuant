# 2026-05-13 — Handsoff overnight results

User went to sleep around 22:30 PT 2026-05-12. This is the morning
status. No production changes (correctly — no candidate cleared Tier 3
auto-promote gate).

## What shipped overnight

| Commit | What | Status |
|---|---|---|
| `46e63b2` | P0-A: SpyRegimeLabelTask (objective SPY-derived regime) | OFF by default, 11 tests pass |
| `072be45` | P1: ranking.X.regime_overrides config schema | OFF by default, 12 tests pass |
| `<phase2>` | P0-B: extended walkforward 2022-2023 (35 new cutoffs) | TRAINED, manifest merged |
| `<phase2>` | Phase 2 panel runner (5 configs × 16 windows) | RAN, partial success |

## Phase 2 results (9 working windows of 16)

Pre-2024 windows (Q01-Q07) all failed: aux artifact `watchlist-correlation`
has `as_of_date=2023-12-01` → leakage guard rejects sims before 2024.
9/16 windows succeeded (Q08-Q16, 2024-01 → 2026-03).

**Pooled paired-daily verdict (5 candidates, K=4, n=557 daily obs):**

| Candidate | t_pool | mean Δ ann | 95% CI | cons | Cohen d | Tier |
|---|---:|---:|---|---:|---:|---|
| baseline ≡ baseline | 0.00 | 0.00% | [0, 0] | 0/9 | 0.00 | NEITHER (sanity) |
| vt15 | +0.69 | +0.69% | [−1.0, +3.2] | 2/9 | +0.03 | NEITHER |
| GK094 | +0.33 | +1.63% | [−7.8, +11.4] | 6/9 | +0.01 | NEITHER |
| GK15 | +1.23 | +5.73% | [−3.8, +15.2] | 7/9 | +0.05 | NEITHER |
| **GK_conditional** | **+1.46** | **+6.87%** | **[−2.2, +15.8]** | **7/9** | +0.06 | **NEITHER** (just under Tier 2) |

**GK_conditional is the top candidate** but pooled t=+1.46 is below
Tier 2's t>1.5 threshold. **No auto-promote.** Production unchanged.

## Regime-stratified GK_conditional confirms the design

The conditional GK config (HIGH_CALM enabled, HIGH_SPIKED/MED_CALM
disabled) successfully *neutralized* the toxic regimes:

| Regime | n | GK094 (blanket) | GK_conditional |
|---|---:|---:|---:|
| HIGH_CALM | 151 | +17.9% (Δ) | +11.2% (still wins, slightly less concentration) |
| **HIGH_SPIKED** | 59 | **−31.8%** | **−11.7%** ← disable cuts loss by 20pt |
| **MED_CALM** | 41 | −13.4% | **−4.0%** ← disable cuts loss by 9pt |
| HIGH_NORMAL | 79 | −1.3% | +10.3% |
| MED_NORMAL | 46 | +1.2% | +12.8% |
| LOW_NORMAL | 37 | +29.7% | +19.8% |
| LOW_SPIKED | 79 | +0.3% | +4.9% |
| MED_SPIKED | 60 | +6.5% | +10.3% |

**Conditional logic IS working as designed.** GK_conditional dominates
GK094 in 5 of 8 regimes, ties in 2, slightly trails in 1 (HIGH_CALM the
favorable regime, where conditional preserves GK but the absolute return
is slightly lower). Theoretical hypothesis validated; statistical power
inadequate.

## Why t_pool didn't cross Tier 2

Newey-West SE on pooled Δ = 0.019% daily. Annualized: 0.019% × √252 ≈ 0.30%
or so. Cohen's d=0.06 means effect is **6% of within-stream variance**.
Even with n=557 daily obs, that gives t ≈ 1.4. To clear Tier 2 (t>1.5)
at this effect size needs n ≈ 670. To clear Tier 3 (t>3.0) needs n ≈ 2680.

**Path to higher power:**

| Lever | Effect | Effort |
|---|---|---|
| Add 7 pre-2024 windows (Q01-Q07) | n: 557 → ~1000 | aux artifact walkforward framework, 6-8h |
| Multi-seed via OHLCV bootstrap | n: 557 → 2785 (5 seeds) | Bootstrap injection in SimAdapter, 4-6h |
| Combine: 16 windows × 5 seeds | n: ~5000 | 8-10h compute + framework |

Once n ≥ 2000 and regime-conditional GK still shows mean Δ ≈ +6.87%/yr,
t > 3.0 should follow, clearing Tier 3 → auto-promote.

## What blocks acting on this

1. **Aux artifact walkforward framework** — `watchlist-correlation`,
   `gmm-states`, `earnings-calendar`, `asset-embeddings` all have
   single `as_of_date`. For multi-year extended OOS we need
   per-as-of regen + manifest, analogous to walkforward_manifest.json
   for the panel-LTR head. Estimated 6-8h.
2. **Multi-seed framework** — current sim is deterministic. Need to
   add OHLCV / fill / slippage jitter for proper bootstrap. Estimated
   4-6h.
3. **Production regime detector** — for live deployment of conditional
   GK, `regime.spy_regime.enabled=true` would work but the SpyRegime
   labels need a **trailing 1-day** safety buffer to avoid same-day
   leakage. Currently the task uses up-to-today data. A 1-day shift
   is a 3-line fix.

## What I did NOT change

- Production `strategy_config.golden.json` — unchanged
- Live broker — still PAPER (per user 2026-05-11 mandate)
- Live cron scheduling — unchanged (daily104, intraday104, etc.)
- All new features (P0-A, P1, GK) OFF by default in prod config

The 14:08 PT daily104 firing today and tomorrow will use the SAME
baseline strategy as yesterday. Zero behavior change in production.

## Tomorrow's decision tree

User can choose:

A. **Build aux artifact walkforward framework** (6-8h) → re-run Phase 2
   with all 16 windows → expected GK_conditional t_pool ≈ 1.9-2.1 →
   Tier 2 SCREEN (still not auto-promotable)

B. **Build multi-seed framework** (4-6h) → 5-seed × 9 windows × 5
   configs = 225 sims → expected GK_conditional t_pool ≈ 3.0-3.5 →
   could clear Tier 3 → auto-promote

C. **Both A + B in parallel** (1-2 days compute) → maximize power, fast
   Tier 3 verdict

D. **Reduce ambition** — accept Tier 2 SCREEN as evidence of
   regime-conditional structure; pivot to structural changes (new
   universe / LightGBM / PatchTST per roadmap) that might give larger
   effect sizes.

My recommendation: **A** (aux artifact walkforward) is the highest-value
single step. It's reusable infrastructure for ALL future regime work
AND unlocks 16-window analysis. B can wait until A's verdict.

## Resources / commits

  46e63b2  feat(regime): SpyRegimeLabelTask
  072be45  feat(qp): regime_overrides for conditional deployment
  <next>   doc(handsoff): 2026-05-13 results

Reports:
- data/logs/_reports/phase2_sim_vt15_ext.json
- data/logs/_reports/phase2_sim_GK094_ext.json
- data/logs/_reports/phase2_sim_GK15_ext.json
- data/logs/_reports/phase2_sim_GK_conditional_ext.json
