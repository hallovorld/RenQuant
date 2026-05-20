# 2026-05-14 evening session — final hand-off


> **📅 Historical snapshot — content below reflects state at the date in filename/header.**
> Verify against current code per CLAUDE.md §1 "code is the source of truth" before acting on
> present-tense claims. For current state see `doc/roadmap.md` § "📍 Current state" +
> `CLAUDE.md` § "🗂 Current state".

## TL;DR

Tonight's autonomous run **exhausted the long-short / leverage parameter space**.
**Zero promotes from 12+ panels run this session.** The strategy sits at a
local optimum within scaling-knob space. Pure shorts add small regime-
conditional alpha; pure leverage adds essentially nothing; the v7
+13.69pt win was an **interaction** of shorts × leverage that does not
survive multiple-comparison correction.

**Production stays unchanged.** Shorts feature is built and validated but
remains `enabled=false`.

## Numbers (vs paired 16-window baseline)

| Config | Mean Δ_APY | DSR (psr) | Catastrophes | Verdict |
|---|---|---|---|---|
| longshort_v1 v7 (shorts + gross_max=1.30) | +13.69pt | 0.95+ | (not measured) | Tier 3 BUT leverage-confounded |
| **longshort_clean (shorts + gross_max=1.00)** | **+6.23pt** | 0.90 | Q07 −20.5pt, Q11 −26.1pt | **NEITHER** |
| **leverage_only (NO shorts + gross_max=1.30, partial n=10)** | **+0.86pt** | n/a | 6/10 identical to baseline | **REJECT** |

Decomposition: v7 +13.69pt ≈ pure_shorts (+6.23pt) + pure_leverage (+0.86pt)
+ interaction (+6.60pt). The interaction term is unstable.

## What worked tonight

1. **Vectorbt validator framework**: 8 cross-checks (4 short + 4 long) pin
   SimAdapter cash-flow math against mature library. Future cash-flow
   changes must pass this.
2. **5-test methodology**: Pooled mean + Wilcoxon + regime-stratified +
   DSR + no-catastrophe correctly rejected the leverage-confounded v7
   "win" that single-metric pooling would have approved.
3. **Knob path audit discipline**: Pre-flight grep + config-diff caught
   the `gross_max=None when shorts disabled` issue BEFORE launching an
   80-sim sweep. Saved ~3 hours.

## What didn't work tonight

1. Single-knob parameter optimization (10 panels earlier today: all
   NEITHER) confirmed exhausted.
2. Regime-conditional shorts blocked on regime detector (labels 95% of
   days BULL_CALM regardless of actual market state).
3. Pure leverage path (`leverage_only`) couldn't be completed — Q11-Q16
   sub-processes died during parallel panel contention. 10/16 are enough
   to conclude "small effect" but not enough for Tier statistical bar.

## Strategic outlook

Strategy is at a local optimum in scaling-knob space. **Further alpha
requires STRUCTURAL change**, not parameter tuning. Candidates ordered
by tractability:

1. **Fix the regime detector** (multi-hour code task; unblocks options 2-5).
   - Current: 95% BULL_CALM regardless of actual market state.
   - Without this, every regime-conditional strategy is theatrical.
2. **Tighter shorts + BULL_STRONG carve-out** (after regime fix).
   - max_short_pct=0.03 (from 0.05), disable in BULL_STRONG.
   - Estimated upside: +3-5pt with much lower catastrophe risk.
3. **Universe expansion** (R1K / wl183) — deferred from earlier session,
   requires multi-hour panel retrain.
4. **New feature engineering** (alt-data PEAD/SUE freshness; intraday
   features for daily model; etc.).
5. **New model class** (LightGBM / Transformer panel ranker — significant
   compute investment).

## Pending user decisions

| # | Question | If yes |
|---|---|---|
| 1 | Accept "no promote, keep shorts dormant" verdict for today? | Lock current prod config (already done) |
| 2 | Begin regime detector fix as next session's structural work? | Multi-hour task; pinned as P0 for next session |
| 3 | Roll forward to universe expansion (R1K) instead? | 3-6 hr retrain compute; lower probability of structural win |
| 4 | Defer experiments and pivot to live-trading reconciliation work? | Slice 3b/4b/5 from earlier roadmap |

## Production state at session close (2026-05-14 19:30 PT)

- `strategy_config.golden.json` unchanged from morning baseline
- `long_short.enabled = false` (default)
- 8 vectorbt cross-validators pinned in tests/
- All 11 launchd plists active in PAPER mode
- 130 commits ahead of origin/main (unchanged from morning)
- All experiment artifacts in `data/logs/sim_2026-05-14_*` for replay

## Memory entries written tonight

- `feedback_validate_with_mature_lib.md` — vectorbt cross-check before scaling
- `project_longshort_clean_verdict_2026-05-14.md` — NEITHER + 2 catastrophes
- (Pre-existing) `feedback_qp_gross_max_is_leverage.md` — knob disentanglement
- (Pre-existing) `feedback_eval_robust_methodology.md` — 5-test framework
- (Pre-existing) `project_strategy_local_optimum_2026-05-14.md` — local-optimum diagnosis

## Files committed this session

```
548c76e test: vectorbt cross-validators for SimAdapter cash flow paths
af9a966 fix(short): §1233(e) wash-sale exposure on short-cover loss
... + 7 prior shorts-build commits this session
```
