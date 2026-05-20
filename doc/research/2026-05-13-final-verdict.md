# 2026-05-13 — Final verdict on regime-conditional GK (16-window panel)


> **📅 Historical snapshot — content below reflects state at the date in filename/header.**
> Verify against current code per CLAUDE.md §1 "code is the source of truth" before acting on
> present-tense claims. For current state see `doc/roadmap.md` § "📍 Current state" +
> `CLAUDE.md` § "🗂 Current state".

After Phase 2 (9 windows 2024-04 → 2026-03) suggested GK_conditional
was approaching Tier 2 (t=+1.46), we extended the OOS panel to 16
non-overlapping 3-month windows spanning 2022-04 → 2026-03 (48 months,
n=990 paired daily observations).

The verdicts FLIP COMPLETELY:

| Candidate | 9-win t | 16-win t | 16-win mean Δ ann | Verdict (16-win) |
|---|---:|---:|---:|---|
| vt15 | +0.69 | +0.80 | +0.48% | NEITHER |
| **GK094** | +0.33 | **−0.60** | **−2.02%** | **TIER 1 REJECT** |
| GK15 | +1.23 | −0.04 | −0.11% | NEITHER |
| **GK_conditional** | **+1.46** | **+0.20** | +0.67% | **NEITHER** |

## Why the reversal — LOW_CALM regime cell emerged

In the 9-window panel (2024-only), LOW_CALM had only n=5 days. In the
16-window panel (2022-2026), it grew to **n=52** and revealed huge
negative deltas for ALL GK variants:

| Regime | n | GK094 Δ ann | GK_conditional Δ ann |
|---|---:|---:|---:|
| HIGH_CALM | 237 | +5.08% | +4.60% |
| LOW_SPIKED | 201 | −3.95% | −2.17% |
| MED_CALM | 119 | −5.21% | −11.29% |
| LOW_NORMAL | 101 | +6.67% | +7.02% |
| HIGH_NORMAL | 85 | −0.51% | +8.83% |
| MED_NORMAL | 68 | −3.95% | +12.20% |
| MED_SPIKED | 68 | +16.21% | +17.74% |
| HIGH_SPIKED | 59 | **−29.18%** | −11.69% |
| **LOW_CALM** | 52 | **−29.56%** | **−27.94%** |

Both GK094 and GK_conditional are devastated in LOW_CALM (sideways
markets with low realized vol — 2022-2023 was full of these). My
GK_conditional config disabled HIGH_SPIKED + MED_CALM but did NOT
anticipate LOW_CALM, which had been invisibly absent from the small
2024-only panel.

HIGH_CALM Δ also shrank significantly (8-win: +17.90% → 16-win: +4.60%).
The apparent regime-conditional win in the 9-window panel was
**overfit to 2024-2026 bull-market HIGH_CALM days**.

## What this proves about methodology

1. **9 windows is statistically inadequate.** Small panels can produce
   apparent t>1.4 effects that vanish or reverse with more data. Per
   §5.13.4, single-measurement OR single-panel claims are forbidden.
2. **New regime cells emerge with longer OOS.** LOW_CALM was n=5 in the
   24-month panel; n=52 in 48-month. A conditional config needs to
   cover all encountered regimes, which requires fully-powered panels
   to discover.
3. **The earlier "regime-conditional GK validates the theory" claim
   was premature.** It validated the theory under the 2024-favorable
   slice. Under full 4-year history, the theory doesn't hold.
4. **The user's frustration last night was correct.** Underpowered
   methodology produces unreliable verdicts that flip with more data.

## Auto-promote decision: NO

Per `doc/research/promotion-methodology.md`:
- GK094 (TIER 1 REJECT) — explicit reject, do NOT deploy
- GK_conditional (NEITHER, t=+0.20) — far from Tier 2, do NOT deploy
- vt15 (NEITHER, t=+0.80) — do NOT deploy
- GK15 (NEITHER, t=−0.04) — do NOT deploy

**Production unchanged.** `strategy_config.golden.json` retains baseline.

## What we keep from this session

🟢 **Infrastructure (real value)**:
- Industry-grade evaluator (statsmodels HAC + arch bootstrap)
- 8 non-overlapping 3-month + extended-to-16 walkforward panel
- SpyRegimeLabelTask (P0-A, off by default)
- ranking.X.regime_overrides config schema (P1, off by default)
- Aux artifact per-as-of regen workflow (correlation + GMM + earnings)
- Extended walkforward 2022-01 → 2026-03 (74 cutoffs trained, manifest merged)

🔴 **Verdicts (all REJECT or NEITHER on 48-mo OOS)**:
- vol-targeting (Moskowitz-Ooi-Pedersen 2012): no observable effect
- Grinold-Kahn α→μ at any IC: harmful at full panel
- Regime-conditional GK: only works in HIGH_CALM at small samples, doesn't survive longer history

🟡 **Open research direction**:
- The HIGH_CALM win pattern is real but small (Δ≈+5%/yr) and dominated
  by other regime losses. A FULL regime-stratified config (disable in
  LOW_CALM + MED_CALM + HIGH_SPIKED + LOW_SPIKED, enable only in
  HIGH_CALM + HIGH_NORMAL + MED_NORMAL + LOW_NORMAL) might be Tier 2.
- But this is parameter-fitting at this point. Better path: structural
  changes (new universe, new model class, new features) per roadmap.

## What the data tells us about the strategy

Across 990 paired daily observations spanning 4 years of OOS, **no
mechanical parameter tweak** (vol-target, Grinold-Kahn, conditional
deployment) materially improves the baseline. The strategy at the
current feature set (alpha158 + 5 fund + 3 PEAD + 3 SUE = 169
features), model class (XGBoost rank:pairwise), and universe (wl103)
is at its **measurable local optimum**.

Further alpha requires structural changes:
1. Universe expansion (wl200/wl500/R1K) — per `doc/roadmap.md` ★ ROI items
2. Model class swap (LightGBM / PatchTST / Transformer)
3. New feature blocks (options-implied vol / FinBERT sentiment /
   alternative data)

These are 1-2 week investments per item, not overnight tweaks.

## Files

- `data/logs/_reports/final_sim_*.json` — analyzer outputs
- `doc/research/evaluation-protocol.md` — methodology spec
- `doc/research/2026-05-12-findings-and-next.md` — prior 8-window findings (now superseded)
- `doc/research/2026-05-13-handsoff-results.md` — overnight 9-window snapshot (now superseded)

## Commits this round

  46e63b2  P0-A SpyRegimeLabelTask
  072be45  P1 ranking.X.regime_overrides
  555f5b1  fix: vol-target + DD-Kelly dead-path
  7bc9b56  feat: Grinold-Kahn α→μ + Option A
  ef52039  feat: regime-stratified analyzer
  55c3c64  doc: 2026-05-13 handsoff snapshot
  <next>   doc: 2026-05-13 FINAL verdict + extended-OOS confirms REJECT
