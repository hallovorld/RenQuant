# 2026-06-06 — Track C (BULL_CALM specialist) verdict: FIRST POSITIVE, placebo-clean

**Status**: §7.4 **Tier-2 SCREEN** (promising, keep researching — NOT yet
live-promotable). A BULL_CALM-only specialist model lifts BULL_CALM IC from
**+0.0035 (pooled, placebo-noise) → +0.0241 (real, placebo-clean)** — the first
genuine signal-side win in the campaign after 5 negatives. Lands just short of
the +0.030 Tier-1 bar.
**Owner**: Claude. **Reviewer requested**: codex (caveats in §5).

---

## 1 · What was tested

The 4 prior negatives (Kelly σ-horizon, cash overlay, QP allocator, Track B
momentum-on-pooled) all bottomed out on the same fact: the **pooled** panel-LTR
model has near-zero BULL_CALM signal. Track B confirmed the mechanism — momentum
works in BULL_CALM but reverses in BULL_VOLATILE, so a single pooled model
learns one global loading that dilutes to noise.

**Track C thesis** (§1 PRIME DIRECTIVE): train a model on **BULL_CALM rows
only** so it can express the regime-specific loading with no cross-regime
dilution. Infra shipped in #119/#121 (per-regime training CLI +
`RegimeEnsemblePanelScorer`) but had **never been trained or evaluated** — this
is the first evaluation.

**Procedure**: `train_per_regime_walkforward.py --regimes BULL_CALM
--cadence-days 21`, 34 cuts 2024-01-01 → 2025-11-24, base 172-feature recipe (NO
momentum add-ons — pure specialist). Per-regime IC + shift-60 placebo via
`analyze_manifest_sanity_placebo.py`, leak-free WF scoring. Baseline = the
pooled model on a manifest **truncated to the identical 34-cut range** (matched
val window AND model-freshness), run through the **same code**.

## 2 · The data — all under identical code, 34 cuts, n≈400, val 2024-02→2026-02

| BULL_CALM | mean_ic | hit-rate | shift-60 placebo ratio | real signal? |
|---|--:|--:|--:|---|
| Track B (pooled + momentum) | −0.0049 | — | 6.5 | ✗ noise |
| **pooled baseline (matched)** | **+0.0035** | 48.5% | **1.11** | ✗ **noise** |
| **BULL_CALM specialist** | **+0.0241** | 52.8% | **0.47** | ✓ **real** |
| Tier-1 promotion bar | ≥ +0.030 | — | < 0.5 | — |

Other regimes (specialist): BEAR +0.295 (n=50, still strong), CHOPPY +0.0247
(n=39), BULL_VOLATILE −0.0369 (n=19, momentum reverses — expected). Pooled
real_ic rises +0.039 → **+0.049**.

## 3 · The placebo is the headline (§7.2.1 R2)

Shift-60 placebo (label shifted by the full forward-label horizon; if the
"signal" survives, it's not signal):

| | aligned_real_ic | model_placebo_ic | ratio | verdict |
|---|--:|--:|--:|---|
| pooled baseline | +0.0229 | +0.0254 | **1.11** | placebo ≥ real → **NOISE** |
| **BULL_CALM specialist** | **+0.0382** | +0.0178 | **0.47** | placebo < ½ real → **REAL** |

**This is the load-bearing result.** It is not merely that the specialist has a
higher IC — it is that **the pooled model's BULL_CALM IC is itself placebo-noise
(ratio 1.11), while the specialist's is a genuine signal (ratio 0.47).** The
pooled model literally cannot express BULL_CALM signal; the specialist can. That
is the regime-conditional thesis (§1) validated empirically, not asserted.
`promotion_evidence = True` for the specialist (real ≥ 0.005, placebo < ½ real).

## 4 · Verdict — Tier-2 SCREEN (keep going), not Tier-3 (promote)

`§7.4`: specialist ΔIC = +0.0206 over the matched baseline, placebo-clean →
**not a REJECT, a positive SCREEN**. But +0.0241 < +0.030, so it does **not**
clear the Tier-1 promotion bar. No live config flip. The specialist is wired in
the runtime (`ranking.panel_scoring.specialists`, #121) but stays off until it
clears the bar.

**Next step (well-motivated):** momentum features on the SPECIALIST (Track B ⊕
Track C). Track B failed only because momentum was added to the *pooled* model
where the VOLATILE reversal cancels it; on a BULL_CALM-only specialist there is
no such conflict, so the +0.039 naive-momentum IC the diagnostic found
([`2026-06-05-bull-calm-...-momentum-diagnostic.md`](../../2026-06-05-bull-calm-signal-is-catchable-momentum-diagnostic.md))
should actually land. That is the candidate to push +0.0241 over +0.030.

## 5 · Caveats — please attack (codex)

1. **Comparison parity.** Specialist and matched baseline are both 34 cuts, same
   val window (2024-02-01 → 2026-02-10, n=400), same feature recipe (172,
   verified identical `feature_cols`), and run through the **same patched code**
   (the coverage-gap drop in this PR applies to both). The pooled baseline's
   own placebo (ratio 1.11) is the control that makes the specialist's 0.47
   interpretable.
2. **Coverage-gap drop.** Both evals drop the same 109/715,629 rows (0.02%, the
   rawlabel's 2026-02-11 tail not yet in the training panel) — see the
   `_load_sanity_panel` change in this PR (bounded at 1%; a larger gap still
   raises). Immaterial to a 400-date BULL_CALM IC.
3. **+0.0241 is below +0.030.** This is a SCREEN, not a promotion. The
   regime-conditional win is architectural (pooled-noise → specialist-real); the
   *magnitude* still needs the momentum synthesis to clear the live bar.
4. **No multi-seed** (§7.3). Single WF retrain per arm; the placebo is the
   load-bearing control. A 5-seed repeat would tighten it.
5. **shift-60 = 1× horizon.** Per the #226 review note, a full-strength R2 wants
   shift-120; shift-60 already separates real (0.47) from the pooled noise
   (1.11), but a shift-120 confirm is the rigorous follow-up.

## 6 · Artifacts

- Specialist: `backtesting/renquant_104/artifacts/diagnostics/sanity_placebo_specialist_bull_calm_20260606/panel-ltr.json`
- Matched baseline (same code): `.../diagnostics/sanity_placebo_baseline_trunc_patched_20260606/panel-ltr.json`
- Specialist WF manifest: `.../artifacts/sim/walkforward_manifest_v2_20260606_per_regime_bull_calm.json` (34 cuts, 34 ok / 0 failed)
- Specialist models: `.../artifacts/walkforward_bull_calm_specialist/bull_calm/<cut>/panel-ltr.json`
