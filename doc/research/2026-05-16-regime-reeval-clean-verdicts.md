# 2026-05-16 regime-reeval: clean knob-only verdicts

## TL;DR

After fixing the 2026-05-15 no-op build script and re-running 6 panels with
correct config paths, the clean knob-only A/B (subtracting the
2026-05-15-EVENING calibrator-refit effect via a proxy baseline) is:

| Panel | Verdict | Pooled Δ vs proxy baseline | Conditional wins |
|---|---|---|---|
| `re_stop007`       (stop_loss → 0.07 all regimes)      | WIN-CONDITIONAL | -0.23pp | BEAR, BULL_STRONG |
| `re_sdl_n2`        (σ-aware SDL n_sigma → 2.0)          | WIN-CONDITIONAL | -1.24pp | BEAR, CHOPPY |
| `re_trail015`      (trailing trigger → 0.15 all regimes) | NEITHER        | +0.07pp | BULL_VOLATILE (narrow) |
| `re_cvar025`       (qp_cvar_lambda → 0.25)              | WIN-CONDITIONAL | -0.57pp | BEAR, CHOPPY |
| `re_cvar050`       (qp_cvar_lambda → 0.50)              | NEITHER        | -0.15pp | BEAR (narrow) |
| `re_kelly_t1_035`  (tiered_thresholds[0].min_score 0.35)| NO-OP          | n/a (used as proxy baseline) | none — kelly is tier-agnostic |

Three knob candidates with conditional-win regimes BEAR/CHOPPY emerge for
further investigation. All small magnitude; per-regime n is small (1–8
windows depending on regime); no Tier-3 promotions earned.

## Why the proxy baseline

The 5/14 dedicated baseline (`sim_2026-05-14_baseline_hmm`) used the
pre-refit calibrator. The calibrator was refit on 2026-05-15 EVENING (per
CLAUDE.md §"calibrator P0 closed end-to-end"). The 5/16 panels run against
the new calibrator. So any A/B vs 5/14 baseline is contaminated by the
calibrator refit, not just the knob change.

Spot evidence: `re_kelly_t1_035`'s pooled "WIN +1.14pp" vs 5/14 baseline
disappeared entirely when re-analyzed vs a same-batch proxy baseline. That
+1.14pp was 100% calibrator effect. Kelly itself is tier-agnostic — there
is no tier-1 score threshold to raise.

`scripts/preflight_analyzer.sh` (shipped this session) compares baseline
equity mtimes vs current artifact mtimes and BLOCKS the analyzer if any
artifact is newer than the baseline. The 5/14 baseline now blocks; the
fallback is the proxy baseline pattern: use any in-batch panel whose knob
is a known no-op as the analyzer baseline.

## Why kelly_t1_035 is a no-op (and a methodology lesson)

The original hypothesis was "raise Kelly tier-1 entry score from 0.27 to
0.35". This is **invalid by construction**: kernel's Kelly implementation
(`kernel/kelly.py:kelly_target_pct()`) is tier-agnostic — it reads four
knobs from `ranking.kelly_sizing` (`fractional`, `max_concentration`,
`min_edge`, regime-resolved `max_pct`) and applies a single flat
fractional multiplier to all admitted candidates. There is no tier.

I mapped the hypothesis to `tiered_thresholds[0].min_model_score = 0.35`
because the name was closest. The kernel DOES read that path, but it
controls **selection admission**, not Kelly sizing — and in 14/14 panel
windows no candidate scored in the [0.27, 0.35) band where raising the
admission threshold would change anything.

Static path validator (kernel-reachable check) said ACTIVE. Smoke (1-month
empirical sim) would have caught this. The
build script now calls `scripts/preflight_panel.sh` (static + optional
smoke) instead of bare static.

## Action items

1. **Don't promote any of these to production.** All five real-knob panels
   show small-magnitude pooled differences (<2pp) with conditional wins
   confined to BEAR and CHOPPY regimes. Not Tier-3.
2. **`re_sdl_n2` BEAR/CHOPPY wins (-1.24pp pooled) are the strongest
   conditional signal.** Worth a follow-up that overlays the knob only in
   BEAR and CHOPPY (per PRIME DIRECTIVE) and re-runs as a 16-window panel
   against a fresh same-day baseline.
3. **Kelly tier-1 hypothesis is dead.** Don't queue any more "raise tier-1
   threshold" experiments — the variable doesn't exist in this kernel.
4. **Adopt `scripts/preflight_panel.sh` + `preflight_analyzer.sh` for all
   future sim work.** Build scripts and queue runners that don't gate on
   these will keep producing contaminated results.

## Files referenced

| Purpose | File |
|---|---|
| Static path validator | `scripts/validate_sim_config_active.py` |
| Panel preflight (static + smoke wrapper) | `scripts/preflight_panel.sh` |
| Analyzer baseline freshness check | `scripts/preflight_analyzer.sh` |
| Build script (auto-gates on preflight) | `scripts/build_regime_reeval_configs.py` |
| Parallel panel runner | `scripts/run_validated_reeval_parallel.sh` |
| Clean knob-only verdicts (JSON) | `data/logs/reeval_results/*_vs_kelly_proxy.{json,txt}` |
| Contaminated verdicts (kept for postmortem) | `data/logs/reeval_results/*_validated.{json,txt}` |
| Failed-experiment log entry | `doc/research/failed-experiments-log.md` 2026-05-16 |
