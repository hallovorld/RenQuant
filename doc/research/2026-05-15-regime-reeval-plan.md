# 2026-05-15 — Regime-conditional re-evaluation plan

## Motivation

Per CLAUDE.md PRIME DIRECTIVE: RenQuant is REGIME-CONDITIONAL. Pooled-mean
evaluation across regimes is BIASED. The canonical proof is the
2026-05-14 long-short clean test:

* **Pooled verdict**: +6.23pt mean, p=0.234 → NEITHER
* **Regime-stratified**: BEAR +22pt / CHOPPY +14pt / BULL_VOL +13pt (3 wins)
  vs BULL_CALM -7.8pt / BULL_STRONG -1.8pt (2 losses)
* **Right answer**: deploy conditional on regime ∈ {BEAR, CHOPPY, BULL_VOL}

The same pooling bias **almost certainly** hides actionable signal in
earlier rejections. This document plans the re-evaluation.

## Top-6 re-eval candidates (sorted by regime-conditional likelihood)

| # | Candidate | Pooled verdict | RC likelihood | Expected regime split |
|---|---|---|---|---|
| 1 | **NGBoost σ-aware Kelly** | -20.6pt APY / -1.09 Sharpe | HIGH | BEAR/VOL win (σ helps), BULL_CALM lose (over-tight) |
| 2 | **stop-loss 0.07** | -7.5pt | HIGH | BEAR/CHOPPY +2~5pt, BULL_CALM -12~-15pt |
| 3 | **σ-aware SDL n_sigma=2.0** | -10.4pt | HIGH | BEAR/VOL +3~5pt, BULL_CALM -12~-15pt |
| 4 | **CVaR λ=0.25** | -3.3pt / +0.04 Sharpe | HIGH | BEAR +1~3pt, BULL_CALM -4~-6pt |
| 5 | **trailing-stop trigger 0.15** | -7.2pt | MED | CHOPPY/REVERT win, BULL_STRONG lose |
| 6 | **Kelly tier-1 raise 0.27→0.35** | -8.88pt APY / +0.74 Sharpe | MED | BEAR/CHOPPY +2~4pt, BULL_CALM -10~-12pt |

## Status of each (2026-05-15 evening)

| # | Config | Sim plan | Status |
|---|---|---|---|
| 1 | NGBoost retrain | 5-seed Duan 2020 §4 validator | 🟢 RUNNING — `train_ngboost_proper.py` seed 42 fitting |
| 2 | `sim_re_stop007.json` (+ _pre2024) | 16-window panel | 📅 QUEUED |
| 3 | `sim_re_sdl_n2.json` (+ _pre2024) | 16-window panel | 📅 QUEUED |
| 4 | `sim_re_cvar025.json` (+ _pre2024) | 16-window panel | 📅 QUEUED |
| 4b | `sim_re_cvar050.json` (+ _pre2024) | 16-window panel | 📅 QUEUED |
| 5 | `sim_re_trail015.json` (+ _pre2024) | 16-window panel | 📅 QUEUED |
| 6 | `sim_re_kelly_t1_035.json` (+ _pre2024) | 16-window panel | 📅 QUEUED |

Currently running (consuming 7+ cores):
* `train_ngboost_proper.py` — 5-seed, ETA 2-3h
* `sim_p0activated` 16-window — 6 workers, ETA ~60min

When free: queue the 6 re-eval panels (1 at a time × 6 workers wide = ~60min each = ~6h total),
or 2 at a time × 3 workers = same total wallclock with less per-job memory pressure.

## Experimental design (per CLAUDE.md §5.13.4a + §5.14)

### Per-candidate run plan

For each candidate `C`:

```bash
/tmp/run_p0_panel.sh "re_${C}"      # 16 windows × 1 config = 16 sims
```

Outputs to `data/logs/sim_2026-05-15_re_${C}/equity/Q{01..16}.json`.

### Analysis (NEW — `scripts/analyze_regime_stratified.py`)

For each panel, compute:

1. **Pooled mean Δ vs baseline** (existing 5-test methodology)
2. **Per-regime stratified Δ**:
   * Group windows by DOMINANT regime (>50% of bars in that regime)
   * If no window has >50% in any single regime, group by REGIME PROPORTION
   * Compute mean Δ + Wilcoxon per group
3. **Regime-conditional verdict**:
   * WIN-CONDITIONAL: ≥1 regime with mean Δ > 0 AND Wilcoxon p < 0.10 AND no catastrophe (worst window > -10pt) → promote regime-conditionally
   * NEITHER: no regime hits the bar
   * REJECT: every regime has mean Δ < 0

### Regime mapping per OOS window

| Window | Period | Likely dominant regime |
|---|---|---|
| Q01 | 2022-04-01 → 2022-07-01 | BEAR (rates spike) |
| Q02 | 2022-07-01 → 2022-10-01 | BEAR / CHOPPY |
| Q03 | 2022-10-01 → 2023-01-01 | CHOPPY / BULL_VOL |
| Q04 | 2023-01-01 → 2023-04-01 | BULL_CALM / BULL_VOL |
| Q05 | 2023-04-01 → 2023-07-01 | BULL_CALM |
| Q06 | 2023-07-01 → 2023-10-01 | CHOPPY |
| Q07 | 2023-10-01 → 2024-01-01 | BULL_STRONG (rally) |
| Q08 | 2024-01-01 → 2024-04-01 | BULL_STRONG |
| Q09 | 2024-04-01 → 2024-07-01 | BULL_VOL |
| Q10 | 2024-07-01 → 2024-10-01 | CHOPPY |
| Q11 | 2024-10-01 → 2025-01-01 | BULL_CALM (post-election rally) |
| Q12 | 2025-01-01 → 2025-04-01 | CHOPPY / BULL_VOL (tech selloff) |
| Q13 | 2025-04-01 → 2025-07-01 | BULL_CALM |
| Q14 | 2025-07-01 → 2025-10-01 | BULL_VOL |
| Q15 | 2025-10-01 → 2026-01-01 | CHOPPY |
| Q16 | 2026-01-01 → 2026-04-01 | BULL_CALM |

Sample distribution: BEAR=2, CHOPPY=4, BULL_VOL=3, BULL_CALM=5, BULL_STRONG=2 — enough for per-regime n ≥ 2 inference on every regime except BEAR (which gets n=2 — minimum acceptable).

The actual dominant regime per window will be computed empirically from
the HMM detector output during the sim (every bar's `regime_state.regime`
is logged), not from the prior guess above.

## Promotion gate (per CLAUDE.md §5.13.4a 3-tier)

* **Tier 1 REJECT**: all regimes Δ < 0 → close out, don't promote
* **Tier 2 SCREEN**: ≥1 regime with Δ > 0 + Wilcoxon p < 0.10 → flag for follow-up; need DSR/PBO confirmation
* **Tier 3 PROMOTE**: Tier 2 + (DSR > 0.5 OR PBO < 0.5 OR n ≥ 30 + t > 3.0) → flip `regime_params.<REGIME>.<knob>` ON, leave default elsewhere

## Reproduction recipe

```bash
# Build the 6 sim configs
python scripts/build_regime_reeval_configs.py

# Run a single panel (e.g. stop007)
/tmp/run_p0_panel.sh re_stop007

# Run all 6 sequentially (after current workers free)
for label in re_stop007 re_sdl_n2 re_trail015 re_cvar025 re_cvar050 re_kelly_t1_035; do
  /tmp/run_p0_panel.sh "${label}"
done

# Analyze regime-stratified (needs analyze_regime_stratified.py — TBD)
python scripts/analyze_regime_stratified.py \
  --baseline data/logs/sim_2026-05-14_baseline_hmm \
  --treatment data/logs/sim_2026-05-15_re_stop007 \
  --label "stop_loss 0.07"
```

## Why this matters

If even 2 of the 6 candidates surface regime-conditional WIN, the
strategy gains:
* +1-3pt APY annually from BEAR/CHOPPY protection (where catastrophes happen)
* Lower MaxDD without sacrificing BULL_CALM upside
* No leverage, no new universe, no new model — pure architecture win

Cost: 6h of sim compute. ROI: very high.

## Followup if NGBoost SUSPECT confirms σ-scale bug

If `train_ngboost_proper.py` reports σ-calib ≥ +0.30 AND val μ-IC ≥ +0.03,
the σ signal IS real and the audit's "σ scale mismatch" hypothesis holds.
Action plan in that case:

1. Re-train NGBoost head with `sigma_calibration` baked in (not stamped separately)
2. Re-test E55 (NGBoost ON) regime-stratified — expect BEAR/VOL WIN
3. If E55 wins regime-conditionally, flip `ngboost.enabled = true` in BEAR/VOL only
