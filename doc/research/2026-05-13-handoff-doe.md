# 2026-05-13 — Hand-off mode DOE sweep

## Design

10-panel unified parallel sweep, **5-slot global queue** (xargs -P 5
matches hardware: 10 cores ÷ 2 cores-per-sim). Each panel = 16
non-overlapping 3-month windows, paired-daily HAC + stationary
bootstrap + DSR/PBO via `eval_paired_returns.py`.

### Why this design

- **Single-knob exhaustion**: 3 of 3 single-knob κ values produced
  NEITHER. EMA50-off produced TIER 1 REJECT. The κ knob alone cannot
  beat baseline.
- **Re-test 3 prior candidates that may have been killed by broken
  6-window methodology**: vt15, gk094, gk15. New 16-window paired
  framework may reveal them as Tier 3 winners.
- **Test interaction**: p15_cellA = κ=0.05 + min_dw=0.05 combined.
- **Test 4 untouched knobs**: risk_aversion=5, max_position_pct=0.15,
  sector_cap=4, no_trade_band=2× factor. Each has theoretical backing
  (tighter risk concentration, better sector diversification, less
  trading).

### Sample-size justification

16 windows × ~62 trading days each = ~990 paired daily observations.
Newey-West HAC with Andrews-1991 optimal lag. K_trials=100 for DSR
multi-comparison correction (Bailey-López de Prado 2014). PBO via
CSCV (Bailey-Borwein-LdP-Zhu 2015).

### 3-tier promotion criteria (per `feedback_promotion_methodology.md`)

- **Tier 1 REJECT**: mean ΔAPY < 0 AND mean ΔSharpe < 0
- **Tier 2 SCREEN**: mean ΔAPY > 0, mean ΔSharpe ≥ 0, ≥ 4/16 consistent,
  ΔSPY-α ≥ 0. Not auto-promotable.
- **Tier 3 LIVE-PROMOTABLE**: Tier 2 + (DSR > 0.5 OR PBO < 0.5 OR
  n_days ≥ 30 with t_pool > 3.0).

## Panels in this sweep

| # | Panel | Mechanism | Theory backing |
|---:|---|---|---|
| 1 | kappa05 | QP `cost_kappa = 0.05` | Almgren-Chriss linear t-cost |
| 2 | mindw05 | QP `min_dw_pct = 0.05` | Size-based trade filter |
| 3 | p15_cellA | both 1+2 combined | 2-way interaction (§5.14) |
| 4 | vt15 | vol-target 15% | Almgren-Chriss 2000 §2 |
| 5 | gk094 | Grinold-Kahn α→μ IC=0.094 | Grinold-Kahn 1999 ch.5 |
| 6 | gk15 | Grinold-Kahn IC=0.15 | Grinold-Kahn (higher IC) |
| 7 | riskav5 | risk_aversion γ=5 | Markowitz; tighter risk |
| 8 | maxpos15 | max_position_pct=0.15 | Concentration risk (currently 0.20) |
| 9 | sectorcap4 | max 4 holdings/sector | Diversification (currently 6) |
| 10 | ntband2x | no_trade_band_factor=2.0 | Smaller perturbations |

## Hardware budget

| Resource | Value |
|---|---:|
| Machine | M2 Pro |
| Cores | 10 |
| RAM | 32 GB |
| OMP/MKL/OPENBLAS_NUM_THREADS | 2 |
| sim_slots | 5 (= 10 ÷ 2) |
| sim wallclock (full machine) | 20-25 min |
| sim wallclock (5-up oversubscribed marginally) | 25-30 min |
| Panels × windows | 10 × 16 = 160 sims |
| Total CPU time | 160 × 25 = 4000 sim-min |
| Wallclock at 5-slot | 800 min ≈ **13.3h** |

## Hand-off auto-promote policy

Per `feedback_auto_promote_to_prod.md`:

1. **Auto-promote without asking** any panel with `verdict ∈ {TIER3_PROMOTE}` OR
   (verdict=TIER2_SCREEN AND DSR > 0.5). Apply the 9-step protocol:
   - Backup `strategy_config.golden.json` and `strategy_config.json`
   - Flip the winning knob in both
   - Pin invariant test (`tests/test_promoted_<knob>.py`)
   - Run pytest full suite
   - 1-window sanity sim with the new config
   - Commit with `[promote]` tag
   - Update `MEMORY.md`
   - Append finding to this doc

2. **Flag for user OK (don't auto-promote)**:
   - EMA50-off variants (risk-loosening; already TIER 1 REJECT anyway)
   - Broker/architecture changes (none in this sweep)
   - Anything outside the panel list above

3. **Tie-break** when multiple Tier 3 winners surface:
   - Pick highest DSR
   - Then highest |t_pool|
   - Then highest consistency

4. **Combined promotion** (Phase 4): if 2+ orthogonal winners, queue a
   combined-config 16-window confirmation panel before promoting both.

## Status cadence (every 3 hours)

`/tmp/cycle_analyze_promote.sh` runs each cycle. Output:

- `doc/research/2026-05-13-handoff-status.md` — running status table
- `/tmp/cycle_<ts>.txt` — per-cycle log
- Auto-promotion commits with `[promote]` tag if any Tier 3 surfaces

## Prior Phase 1 results (for cross-reference, not in v3 queue)

| Panel | Verdict | mean Δ/yr | t_pool | DSR | cons |
|---|---|---:|---:|---:|---:|
| κ=0.0001 (baseline) | — | — | — | — | — |
| κ=0.001 (range-find only) | inconclusive | — | — | — | — |
| κ=0.003 | NEITHER | +0.48% | +0.22 | 0.690 | 8/16 |
| κ=0.01 (range-find only) | inconclusive | — | — | — | — |
| κ=0.1 | NEITHER | +0.58% | +0.17 | 0.514 | 9/16 |
| EMA50-off | TIER 1 REJECT | −3.21% | −0.63 | 0.000 | 6/16 |
| min_dw_pct=0.05 | RE-RUN in v3 sweep | — | — | — | — |
| κ=0.05 | RE-RUN in v3 sweep | — | — | — | — |
