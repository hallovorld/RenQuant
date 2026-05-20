# 2026-05-13 — Today's GE trade audit + risk-control verification


> **📅 Historical snapshot — content below reflects state at the date in filename/header.**
> Verify against current code per CLAUDE.md §1 "code is the source of truth" before acting on
> present-tense claims. For current state see `doc/roadmap.md` § "📍 Current state" +
> `CLAUDE.md` § "🗂 Current state".

## What happened

Daily104 at 14:06 PT placed: **BUY GE x3 @ $297.45 = $892** (8.4% of $10,686 NAV).

Post-trade holdings (7 positions):

| Ticker | Target w | GICS sector | Realized 60d corr with GE |
|---|---:|---|---:|
| BA | 6.79% | industrial | 0.455 |
| FTNT | 9.67% | software | — |
| MU | 7.48% | ai_chip | — |
| DOCU | 9.67% | software | — |
| MCD | 9.67% | consumer | — |
| VRT | 9.67% | datacenter_hw | — |
| **GE** | **9.67%** | **industrial** | **— (new)** |

## Initial concern: BA+GE concentration in aviation theme

Industrial sector = 2 holdings (BA 6.79% + GE 9.67% = 16.5%). Both rely on
commercial aviation cycle. Surface-level "thematic concentration" critique.

## Verification (what I actually found)

### Risk control #1: Sector hard cap — ACTIVE
- `qp_sector_cap_enabled = True` (default)
- `max_positions_per_sector = 6`
- Current: 2/6 industrial holdings. Cap loose, not binding.

### Risk control #2: Correlation group cap — ACTIVE
- `qp_correlation_cap_enabled = True` (default)
- `correlation_guard_threshold = 0.70`
- Group cap = 2 × per_name_cap when |corr| > threshold
- **BA-GE realized corr = 0.455** (60d, well below 0.70)
- Cap does NOT bind on BA-GE pair → QP correctly allowed both

### Counter-intuitive finding

GE's top realized-correlation partners:
1. **RTX: 0.572** (engine maker peer)
2. **ASML: 0.560** (semicap — aerospace capex cycle sync)
3. **LRCX: 0.555** (semicap)
4. **CAT: 0.499** (industrial cycle)
5. **AMAT: 0.476** (semicap)

GE-BA (0.455) is only #6+ on GE's correlation list. The intuition
"BA+GE = same aviation bet" is **semantically true but realized
correlation is moderate**. Statistical theme detection (correlation) ≠
verbal theme detection (sector/industry labels).

## Conclusion

**My initial concern was OVERSTATED.** The risk-control system worked as
designed:
- Sector cap is active but loose (6 per sector × 20% × 0.64 conf ≈ 76% sector weight cap → 16.5% is well below).
- Correlation cap is active with threshold 0.70 (BA-GE 0.455 below threshold).
- GE buy was a **legitimate ranking-based pick** that passes all risk gates.

## What is a REAL gap

**Theme-level concentration that GICS+0.70-corr misses**: pairs with
0.45-0.70 correlation (like BA-GE) don't trigger group cap but DO share
underlying theme exposure. The 0.70 threshold is tuned for blatant
duplicates; moderate correlation pairs sneak through.

Possible tightenings (need experimental verification, NOT bug fixes):

1. **Lower sector cap** (`max_positions_per_sector: 6 → 3`)
   - Forces broader sector diversification
   - May reduce alpha capture in trending sectors
2. **Lower correlation threshold** (`0.70 → 0.50`)
   - Catches BA-GE (0.455 — close) and GE-RTX (0.572 — yes)
   - May over-constrain (many pairs in 0.45-0.55 range)

## Experiments queued

Side configs written (don't touch live):
- `sim_sector_cap3_ext.json` + `_pre2024.json`
- `sim_corr05_ext.json` + `_pre2024.json`

Standard 16-window paired-daily eval per `doc/research/evaluation-protocol.md`.
Tier 3 promotion requires t_pool > 3.0, DSR > 0.5, p < 0.01.

**Will run AFTER current wl174_retrained + horizon batches complete**
(currently consuming the 8-concurrent sim slot). Estimated launch
~16:00-17:00 PT.

## Updated trade review

My GE evaluation:
- **6.5/10 → 7/10**: risk controls properly active; my concentration
  worry was misplaced (correlation supports the diversification claim).
- The "追高入场" critique stands (entry near 52w high on a momentum factor).
- Single-name 8.4% conservative under 20% max.
- Strategy mechanically correct; entry timing is the only honest concern.
