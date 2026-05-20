# 3-Tier Promotion Methodology

Authoritative criteria for moving a strategy / config / model variant from
"observed positive ΔAPY" to "promoted live". This replaces the prior
ad-hoc "+2pt floor" with a tier-based system grounded in the
multiple-testing literature.

> **Scope.** Applies to: hyperparameter sweeps, label-window changes,
> feature-set changes, exit-rule changes, universe changes, and any
> A/B vs the current prod baseline. Does NOT apply to: bug-fix
> rollbacks, broker-parity restorations, or operational hardening
> (these ship under §2a "theory-aligned wins").

## Why three tiers

A single threshold ("ΔAPY ≥ +2 pt → promote") fails on two failure
modes simultaneously:

1. **Too strict:** rejects real but small edges. With N=6 OOS windows
   and σ_APY ≈ 4-5 pt across same-seed runs (post-Bug-C), the
   minimum detectable effect at 80% power, two-sided α=0.05 is
   **~10-14 pt** — i.e., any threshold ≥ 2 pt rejects a true +5 pt
   edge with overwhelming probability.
2. **Too lax:** with K ≈ 50 trial configs in a session, by chance
   alone ~2-3 will show >2 pt ΔAPY purely from sampling noise
   (Harvey-Liu-Zhu 2016).

The fix is to separate **screening** (cheap, generous) from
**confirmation** (expensive, rigorous). Tier 2 catches small but
consistent edges *for further testing*; Tier 3 gates *live promotion*
on a multiple-testing-corrected statistic.

## The three tiers

### Tier 1 — REJECT (hard)

  mean ΔAPY < 0 AND mean ΔSharpe < 0
  OR
  mean ΔAPY < -1.0 pt

→ Worse than baseline on every dimension. Reject and document in
`failed-experiments-log.md`. No re-test.

### Tier 2 — SCREEN (soft accept, NOT live-promotable)

ALL of:
  - mean ΔAPY > 0
  - mean ΔSharpe ≥ 0
  - consistent_pos ≥ 4 / N  (out of N OOS windows; N typically 6)
  - mean Δα-SPY ≥ 0  (SPY-relative alpha not worse than baseline)

→ Soft candidate. Eligible for: extended OOS testing, retraining at
different cutoffs, cross-broker parity check. NOT eligible for
flipping prod config.

**Expected Type-I error rate:** ~30-40% under multiple testing
(Harvey-Liu-Zhu 2016 — without correction, this is the rate at which
purely-noise factors clear a "consistent positive ΔSR" hurdle).

### Tier 3 — CONFIRMED (live-promotable)

Tier 2 criteria + ONE of:

  (a) **DSR > 0.5** (Bailey-LdP 2014) computed with `n_trials = K`
      where K = number of configs in the batch
  (b) **PBO < 0.5** (Bailey-Borwein-LdP-Zhu 2015 via CSCV) computed
      across ≥ 4 OOS bar-orderings
  (c) **n ≥ 30 samples** with t-stat > 3.0 (Harvey-Liu-Zhu 2016
      empirical threshold for "real" anomalies after multiple-testing
      correction)

→ Publishable-rigor edge. Flip prod config in the next commit.

## References

- Bailey & López de Prado (2014). "The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting, and
  Non-Normality." *J. Portfolio Management* 40(5):94-107.
- Bailey, Borwein, López de Prado & Zhu (2015). "The Probability of
  Backtest Overfitting." *J. Computational Finance* 20(4):39-69.
- Harvey, Liu & Zhu (2016). "...and the Cross-Section of Expected
  Returns." *Review of Financial Studies* 29(1):5-68.

## Implementation

- **Script.** `scripts/analyze_experiments.py` walks
  `data/logs/sim_*/W*_<cfg>.log`, parses APY/Sharpe/Return/MaxDD,
  aligns vs the baseline run, applies the 3-tier classification, and
  emits a ranked report.
- **Runtime infrastructure.** `sim/runner.py::run_backtest_multi_seed`
  computes DSR and PBO across a K-seed ensemble; the headline log
  line `Falsifiability: DSR=X PBO=Y` is the in-loop equivalent of
  Tier 3 confirmation.
- **Library refs.** `kernel/metrics/deflated_sharpe.py`,
  `kernel/metrics/pbo.py`.

## Worked examples

### Example 1 (2026-05-11 session — historical)

53 configs analyzed across 6 OOS windows post-Bug-C. Baseline
`lambda_000` mean APY +15.2%, mean Sharpe 0.41, mean SPY-α −2.3 pt.

Result: TIER3_CONFIRMED 0 / TIER2_SCREEN 0 / NEITHER 16 (knob inert) / TIER1_REJECT 20.

Best candidate `E42_fwd60d`: mean ΔAPY +3.3, ΔSharpe +0.12, but consistency
only 3/6 → falls short of Tier 2 (needs ≥ 4/6). Kept prod baseline.

### Example 2 (2026-05-17 σ-wire A/B — historical, gate fired)

3 σ-wire conditions tested:
- global σ-on (8 dense windows): mean Δ=+3.01pp NULL (CI crosses 0)
- per-regime σ-on (BEAR/CHOPPY/BULL_VOL): mean Δ=−4.70pp negative
- per-regime + hysteresis: mean Δ=−7.89pp negative

All NEITHER/REJECT. σ-wire stays OFF in golden per Tier 3 gate. Per-regime
overlay infrastructure ADDED to golden as DORMANT for future re-evaluation.

### Example 3 (2026-05-17 long-short Phase 1 gate — SKIP)

Empirical pre-requisite test (commit `28251c2`): model bottom decile 60d-ann
return = +0.58% (positive!). All alpha on LONG side. Kelly-Gu-Xiu 2020 RFS
needs −10% to −15%/yr short alpha to justify infrastructure. **Verdict:
SKIP. Saves 3-4 weeks engineering.**

### Example 4 (2026-05-19 PatchTST DOE Phase 2 — pending verdict)

Pt_07 best (lr=1e-4, wd=0.3, seq_len=24): bull_IC +0.098, DSR +15.99 cut3
BUT fails cut5_unwind. 70/81 trials confirms "structural limit, router
thesis holds" (commit `1863a4d`). Triggered HF Trainer refactor + FiLM
Pillar B 2026-05-19 (commits `ca21654`, `78e59d3`); 5-cut × 5-seed eval
running in-flight to confirm.

## Cadence

- Run `python scripts/analyze_experiments.py` at the end of every
  multi-config experiment batch.
- Auto-emit a JSON summary into the experiment dir:
  `python scripts/analyze_experiments.py --json-out data/logs/<dir>/report.json`
- Append the headline counts (TIER1/TIER2/TIER3/NEITHER) to
  `doc/experiments/ab-journal.md` as the experiment closes.

## Anti-patterns

1. **Headline-Sharpe quoting** — quoting raw Sharpe without DSR
   inflates by ~1.5-3× when K ≥ 20. Always quote
   `SR_raw / DSR / PBO` triple (§5.13.4).
2. **Best-of-N reporting** — picking the top run from N seeds and
   citing its APY ignores that 1/N of the time noise alone produces
   a leader. Use cross-seed mean ± std.
3. **Threshold-of-the-week** — moving the bar to fit the result of
   the current experiment is fitting to the test set. Threshold is
   pinned here; changes require updating this doc, not the bar.
4. **Demanding Tier 3 for screening** — Tier 2 exists so we don't
   throw away small real edges. The block on live-promotion is
   Tier 3; Tier 2 lets ideas survive to be retested.
