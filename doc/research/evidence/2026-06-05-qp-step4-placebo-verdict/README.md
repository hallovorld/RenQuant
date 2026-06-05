# 2026-06-05 — QP §8 Step 4 placebo verdict: allocator edge is ARTIFACT, not signal

**Status**: §7.2.1 R2 placebo battery on the Step-4 A/B replay. **NEGATIVE
result** — the measured allocator differences do NOT survive the placebo,
so no allocator is a credible promotion candidate. The operator question
*"QP 改动有没有提升?"* is now answered: **no credible improvement**.
**Owner**: Claude.

---

## 1 · The placebo gate (§7.2)

The first decision-grade verdict (Step-4h, #40/#214) ranked the QP-family
allocators above the fractional-Kelly incumbent. Per §7.2.1 R2, that
ranking may NOT be quoted as a finding without a placebo block. Two
placebos, injected via the `--loader-module` hook:

- **shuffle** (`load_shuffle_placebo`): per-bar permute `fwd_return` —
  reassign each asset's realised return to a random asset in the same
  bar. Severs the mu/sigma↔fwd asset alignment the allocators size on,
  preserves the bar's marginal return distribution.
- **timeshift** (`load_timeshift_placebo`): each bar realises the NEXT
  bar's per-asset returns (re-aligned by ticker). Severs time alignment.

If an allocator's edge is REAL (depends on the signal), the paired
ΔSharpe collapses toward 0 under the placebo. If it SURVIVES, it is a
structural artifact of the allocator and must not be promoted.

## 2 · Result — the edge survives the placebo (it is artifact)

Per-allocator raw Sharpe (fwd_20d, 497 bars):

| Allocator | real | shuffle | timeshift |
|---|--:|--:|--:|
| hard_only_qp | 8.83 | 7.49 | 8.54 |
| hybrid_option_f | 8.75 | 7.60 | 8.44 |
| fractional_kelly (incumbent) | 8.41 | 7.13 | 8.10 |
| inverse_vol | 8.21 | 7.09 | 7.91 |
| equal_weight | 7.36 | 7.06 | 7.19 |

Per-allocator Sharpe barely drops under placebo — because shuffle keeps
each bar's MARGINAL return distribution (market beta survives). The
load-bearing test is the **paired ΔSharpe** (allocator A − B), which
removes the common market component and isolates the allocator
difference:

| kelly vs X | real ΔSharpe | shuffle | timeshift | verdict |
|---|--:|--:|--:|---|
| vs hard_only_qp | −5.26 | **−3.92** | −5.10 | ✗ ARTIFACT (75% survives) |
| vs hybrid_option_f | −5.31 | **−4.24** | −5.09 | ✗ ARTIFACT (80% survives) |
| vs equal_weight | −1.88 | **−1.49** | −1.79 | ✗ ARTIFACT (79% survives) |
| vs inverse_vol | +1.31 | **−0.31** | +1.29 | ✓ collapses (only one) |

**The allocator differences persist after shuffling the signal.** A real
mu→fwd edge would collapse; instead ~75–80% of every paired ΔSharpe
survives. The differences therefore come from the allocators' STRUCTURAL
weight properties (concentration, vol-targeting) interacting with the
bars' marginal return distribution — NOT from the signal predicting which
asset outperforms.

## 3 · Why — same root as the Kelly / cash-overlay rejections

This is the third independent NEGATIVE on the same underlying fact: the
panel signal is weak (IC ≈ 0.04). At that strength:
- **Kelly σ-horizon A/B** → REJECT / null (#201, #203)
- **Cash overlay** → REJECT (conditional adverse selection)
- **QP allocator A/B** → edge is artifact (this doc)

When the signal is near-zero, any sizing/allocator "improvement" is a
structural artifact of how the weights interact with marginal return
noise, not captured alpha. The cash drag and the allocator ranking both
live downstream of a signal that isn't strong enough for the choice to
matter.

## 4 · The verdict the framework would have wrongly promoted

Without the placebo, the Step-4h decision-grade verdict had
`promotion_candidate = equal_weight_top_k, next_action =
promote_to_shadow`. The placebo shows even that candidate's edge over
kelly is artifact (−1.88 real → −1.49 shuffle). **Nothing should be
promoted.** The correct `next_action` is `iterate` — and the iteration
that matters is on the SIGNAL (panel IC), not the allocator.

This is the complete value of the §7.2 placebo: it caught what the
Sharpe ranking (§212) and even the sector-cap gate (Step-4h) did not —
that the whole comparison is downstream of a non-signal.

## 5 · Caveats on the placebo itself

- 497 sparse sim-run bars; the placebo ΔSharpe magnitudes are themselves
  in the inflated-Sharpe regime. The RELATIVE collapse (real vs shuffle)
  is the interpretable quantity, not the absolute number.
- A stronger placebo (full A/A across seeds + bar-orderings, DSR/PBO on
  the paired series) would tighten the confidence. But the directional
  conclusion — edge survives shuffle → artifact — is robust: a real
  signal cannot survive having its asset alignment destroyed.

## 6 · Bottom line

**No QP allocator change is a credible improvement at the current signal
strength.** Yesterday's hard-only-QP / Hybrid-F work produces a higher
raw Sharpe, but the placebo proves that edge is structural artifact, and
Step-4h already showed hard-only QP achieves its raw Sharpe by violating
the sector cap. The honest answer to *"QP 改动有没有提升?"* is **no** —
and the bottleneck is the panel signal, not the allocator.

Artifacts: `placebo_shuffle.json`, `placebo_timeshift.json` in this
directory; the decision-grade real verdict is at
[`../2026-06-05-qp-step4h-decision-grade/verdict_fwd20_sector.json`](../2026-06-05-qp-step4h-decision-grade/verdict_fwd20_sector.json).
