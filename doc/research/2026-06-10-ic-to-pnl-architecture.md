# From IC to Sharpe: a ground-up redesign of the signal→portfolio path

**Date:** 2026-06-10 · **Author:** Claude (research proposal) · **Status:** RFC — DESIGN, not a verdict. Requires independent review + the experiments below before any production change.
**Mandate:** Operator: "PatchTST IC ≈ 0.1 but realized APY/Sharpe are terrible — the decision tree wastes the IC. Forget the current architecture; propose something more scientific."

> **Scope discipline.** This document answers a *conditional* question: **IF** the panel model has a real, placebo-clean cross-sectional IC of ~0.10, what is the smallest, most theoretically grounded portfolio-construction path that converts that IC into Sharpe? It does **not** assert the IC is real. The 2026-06-02 validity audit found the PatchTST `B_tuned` IC leak-contaminated (timeshift placebo +0.067 > real +0.044). **IC reality is a hard prerequisite gate (§7), measured independently. A clean architecture on a fake signal is worth zero.** The two questions are orthogonal and must not be conflated again (the 2026-06-09 operator-override incident conflated them).

---

## 1 · The symptom, stated precisely

The complaint "IC is 0.1 but Sharpe is terrible" is the single most common failure mode in quant equity, and it has a precise name: a **low transfer coefficient**. The forensics on the 2026-06-10 WF gate run (GBDT prod recipe, the only path that currently trades) show the fingerprint cleanly:

| Layer | Observation (cut C, 2025-04→2026-03) | Source |
|---|---|---|
| Per-trade economics | win rate 49%, **mean P&L +2.38%** over ~49-day holds | round-trip ledger |
| Holding period | median 35–62d across cuts — **matched to the 60d label** | round-trip ledger |
| Breadth realized | **29 distinct names** traded in a year, of a 142 watchlist | round-trip ledger |
| Portfolio result | **Sharpe +0.39 / +0.04 / +0.69**, 0/3 beat SPY | WF gate |

The individual bets are fine — positive expectancy, horizon-matched. **The portfolio is where the signal dies.** That is definitionally a transfer-coefficient problem, not an alpha problem. (Note: cut C's trades are attributed to per-ticker `Manual/XGBoost/QLearning/Classification` trees, *not* the panel model — see §6.3; the panel IC never even reaches the optimizer in the current decision tree. That is the most direct waste of all.)

---

## 2 · The theory that the current architecture violates

### 2.1 The Fundamental Law, with the term everyone forgets

Grinold & Kahn (2000) give the headline:

> IR = IC · √BR

But the operational form is Clarke, de Silva & Thorley (2002, *FAJ*, "Portfolio Constraints and the Fundamental Law of Active Management"), which inserts the **Transfer Coefficient** TC:

> **IR = TC · IC · √BR**

TC ∈ [0, 1] is the cross-sectional correlation between the *signal-implied* active weights and the *actually-held* active weights. It is the fraction of the alpha that survives the journey from forecast to position. CDST's central empirical result: realistic long-only + turnover + position-cap constraints drive TC to **0.3–0.6**. Every additional nonlinear gate lowers it further.

Plugging in the (hypothetical) numbers:

- IC = 0.10, BR ≈ 29 names × (252/49) ≈ 29 × 5.1 ≈ **148 independent bets/yr** → √BR ≈ 12.2
- Ideal IR = 0.10 × 12.2 ≈ **1.22** (gross, before TC and costs)
- Observed IR ≈ 0.37 → implied **TC ≈ 0.37 / 1.22 ≈ 0.30**

**~70% of the theoretical information ratio is being destroyed between the forecast and the fill.** That is the number to attack. (And breadth itself is being thrown away: 29 of 142 names → if the signal is real across the universe, √BR could be 1.7× higher just by holding more of it — Grinold-Kahn breadth.)

### 2.2 Where TC leaks — the seven gates between rank and weight

The current `JointPortfolioQPJob` interposes (in order): admission rank-floor (0.55) → ER-horizon/floor gates → QP μ-contract → exposure/conviction caps → Davis-Norman no-trade bands → sector/correlation caps → emission-side single-day-loss stops + soft-sell horizon guards + calibrator-saturation abstain. Each is individually defensible; **composed, they are a low-pass filter on the alpha.** Three are especially lossy for a *ranking* signal:

1. **Long-only.** IC measures the full cross-section — top *and* bottom decile. Long-only discards the entire short leg. For a symmetric signal CDST show this alone caps TC near **0.5** (you keep ~half the information). PatchTST's measured edge is in fact *stronger in the tails* (BEAR +0.22) — exactly the part long-only cannot harvest.
2. **Hard admission floors (rank ≥ 0.55).** A monotone signal's value is in the *full ordering*; a hard floor turns a continuous forecast into a coarse step function and collapses breadth (142 → 29). This is the Qian-Hua-Sorensen (2007) "IC is a continuous quantity; don't threshold it" point.
3. **Daily path-dependent stops on a 60-day thesis.** A single-day-loss stop on a signal whose information horizon is 60 days is pure variance — it sells the bottom of noise (we measured +3.6pp/trade post-exit regret in the live ledger, dominated by momentum names). Han-Zhou-Zhu (2016, *JFE*) show stop-loss rules add value only in high-vol/down regimes; in calm-bull they are a TC tax.

### 2.3 The deeper error: one pipeline doing two incompatible jobs

The current design fuses **alpha capture** (which wants to track the cross-sectional forecast as faithfully as possible) and **risk control** (which wants to cut exposure, stop losses, respect caps) into one optimizer + one emission stage. These objectives fight: every risk gate lowers TC, and the optimizer cannot tell "I'm trimming for risk" from "I'm trimming because the signal weakened." The result is an unattributable blend where you cannot measure how much IC you kept.

---

## 3 · The proposal: a two-stage architecture that *measures* its own transfer coefficient

Forget the monolithic decision tree. Replace it with two **separable, individually-testable** stages, in the spirit of Grinold (1994, *JPM*, "Alpha is Volatility times IC times Score") and the Gu-Kelly-Xiu (2020, *RFS*) ML-asset-pricing evaluation standard.

```
 panel forecast (z-scored cross-sectional rank, per day)
        │
        ▼
 ┌─────────────────────────────┐   STAGE A — ALPHA PORTFOLIO
 │  α_i = IC · σ_i · z_i        │   (Grinold 1994: the ONLY honest map
 │  target_w ∝ α / (γ·Σ)        │    from a rank to an active weight)
 │  long-short, horizon-held    │
 └─────────────────────────────┘
        │  w_alpha (the signal's own opinion, nothing else)
        ▼
 ┌─────────────────────────────┐   STAGE B — RISK OVERLAY
 │  vol-target scale, drawdown  │   (scales the WHOLE book up/down;
 │  halt, gross/net caps, costs │    never re-picks names → TC-preserving)
 │  GP-2013 cost-aware glide    │
 └─────────────────────────────┘
        │  w_final
        ▼  orders
```

**The discipline that makes this scientific:** Stage A is a *deterministic monotone function of the forecast*. Its TC versus the raw rank is measurable and near 1.0 by construction. Stage B scales the entire portfolio — it changes leverage, not selection — so it **cannot lower the cross-sectional TC** (it multiplies all active weights by a scalar). Every place TC is lost is now explicit and attributable, instead of smeared across seven gates.

### 3.1 Stage A — the alpha portfolio (three variants, ranked by ambition)

| Variant | Construction | Harvests | Reference |
|---|---|---|---|
| **A0 — rank-decile L/S (the ceiling)** | long top decile, short bottom decile, equal-weight, rebalance at horizon | full IC, both tails, max breadth | Fama-French sorts; Gu-Kelly-Xiu 2020 |
| **A1 — α-proportional L/S** | w_i ∝ IC·σ_i·z_i, dollar-neutral, vol-scaled | continuous ordering, not just deciles | Grinold 1994 |
| **A2 — long-only α-tilt** | A1 projected onto w ≥ 0, Σw = 1 (the current real-money constraint) | top half of IC; the production-feasible point | CDST 2002 §long-only |

A0 is **not a deployable strategy** — it is the **measurement instrument**: the Sharpe of A0 is the empirical ceiling the IC implies, the honest answer to "what is this IC worth?" If A0's Sharpe is also terrible, the IC is not real (or not tradeable at this horizon/universe) and no architecture saves it — fail fast, route to §7. If A0 is strong and A2 is weak, the long-only constraint is the tax and we quantify it exactly (and can revisit whether a small short sleeve is worth it).

### 3.2 Stage B — risk overlay (scalar, TC-neutral)

- **Volatility targeting** (Moskowitz-Ooi-Pedersen 2012): scale gross exposure to a target realized vol. Multiplies all weights equally → TC-invariant.
- **Drawdown throttle** (Grossman-Zhou 1993): reduce gross as drawdown deepens. Scalar.
- **Cost-aware glide** (Gârleanu-Pedersen 2013): trade *toward* the new target a fraction per day (the "aim portfolio"), so turnover is smooth and costs are paid only for persistent signal — *this* is the principled replacement for daily stops and no-trade bands.
- **No stock-level second-guessing.** Hard risk exits (a genuine blow-up stop, wash-sale law, liquidity) remain as a thin safety layer, but they are the exception path, logged as TC leakage, not the main loop.

---

## 4 · Why this directly fixes the operator's complaint

| Current waste (TC leak) | Fix |
|---|---|
| Panel IC never reaches the optimizer (per-ticker trees trade instead, §6.3) | Stage A consumes the panel forecast directly as the *only* selection input |
| Long-only discards the short leg + tail edge | A0/A1 measure and (optionally) harvest both legs; A2 quantifies the long-only tax |
| Rank-floor 0.55 collapses 142→29 names | continuous α-weighting restores breadth → √BR up ~1.7× |
| Daily stops shred a 60d thesis (+3.6pp regret) | GP-2013 glide + horizon-held; stops demoted to safety-only |
| Risk trims indistinguishable from alpha trims | Stage B is scalar; selection and risk are separable and attributable |

---

## 5 · Experiment design (falsifiable, placebo-clean, uses the existing WF harness)

All runs on the existing walk-forward manifold (`walkforward_v2_*`, point-in-time models, the harness fixed on 2026-06-10), per-regime PRIMARY then pooled, DSR/PBO on every number (Bailey-López de Prado 2014), shuffled-label + timeshift placebo on every claim (§5.2 battery).

**E1 — Transfer-coefficient decomposition (the headline experiment).**
Start from A0 (the ceiling) and add one production constraint at a time, measuring Sharpe and TC at each step:

```
A0  rank-decile L/S, horizon-held, no costs        → IC ceiling Sharpe
+1  add realistic costs (κ, impact)                → cost drag
+2  long-only projection (A2)                      → long-only tax
+3  add vol-target + drawdown overlay (Stage B)    → risk-overlay effect (expect ≈ neutral TC)
+4  add admission floors                            → floor tax
+5  add daily stops                                → stop tax
=   current architecture                           → should reproduce ~Sharpe 0.37
```

The step that drops Sharpe most **is** the thing wasting the IC. This converts the operator's qualitative complaint into a ranked, quantitative target list. This is the deliverable that decides everything downstream.

**E2 — Horizon sweep.** Rebalance at {20, 40, 60, 90}d; confirm the 60d label is the right holding horizon and measure IC decay (Qian-Hua-Sorensen IC-decay curve). Cheap, high-information.

**E3 — Breadth restoration.** A2 with rank-floor removed vs kept; measure the √BR lift the continuous weighting buys.

**E4 — Short-sleeve value.** A2 (long-only) vs A1 (dollar-neutral) vs a capped long-biased (e.g. 130/30); is the short leg worth the operational cost at this account size? (Likely NO at $10k — but now it's a measured decision, not an assumption.)

---

## 6 · What this means for the existing code (no rewrite-from-scratch)

This is an **architecture for the alpha→weight map**, implemented as a new allocator behind the *existing* `ConstraintSnapshot` contract (the §8 measurement plan already built the seam). It is exactly the kind of candidate the **step-4g 5-baseline A/B replay** was built to adjudicate — A0/A1/A2 become three more baselines in that harness. Nothing about this proposal requires bypassing the gate; it *feeds* the gate cleaner candidates.

### 6.1 Build order
1. Implement A0/A1/A2 as `AlphaPortfolioAllocator` variants (renquant-pipeline, behind ConstraintSnapshot).
2. Stage B as a post-allocator scalar overlay (vol-target already exists in `ApplyExposureScalingTask` — reuse, don't rebuild).
3. Run E1–E4 through the WF + replay harness.
4. Verdict doc; promote only on DSR>0 + placebo-clean + per-regime evidence (§7).

### 6.2 What survives from today
ConstraintSnapshot, the WF gate, the sanity battery, vol-target, cost model, regime detector. This is a **re-wiring of the selection→sizing core**, not a teardown of the platform.

### 6.3 The most damning current finding
In cut C, the trades are attributed to per-ticker `Manual/XGBoost/QLearning/Classification` trees — **the panel IC is not the thing selecting names.** Whatever the panel's IC is, the current decision tree barely uses it for selection. Stage A makes the panel forecast the *sole* selection input, which is the only way an IC of 0.1 can become Sharpe at all.

---

## 7 · Hard prerequisite gate (do not skip — this is the 2026-06-09 lesson)

Before *any* of this is worth running on PatchTST specifically:

1. **IC reality.** Close the leakage investigation (renquant-model PatchTST B_tuned, 2026-06-02 audit). The IC used in Stage A must be the *placebo-clean OOS* IC, not the calibrator fit-window `pool_ic=0.13` (which is in-sample and not what "0.1" should mean). If the clean OOS IC is ~0.03–0.04, Stage A's ceiling is computed on *that*, honestly.
2. **A0 sanity.** If the rank-decile ceiling portfolio (A0) on the clean signal is not materially > SPY on a DSR basis, stop — the IC is not tradeable at this universe/horizon and the architecture cannot manufacture alpha that isn't there.
3. **Only then** does the long-only/overlay engineering (A2 + Stage B) earn its keep.

The architecture is signal-agnostic: it will harvest whatever real IC exists (GBDT or PatchTST) and waste none of it — but it manufactures nothing. **Clean signal first, clean architecture second.**

---

## 8 · References (read before implementing — CLAUDE.md §5.12)

- Grinold & Kahn 2000, *Active Portfolio Management* — Fundamental Law (IR = IC·√BR).
- **Clarke, de Silva, Thorley 2002, *FAJ* — Transfer Coefficient (IR = TC·IC·√BR); long-only TC≈0.5.** ← the central reference.
- Grinold 1994, *JPM*, "Alpha is Volatility times IC times Score" — the α = IC·σ·z map (Stage A).
- Qian, Hua, Sorensen 2007, *Quantitative Equity Portfolio Management* — IC as a continuous quantity, IC decay, don't-threshold.
- Gârleanu & Pedersen 2013, *JF*, "Dynamic Trading with Predictable Returns and Transaction Costs" — aim portfolio / cost-aware glide (Stage B).
- Gu, Kelly, Xiu 2020, *RFS*, "Empirical Asset Pricing via Machine Learning" — decile L/S as the canonical ML-alpha evaluation (A0).
- Moskowitz, Ooi, Pedersen 2012, *JFE* — time-series momentum / volatility targeting (Stage B).
- Grossman & Zhou 1993 — drawdown-conditioned exposure (Stage B).
- Han, Zhou, Zhu 2016, *JFE* — when stop-loss rules add value (regime-conditional; calm-bull they don't).
- Bailey & López de Prado 2014, *JPM* — Deflated Sharpe Ratio (every verdict).

---

**Next action (requires review approval — not self-merged):** implement A0 as the measurement instrument and run E1 on the clean signal. E1's transfer-coefficient decomposition is the single highest-information experiment in the entire recovery program — it converts "the decision tree wastes the IC" from an assertion into a ranked, quantified target list.

Agent-Origin: Claude
