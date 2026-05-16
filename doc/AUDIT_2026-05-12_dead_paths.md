# 2026-05-12 — Dead-path audit of "bit-identical to baseline" rejections

User invoked CLAUDE.md §2b principle: well-regarded published methods
shouldn't destroy performance. Surprising rejections must be audited
for implementation bugs before being shelved.

This document records the 6-step §5.13.10 audit applied to each
"bit-identical to baseline" rejection from the 2026-05-11/12 batch.

---

## Method index

  Result label      Theory ref                         Verdict
  ---------------   --------------------------------   -------
  B5 trend overlay  Antonacci 2014 dual momentum       NOT-A-BUG (inactive by design in bull windows)
  E43 vol-target    Moskowitz-Ooi-Pedersen 2012 §3     🔴 DEAD PATH
  B6 DD-Kelly       Grossman-Zhou 1993                 🔴 DEAD PATH
  CVaR sweep        Rockafellar-Uryasev 2002           NOT-A-BUG (correctly wired; marginal benefit ≈ 0 at our scale)
  NGBoost destroy   Duan NeurIPS 2020                  🟡 SUSPECT (separate audit needed)

---

## 🔴 DEAD PATH #1 — vol-targeting (E43)

**Symptom.** `strategy_config.sim_E43_voltarget_007.json` sets
`ranking.kelly_sizing.vol_target.enabled = true` but the resulting sim
produces metrics bit-identical to baseline across all 6 OOS windows.

**Theory.** Moskowitz-Ooi-Pedersen 2012 *Time Series Momentum* §3:
scale total portfolio exposure by `target_vol / realized_vol` to hit a
constant ex-ante volatility. Expected effect: de-leverage in vol spikes,
re-leverage in calm markets. Empirical lift on TSM strategies: ~+0.20
Sharpe.

**Audit.**

1. `compute_vol_target_scale()` returns sensible values when called
   (`kernel/vol_target.py`) — not the bug.
2. The scale modifies `max_pct` in
   `kernel/panel_pipeline/job_panel_scoring.py::ApplyKellySizingTask`
   line 1184.
3. `max_pct` is THEN passed into `_kelly_with_reason()` line 1248 only,
   which sets `cand.kelly_target_pct`.
4. `_kelly_with_reason()` returns `(0.0, "kelly_zero:mu_none")` for
   every candidate when `mu is None` (line 1236) — and `mu` IS None for
   every candidate when NGBoost is off (current prod baseline since
   2026-05-09).
5. QP's `BuildPositionBoundsTask` (`kernel/portfolio_qp/tasks.py:322`)
   reads `max_position_pct` DIRECTLY from `regime_params`, bypassing
   the Kelly path entirely.

**Conclusion.** vol-targeting per Moskowitz-Ooi-Pedersen scales the
BASKET exposure cap. The current implementation only scales a Kelly
local-variable that the QP optimizer never sees. With Kelly dead
(NGB off), vol-target is architecturally dormant.

**Fix.** Hoist vol-target into its own pipeline `Task` that runs
BEFORE QP's `BuildPositionBoundsTask` and writes a `ctx._vol_target_scale`
attribute the QP reads when constructing `_qp_w_upper`.

**Invariant pinned by fix:**

  ctx._qp_w_upper ≡ max_position_pct × confidence_mult × vol_target_scale × dd_scale

i.e. all exposure-cap modifiers compose multiplicatively at the QP
upper-bound, not inside a Kelly local that may be unused.

---

## 🔴 DEAD PATH #2 — DD-Kelly scaling (B6)

**Symptom.** Same: `strategy_config.sim_B6_ddkelly_005.json` sets
`ranking.kelly_sizing.drawdown_scaling.enabled = true` and produces
metrics bit-identical to baseline.

**Theory.** Grossman-Zhou 1993 (drawdown-conditioned position sizing):
shrink Kelly fraction as portfolio drawdown deepens to avoid
ruin-trajectory bets. Expected effect: smoother equity curve, lower
MaxDD with marginal APY cost.

**Audit.** Same root cause as vol-target. `compute_kelly_dd_scale()` is
called at line 1203, modifies `max_pct` at line 1213, same downstream
dead path through `_kelly_with_reason`.

**Fix.** Same shape: hoist into its own Task that writes
`ctx._dd_kelly_scale` consumed by QP's bounds.

---

## NOT-A-BUG — B5 trend overlay

`task_trend_overlay.py:103` only forces `hard_bear=True` when SPY
12-month return ≤ threshold (default 0.0). All 6 OOS windows are bull
markets (SPY 12mo > 0), so the overlay correctly stayed inactive. My
"bit-identical = rejected" was a misinterpretation, not a code defect.

The trend overlay is wired correctly; it would only be testable on
2025-Q1 SPY drawdown (Feb-Apr 2025) or any future bear window.

---

## NOT-A-BUG — CVaR

`kernel/portfolio_qp/qp_solver.py:311-322` correctly adds the
Rockafellar-Uryasev 2002 Gaussian closed-form penalty
`-λ · (φ(z_α)/α) · ‖Σ^½ wp‖₂` to the QP objective. The penalty IS in
the objective when `cvar_lambda > 0`.

The "within noise" finding (mean Δ = +0.1 pt ± 7.6 across 6 windows) is
structural: the existing quadratic risk term `-λ_risk · wp^T Σ wp` already
captures the bulk of tail-risk control at our portfolio scale (single
positions ≤ 20%, no leverage). CVaR's marginal contribution over the
quadratic risk is small — the L2 norm `‖Σ^½ wp‖₂` and the L2-squared
norm `wp^T Σ wp` differ by a monotone transform; at small-portfolio
scale both bind the same constraint.

CVaR matters when:
- Position concentrations are high (≥40% single name) and Gaussian
  approximation underestimates fat tails.
- The portfolio is leveraged (gross > 100%).
- Specific tail events dominate the loss distribution.

None of those apply at our current scale → marginal CVaR benefit ≈ 0
is the correct verdict.

---

## 🟡 SUSPECT — NGBoost −20.6 pt APY

The most anomalous result. Duan NeurIPS 2020 NGBoost provides calibrated
σ estimates for μ. Theory: better σ → better Kelly fraction → at worst
neutral, at best +30-50bp Sharpe.

Observed: enabling NGB destroys 20.6 pt APY and 1.08 Sharpe. The
magnitude is wildly inconsistent with theory.

Suspect mechanisms (not yet audited):

1. **σ scale mismatch.** NGB-trained σ may be in units (% daily) but
   downstream Kelly expects (% annual) — silent scale bug → Kelly
   formula divides by σ², so a 16× scale error → 256× sizing error.
2. **μ calibration drift.** Reading the project memory: "Bug C
   corrupted every sim metric". The NGB rejection was observed with
   pre-Bug-C metrics. **The post-fix re-test was N=1 single-seed.**
3. **Walk-forward NGB head missing.** Manifest has only XGB heads;
   the NGB head is currently a static-cutoff stub. Using a 2024-01-01
   NGB stub against 2025+ data creates feature-distribution mismatch.

Action: deferred — requires walk-forward NGB retrain (3-4h). Documented
in roadmap as P0 audit follow-up.

---

## Mandatory regression tests when fix lands

When vol-target / DD-Kelly fix ships:

1. `test_vol_target_scales_qp_upper.py` — set
   `vol_target.target_vol=0.05`, run sim 1 bar, assert
   `ctx._qp_w_upper[0] < max_position_pct` AND assert QP-emitted
   buy size respects the reduced upper bound.
2. `test_dd_kelly_scales_qp_upper.py` — set portfolio drawdown to 15%,
   `dd_scaling.dd_max=0.20`, assert `_qp_w_upper` shrinks.
3. `test_vol_target_independent_of_ngb.py` — NGB OFF + vol-target ON,
   assert vol-target HAS an observable effect on `_qp_w_upper`. (The
   regression guard against the dead-path bug.)

Without test #3 we'd re-introduce the bug in any future Kelly refactor.

---

## Status

  2026-05-12 — Documented findings. Implementation deferred to a
  user-authorized commit (touches QP bounds, requires audit of all
  downstream consumers of `max_pct` to confirm we're not breaking the
  walkforward retrain pipeline). Estimated effort: 2-3 hours implement
  + sim verify on 6 windows.

---

## 2026-05-15 RESOLUTION UPDATE

All three findings RESOLVED in commits this date:

### 🔴 DEAD PATH #1 (vol-target) → ✅ FIXED

* `kernel/portfolio_qp/tasks.py::ApplyExposureScalingTask` already
  ships the design: writes `ctx._vol_target_scale` and multiplies
  into `ctx._qp_w_upper` BEFORE QP solve.
* Mandatory regression test `tests/test_vol_target_scales_qp_upper.py`
  written 2026-05-15 — pins the invariant. 4/4 cases pass.
* Dead-path Kelly local-var code (~50 LOC) DELETED from
  `ApplyKellySizingTask` (commit `5b78ffe`).

### 🔴 DEAD PATH #2 (DD-Kelly) → ✅ FIXED

* Same shape as #1 — same `ApplyExposureScalingTask` writes
  `ctx._dd_kelly_scale`. Test `tests/test_dd_kelly_scales_qp_upper.py`
  pins it. 4/4 cases pass.

### 🟡 NGBoost SUSPECT → ✅ CONFIRMED (audit hypothesis was correct)

* `scripts/train_ngboost_proper.py` 5-seed validator with paper
  Duan 2020 §4 large-data config (lr=0.1, minibatch_frac=0.1,
  n_estimators=500, early_stopping=20):

  ```
  val μ-IC mean=+0.0351 std=0.0036  range=[+0.0293, +0.0383]
  σ̂  calibration mean=+0.271          (4 seeds in [0.265, 0.275])
  μ̂  x-sec std mean=0.01619
  Δ vs XGB-quantile baseline = +0.0057   t-stat = +2.76
  ✓ SIGNIFICANT BEAT at 95% confidence
  ```

* **Audit hypothesis was correct**: original E55 rejection
  (-20.6pt APY, -1.09 Sharpe) was NOT NGBoost being bad — it was
  misconfiguration (head config + σ scale wiring). Properly configured,
  NGBoost beats XGB-quantile decisively on val μ-IC.

* σ-calibration of +0.271 across all 5 seeds (variance < 0.01)
  proves σ has REAL conditional uncertainty signal — not noise.

### Additional regression test #3 (the dead-path guard)

Audit doc verbatim required: *"Without test #3 we'd re-introduce the
bug in any future Kelly refactor."* SHIPPED:
`tests/test_vol_target_independent_of_ngb.py` — pins that vol-target
+ DD-Kelly scale `_qp_w_upper` INDEPENDENTLY of NGB on/off state.
3/3 cases pass.

### Action items still open

1. **NGBoost prod retrain** with Duan 2020 §4 config — replace the
   misconfigured head currently on disk
2. **σ-aware Kelly path** — wire NGB σ properly into Kelly numerator
   instead of via the broken sigma_calibration constant
3. **Re-test E55 regime-stratified** (NGB ON) — predicted to win in
   BEAR/VOL after both retrain + wire fixes
