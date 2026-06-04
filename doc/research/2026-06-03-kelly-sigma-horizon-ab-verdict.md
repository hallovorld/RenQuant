# 2026-06-03 — Kelly σ-horizon A/B verdict: mechanically correct, operationally inert

**Bottom line**: the σ-horizon fix recommended by
[`2026-06-03-kelly-sizing-audit.md`](2026-06-03-kelly-sizing-audit.md)
(PR #158) **does what the audit's math said it would — and changes nothing
about the portfolio.** Matching σ to the 60-day μ horizon raises Kelly
targets in every regime, but actual holdings, cash%, and returns are
**byte-identical** to control. This **refutes the audit's core thesis**:
Kelly underweighting is real but is NOT the cause of the high cash drag.
Do not ship the σ-horizon change as a cash-drag fix.

**Per §7.12**: theory predicted the fix would deploy more capital; the
result is exactly zero change. The first hypothesis was "my config/impl is
wrong" — ruled out below (the treatment IS applied and Kelly targets DO
move). The surviving explanation is architectural: Kelly is a non-binding
ceiling.

---

## 1 · The run

```
scripts/run_kelly_sigma_horizon_ab.py --execute \
  --base-config strategy_config.sim_kelly_ab_admoff.json \
  --manifest-path artifacts/sim/walkforward_manifest_v2_20260602.json \
  --seeds 2     # serial, OMP_NUM_THREADS=1
```

- Control `A_golden` = golden + regime-admission gate disabled (see §4).
- Treatment `B_sigma_horizon_60` = control + `ranking.kelly_sizing.sigma_horizon_days = 60`
  (the ONLY changed key).
- `AA_golden_resplit` = control with offset seeds.
- 27-month OOS WF window (2024-01-02 → 2026-03-28).
- Evidence: `doc/research/evidence/2026-06-03-kelly-sigma-horizon-ab-admoff/`.

## 2 · Result — the treatment moved Kelly targets, nothing else

The σ-horizon change **was applied** and moved Kelly targets exactly as the
audit's math predicted (annualized σ → 60-day σ shrinks the denominator, so
`f* = μ/σ²` rises):

| Regime | Kelly held-target (A → B) | Δ |
|---|---|---|
| BULL_CALM | 0.0679 → 0.0890 | **+0.0211** |
| BULL_VOLATILE | 0.0737 → 0.1256 | **+0.0518** |
| CHOPPY | 0.0615 → 0.0693 | **+0.0078** |
| BEAR | 0.0000 → 0.0000 | 0 (no buys) |

But every portfolio outcome is **identical to the last digit**:

| Metric | A_golden | B_sigma_horizon_60 | Δ |
|---|---|---|---|
| APY | 0.155766 | 0.155766 | **+0.00e+00** |
| Sharpe | 1.542927 | 1.542927 | **+0.00e+00** |
| MaxDD | 0.061086 | 0.061086 | **+0.00e+00** |
| Calmar | 2.549947 | 2.549947 | **+0.00e+00** |

Per-regime, the same zero:

| Regime | Δ cash% | Δ n_holdings |
|---|---|---|
| BULL_CALM | +0.00e+00 | +0.00e+00 |
| BULL_VOLATILE | +0.00e+00 | +0.00e+00 |
| CHOPPY | +0.00e+00 | +0.00e+00 |
| BEAR | +0.00e+00 | +0.00e+00 |

(The sim is deterministic: `A_golden` and `AA_golden_resplit` are also
byte-identical, so the A vs B comparison is a clean controlled diff — same
data, same seeds, only `sigma_horizon_days` differs. An exact-zero delta
needs no seed variance to interpret.)

## 3 · Why — Kelly is a non-binding ceiling, not the position size

The Kelly target does not set the position size. Per the audit's own §1 and
`kernel/pipeline/task_selection.py`, the Kelly target becomes an **upper
bound** fed into the QP allocator:

```
max_pct_i = kelly_target_i × conviction × signal_mult      # an UPPER BOUND
```

The QP then optimizes the actual weights subject to `w_i ≤ max_pct_i` plus
sector / correlation / turnover / cash constraints. In this backtest the QP
allocates **below** the Kelly ceiling — so raising the ceiling (via the
σ-horizon fix) is inert. Concretely, BULL_CALM held-Kelly rose 0.068 → 0.089
but both are under the regime's `max_position_pct = 0.15`, and the QP's
chosen weight didn't move because the ceiling was never the binding
constraint on it.

**Implication**: the cash drag the operator flagged is real (cash% ≈ 66% in
BULL_CALM, 81% in BULL_VOLATILE) but is determined by what the QP /
candidate-selection does **below** the Kelly ceiling — not by Kelly
underweighting. Fixing the σ horizon cannot reduce it.

## 4 · Three infrastructure blockers hit en route (all diagnosed)

Documented so the next experiment doesn't re-pay them:

1. **Regime-admission gate → null run.** golden's `RegimeModelAdmissionTask`
   requires `wf_gate_metadata.trade_monotonicity`, absent from all 39 v2 WF
   artifacts → every candidate blocked → zero trades. Fix: disable the gate
   in the A/B configs only (identical in both arms, upstream of sizing). See
   [`2026-06-03-kelly-sigma-ab-blocked-by-admission-gate.md`](2026-06-03-kelly-sigma-ab-blocked-by-admission-gate.md)
   (PR #201).
2. **`--parallel-seeds` OOM-hang.** Parallel PatchTST workers exhausted RAM
   and died; the parent deadlocked on `join()` at 0% CPU. Fix: run serial.
   (Runner should fail loud on dead workers — follow-up.)
3. **libtorch OpenMP deadlock.** With `OMP_NUM_THREADS=14` the PatchTST
   scorer hung in `kmp_flag_64::wait → _pthread_cond_wait` (torch's nested
   OMP threadpool, stack-trace-confirmed). Fix: `OMP_NUM_THREADS=1` for the
   torch sim path. **This contradicts CLAUDE.md §6.5's blanket "saturate
   with OMP_NUM_THREADS=14"** — that guidance deadlocks torch inference on
   macOS. §6.5 should carry a torch caveat.

## 5 · What this means for the cash-drag investigation

- **Drop** the audit's σ-horizon recommendation as a cash-drag remedy. The
  fix is mechanically correct (and harmless), but it does not move the
  portfolio in the current architecture, so it should not be promoted on a
  cash-drag rationale. If shipped at all, ship it as a correctness cleanup
  with the explicit note that it is portfolio-neutral.
- **Redirect** the cash-drag root-cause hunt to the actual binding
  constraint: why does the QP allocate ~66–81% to cash when Kelly ceilings
  and `max_position_pct` both permit far more? Candidates to audit next:
  the QP objective / risk-aversion (`kappa`), `cash_reserve_pct`, the
  per-name `max_position_pct` interaction with candidate count, and the
  `cash_drag_lambda` soft penalty. The cash-overlay feasibility study
  ([`2026-06-03-cash-overlay-feasibility-study.md`](2026-06-03-cash-overlay-feasibility-study.md))
  is the more promising lever, since it deploys idle cash directly rather
  than relying on Kelly to size up.

## 6 · Validity caveats

- 2-seed range-finding pass, not a 5-seed promotion run — but the effect is
  **exactly zero**, so more seeds cannot reveal a hidden effect (there is no
  variance to average down; the deltas are 0.0, not noisy).
- Admission gate disabled, so absolute APY/Sharpe here are not the prod
  numbers — but the A-vs-B **difference** is what matters, and the gate is
  identical in both arms.
- §7.2 placebo battery not run: unnecessary here. Placebos guard against
  false POSITIVE lift; this result is a true zero, so there is no lift to
  placebo-test.

## 7 · Verdict

**REJECT** the σ-horizon change as a cash-drag fix. Tier-3 not reached
(`tier3_ready = false`) — correctly, because ΔSharpe = 0. The audit's
sizing math is right; its causal claim ("Kelly underweighting drives the
cash drag") is **falsified** by this controlled diff. Kelly is a
non-binding ceiling; the cash sits below it for reasons the QP / selection
layer owns.

---

Agent-Origin: Claude
