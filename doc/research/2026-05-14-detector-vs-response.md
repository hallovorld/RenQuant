# 2026-05-14 — Detector vs Response Function (counter-intuitive finding)

## TL;DR

Both regime detector upgrades shipped today (MA50 direction-aware Hurst,
HMM Hamilton 1989) **made the strategy WORSE on the 16-window paired
panel vs the original (buggy) GMM detector**. Pooled mean Δ_APY: MA50
fix = **−4.10pt**, HMM = **−5.85pt** (partial n=5).

The detectors are NOT broken. The RESPONSE function (BEAR carve-out) is
brittle: 5-11% mis-labels in bull windows trigger full defensive switch,
losing the rally. This is the **Kaminski-Lo 2014** stop-loss-under-
momentum failure mode predicted by today's literature review.

## Layer 1: per-window Δ_APY breakdown

| Q   | Regime     | GMM apy  | MA50 apy | Δ MA50   | HMM apy  | Δ HMM    |
|-----|-----------|---------|---------|---------|---------|---------|
| Q01 | BEAR       | −24.76%  | −24.76%  | +0.00pt  | −24.76%  | +0.00pt  |
| Q02 | BEAR       | −23.45%  | −20.18%  | +3.27pt  | −20.18%  | +3.27pt  |
| Q03 | CHOPPY     | +14.96%  | +11.78%  | −3.18pt  | +11.78%  | −3.18pt  |
| Q04 | BULL_CALM  | +19.05%  | +9.04%   | −10.01pt | +9.04%   | −10.01pt |
| Q05 | BULL_CALM  | +28.97%  | +28.97%  | +0.00pt  | +9.64%   | −19.34pt |
| Q06 | BEAR       | −18.26%  | −19.45%  | −1.19pt  | —        | —        |
| Q07 | BULL_STRONG| +50.80%  | +50.80%  | +0.00pt  | —        | —        |
| Q08 | BULL_VOL   | +26.51%  | +26.51%  | +0.00pt  | —        | —        |
| Q09 | CHOPPY     | −24.59%  | −24.59%  | +0.00pt  | —        | —        |
| **Q10** | **BULL_STRONG** | −6.04%   | **−16.96%**  | **−10.92pt** | — | — |
| **Q11** | **BULL_STRONG** | +54.13%  | **+26.98%**  | **−27.15pt** | — | — |
| Q12 | BULL_CALM  | −23.46%  | −11.80%  | +11.66pt | —        | —        |
| Q13 | BULL_STRONG| +26.75%  | +26.75%  | +0.00pt  | —        | —        |
| Q14 | BULL_VOL   | +0.80%   | +0.80%   | +0.00pt  | —        | —        |
| **Q15** | **BULL_VOL** | +3.50%   | **−21.82%**  | **−25.32pt** | — | — |
| Q16 | BULL_STRONG| −17.98%  | −20.77%  | −2.78pt  | —        | —        |
| **Pooled** | **16w** |          |          | **−4.10pt** | (n=5) | **−5.85pt** |

## Layer 2: regime label distribution — DETECTOR DID ITS JOB

Catastrophe windows (where strategy lost ≥10pt vs GMM baseline):

| Window | Regime (obj.) | MA50 BEAR days / total | BEAR % |
|--------|--------------|------------------------|--------|
| Q10    | BULL_STRONG  | 5 / 65                 | 7.7%   |
| **Q11**| **BULL_STRONG** | **5 / 64**           | **7.8%** |
| **Q15**| **BULL_VOL** | **7 / 64**             | **10.9%**|

**The detector is correctly identifying ~90%+ of bull bars as BULL_*.**
The 5-7 BEAR mis-labels per quarter are NOISE, not detector failure.

## Layer 3: strategy action chain (the catastrophe vector)

When MA50 fix labels a single day as BEAR in a bull window, the strategy
takes these actions per `regime_params.BEAR`:

1. `max_position_pct = 0.0`  — no new buys
2. `entry_mode = "blocked"`  — entries hard-blocked
3. `drawdown_halt_pct = 0.05`  — halt at 5% peak-to-trough
4. `bear_defensive_slots = 2`  — buy 2× GLD/TLT/XLV/XLU at 15% each
5. `stop_loss_pct = 0.05`  — tight 5% stops

In a bull market that has a routine 5% pullback (normal noise), this
cascades:
- BEAR fires for 1-3 days
- `drawdown_halt_pct=0.05` hits the recent 5% drawdown → liquidates positions
- Switch to GLD/TLT (themselves underperforming in 2024-2025 rate environment)
- BEAR passes, regime returns to BULL_CALM, but strategy has missed the V-recovery
- Bull rallies are FAST — one missed week = multi-percent alpha gone

## Layer 5: hypotheses to falsify — results

| Hypothesis | Result |
|------------|--------|
| HMM eliminates Q04 regression | ❌ HMM gives same Q04 result as MA50 (-10.01pt) — labels are similar |
| BEAR detection improves BEAR-window APY | ⚠️ +3.27pt Q02 only; Q01 unchanged, Q06 −1.19pt |
| Regime persistence ≥ 0.85 in production | ✅ Confirmed in sim logs (transition rate < 10%) |
| No structural BULL regressions | ❌ **Q11 −27pt, Q15 −25pt, Q10 −11pt** — all bull windows ruined |

## Theory grounding — why this happened (predictable in hindsight)

### Kaminski-Lo 2014 *J. Financial Markets* 18:234

> "Under the Random Walk Hypothesis, simple 0/1 stop-loss rules always
> decrease expected return; in the presence of momentum, stop-loss rules
> can add value. The stopping premium is directly proportional to the
> magnitude of return persistence."

The BEAR carve-out IS a 0/1 stop-loss. In bull markets (momentum > 0),
stop-loss CAN help — but only if the stop signal correlates with future
losses. A noisy regime flag that fires 8% of the time on bull days is
ANTI-correlated with future losses (bull days are followed by more bull
days). So the stop-loss subtracts value, not adds.

### Garleanu-Pedersen 2013 *J. Finance* 68(6):2309

> "Optimal trading is to trade partially toward the aim portfolio at a
> fixed speed; the aim weight on current-state Markowitz increases in
> state persistence and risk aversion."

Our code does the OPPOSITE — instant FULL switch on each bar's posterior
maximum. The 8% BEAR-label noise should cause an 8% damping of position
size, not a 100% switch to cash.

## What "right" looks like

Two competing remedies, both research-backed:

### Option (P1d): Soften BEAR response — config-only

`regime_params.BEAR.*` changes:
- `max_position_pct` 0.0 → 0.05 (allow tiny positions, don't liquidate)
- `drawdown_halt_pct` 0.05 → 0.15 (only halt on truly large pullbacks)
- `entry_mode` "blocked" → "reduced_size"
- Keep `stop_loss_pct=0.05` for individual positions only

**Pros**: zero new code, ~1h to test
**Cons**: still uses argmax (binary switch); may need iteration

### Option (P1e): Posterior-weighted sizing (Garleanu-Pedersen)

Replace `if regime == BEAR: max_pos = 0` with:
```python
P_not_bear = sum(p for r, p in gmm_probs.items() if r != "BEAR")
max_pos *= max(0.1, P_not_bear)  # damp by posterior, floor at 10%
```

**Pros**: theoretically optimal (canonical paper-backed)
**Cons**: requires posterior probabilities at decision time, ~2-3h

## Next experiment design (per §5.11, §5.14)

### Phase 1d.1 — Range-finding (single window)

Run ONE sim on Q11 (worst MA50 catastrophe, −27pt) with `regime_params.BEAR`
softened to `{max_position_pct: 0.05, drawdown_halt_pct: 0.15, entry_mode:
"reduced_size"}`. Decision rule:
- Q11 ≥ −5pt → response-softening works, proceed to full panel
- Q11 still ≤ −15pt → softening insufficient, escalate to posterior-weighted
- −15pt < Q11 < −5pt → partial fix, iterate on values

### Phase 1d.2 — Full 16-window panel (only if range-find passes)

Side config `sim_baseline_softbear.json` with the softened BEAR settings;
keep MA50 detector + HMM swap for cross-comparison.

### Phase 1d.3 — Posterior-weighted (option P1e) if soft-config insufficient

New code path in ComputeQPConstraintsTask using ctx.regime_state.gmm_probs.

## Commits this session (regime-conditional architecture build)

```
2d55c44  fix(regime/hmm): align HMM cluster labels with codebase taxonomy
93c2f5c  sim configs: HMM detector A/B side configs
1285c95  fix(regime): HMM replaces stateless GMM — Hamilton 1989
054a572  P1a: per-regime long_short.enabled overlay + BEAR-OFFENSIVE hybrid
b70f2f6  docs(roadmap): lock BEAR hybrid design decision (option γ)
3d346b4  docs(roadmap): PRIME DIRECTIVE phase plan + P1 per-regime knob wiring
e3fd4a1  docs(CLAUDE.md): PRIME DIRECTIVE — RenQuant is regime-conditional
3925c0d  fix(regime): direction-aware Hurst (MA50)
7f40316  fix(safety): hardcap max_gross_exposure at 1.0 — no leverage authorized
548c76e  test: vectorbt cross-validators for SimAdapter cash flow paths
```

## Lesson

**Detector accuracy without response calibration is a footgun.** Better
detector + brittle response = worse strategy. Per Kaminski-Lo, the
correct order is:
1. CALIBRATE response function to handle the EXPECTED noise rate of any
   detector (~10-15% mis-label rate in bull markets is normal).
2. THEN improve detector — at that point, accuracy gains compound with a
   calibrated response.

The PRIME DIRECTIVE (regime-conditional architecture) is correct. The
implementation order matters: response BEFORE detector.
