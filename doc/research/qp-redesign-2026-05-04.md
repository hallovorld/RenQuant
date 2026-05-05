# Portfolio QP Redesign — Industry-grade refactor

**User mandate (2026-05-04)**: stop the patch-sim-patch loop. Build
to industry standard before next sim.

**Goal**: after-tax APY and Sharpe optimization. Tax IS in scope; we
want the QP to solve for after-tax-optimal trades.

This doc maps every gap between our current QP and the academic /
open-source standard, with citation, sketched math, and done-criteria.

---

## Reference benchmarks

| System | Why it's the bar |
|---|---|
| **cvxportfolio** (Boyd, Busseti, Diamond, Kahn, Koh, Nystrup, Speth — Stanford 2017) | Multi-period convex portfolio with TC + holding cost + tax. Open source: github.com/cvxgrp/cvxportfolio. Cited 1500+. |
| **Almgren & Chriss (2000)** "Optimal Execution of Portfolio Transactions" | Sqrt-impact transaction cost is the academic + practitioner standard |
| **Garleanu & Pedersen (2013)** "Dynamic Trading with Predictable Returns and Transaction Costs" | Closed-form partial-move (1/(1+ψ)) with signal decay |
| **Brown & Smith (2011)** "Dynamic Portfolio Optimization with Capital Gains Taxes" | LT-bridge + lot-aware tax cost |
| **Berkin & Jeffrey (1990)** "Tax Alpha" | Empirical: tax-loss harvesting adds 1-2%/yr alpha when systematic |
| **Ledoit & Wolf (2004)** "Honey, I Shrunk the Sample Covariance Matrix" | Shrinkage estimator Σ̂ = δ·F + (1-δ)·S with optimal δ |
| **Garlappi, Uppal, Wang (2007)** "Portfolio Selection with Parameter and Model Uncertainty" | Robust μ adjustment: μ_robust = μ - κ·σ |
| **Rockafellar & Uryasev (2002)** "Conditional Value-at-Risk for General Loss Distributions" | CVaR tail-risk formulation |

---

## Current QP (single-period myopic, 2026-04-29 era)

```
min   -μ'(w+Δw) + γ·(w+Δw)'Σ(w+Δw) + κ|Δw|₁ + Σ tax_i·max(0, -Δw_i)
s.t.  bounds, cash_constraint, turnover_constraint, wash_sale_mask
```

Solver: scipy.optimize.minimize, method=SLSQP, jac=analytical, ftol=1e-9.

---

## Gap analysis (each item = backlog ticket)

### G1 — Tax cost is myopic; needs Brown-Smith dynamic ✅ SHIPPED 2026-05-04 evening

**What was wrong**: tax_cost_vec[i] = gain_pct × rate (current rate).
Doesn't know about days_to_LT.

**Reference**: Brown & Smith (2011) §3.2.

**Math**: 
```
rate(days_held) = lt_rate                           if days_held >= 365
                = st_rate + (st_rate - lt_rate)·(1 - days_to_lt/W)   if days_to_lt <= W
                = st_rate                            otherwise
```
where W = bridge window (~30 days).

**Done criteria**: 
- task_joint_qp.py uses Brown-Smith formula ✅
- unit test for each regime ☐ pending
- Holdout: positions held 350-365d should NOT be sold

### G2 — No tax-loss harvesting credit ✅ SHIPPED 2026-05-04 evening

**What was wrong**: tax_arr clamped ≥ 0 in qp_solver. Selling losers
got zero benefit even when YTD pool > 0.

**Reference**: Berkin & Jeffrey (1990).

**Math**: per-asset tax_cost can be NEGATIVE:
```
tax_cost_i = -(loss_used × st_rate / NAV / w_i)      when gain_pct < 0
                                                      and ytd_pool > 0
                                                      and loss_used = min(loss, ytd_pool)
```

**Done criteria**:
- qp_solver removed `np.maximum(tax_arr, 0.0)` clamp ✅
- task_joint_qp tracks remaining_offset_dollar ✅
- Need: ctx.ytd_realized_gain_dollar tracking across bars ☐ NOT YET (sim adapter must accumulate)
- Unit test: when ytd_pool=0, no negative tax costs ☐
- Unit test: when ytd_pool > 0, loser sells get reward ☐
- Holdout: confirms harvest events fire near year-end

### G3 — Linear transaction cost; should be Almgren-Chriss sqrt-impact ☐

**What's wrong**: `cost_kappa·|Δw|` (κ=0.0001) treats 1-share trim and
100-share liquidation as proportional. Real market impact scales with
sqrt(volume_traded / daily_volume).

**Reference**: Almgren & Chriss (2000), eqn (4.4).

**Math**:
```
TC_i(Δw) = (a + b·σ_i·sqrt(|Δw_i|·NAV/V_daily_i))·|Δw_i|
         = a·|Δw_i| + b·σ_i·|Δw_i|^1.5·sqrt(NAV/V_daily_i)
```
where a = spread/2 ≈ 0.0002, b = impact coefficient ≈ 0.05, V_daily =
20-day median dollar volume.

**Why this kills micro-trims**: marginal cost rises with size BUT
fixed component a·|Δw_i| means infinitesimal trades have ~zero
benefit per cost. Combined with smooth-tanh fixed cost (G4), the
optimizer will batch.

**Done criteria**:
- Replace linear `cost_kappa` term with two-component cost
- Need V_daily for each ticker (daily_volume × close)
- 20-day median dollar volume (already in panel data?)
- Unit test: 1% trade has higher per-unit cost than 0.5% trade
- Unit test: limit κ→0, b→∞ recovers AC formula

### G4 — No fixed-cost-per-trade; need smoothed step ☐

**What's wrong**: SLSQP gradient descent finds any Δw that improves
objective marginally; no penalty for the EVENT of trading. Real
brokerages have fixed per-trade fee ($0.005/share or $1 minimum).

**Math**: smoothed step
```
fixed_cost(Δw) = c_fix · tanh(β·|Δw|)
```
For small |Δw|, tanh ~ β·|Δw| (linear). For large, tanh saturates → c_fix.
Marginal cost is HIGHEST at zero. Discourages micro-trades.

β tuned so half-saturation is at 0.5% NAV (= the trim threshold).

**Done criteria**:
- Add `qp_fixed_cost_per_trade` config (default $0.005/share or 0.0002 of NAV)
- Add to objective + gradient
- Unit test: single big trade preferred over many small trades
- Unit test: |Δw| < 0.001 produces ~zero cost (smooth)

### G5 — Σ is raw sample covariance; need Ledoit-Wolf shrinkage ☐

**What's wrong**: full-Σ correlations from `kernel/data.py` are raw
sample correlations. Noise dominates off-diagonal entries with ~250
observations × 100+ assets. Marginal risk Σw is then noise-amplified.

**Reference**: Ledoit & Wolf (2004).

**Math**:
```
Σ̂_LW = δ·F + (1-δ)·S
```
where F = constant-correlation target, S = sample, δ = optimal
shrinkage intensity (closed-form, eqn 14 of paper).

**Implementation**: `sklearn.covariance.LedoitWolf().fit(returns)`.

**Done criteria**:
- Replace correlation_artifact build with LW
- Compare backtest: pre/post-LW for same QP
- Unit test: LW Σ has lower condition number than raw S
- Unit test: 1-asset edge case (LW degenerates correctly)

### G6 — Single-period myopic; need Garleanu-Pedersen partial move ☐

**What's wrong**: each bar QP solves "if I had to set w today forever,
what's optimal" — doesn't account for tomorrow's signal decay.

**Reference**: Garleanu & Pedersen (2013) §III, closed-form (eqn 12).

**Math**:
```
w*_t = (1/(1+ψ_t))·w*_{static,t} + (ψ_t/(1+ψ_t))·w_{t-1}
where ψ_t = ratio of trade cost to risk × signal decay
```
Equivalently: target a CONVEX COMBINATION of current and static-optimal,
weighted by signal half-life.

**Done criteria**:
- Estimate φ_decay from NGBoost μ time series autocorrelation
- Default config `qp_signal_decay = 0.85` (typical equity signal halflife ~1wk)
- Currently `qp_signal_decay=0` means full instant move (Markowitz)
- Unit test: φ→0 recovers Markowitz, φ→1 freezes weights
- Holdout: turnover should DROP with realistic φ

### G7 — Tax lot tracking; need FIFO/HIFO accounting ☐

**What's wrong**: we track only `entry_price` (averaged). Real taxes
work on tax LOTS — each buy has its own basis + holding date. Selling
is FIFO (or HIFO for tax optimization).

**Reference**: cvxportfolio TaxModel + IRS Pub 550.

**Math**: maintain list[TaxLot] per ticker, each with (basis, qty,
entry_date). On sell, choose lots to minimize gain (HIFO) or follow
FIFO (default).

**Done criteria**:
- New dataclass TaxLot, ctx.tax_lots: dict[ticker, list[TaxLot]]
- HoldingState gains a property `lots_sorted_by_basis_desc` for HIFO
- Sell logic picks specific lots (not just averaged basis)
- Brown-Smith tax_cost uses LOT-level days_held
- Unit test: HIFO produces strictly lower tax than FIFO when prices have rallied

### G8 — Re-entry blackout post-stop ☐

**What's wrong**: APP case — trailing_stop fired @ $587, same day
buy 1 share back at $587, next day +12 shares at $631. Got hit
with single_day_loss soon after. No protection against re-entering
the just-stopped name.

**Reference**: Standard practitioner protocol — wash-sale window
extends both buy and sell direction.

**Done criteria**:
- New config `risk.post_stop_reentry_cooldown_bars` default 5
- Track per-ticker `last_stop_exit_date` on HoldingState (or ctx)
- Filter candidates / rotation buys / TopUp against this
- Unit test: candidate excluded for 5 bars after trailing_stop / SDL
- Unit test: stop expiry restores eligibility

### G9 — Robust μ via Garlappi-Uppal-Wang ☐

**What's wrong**: NGBoost gives point estimate μ̂ + σ̂. Real-world
μ̂ has estimation error; ignoring it gives over-aggressive positions
on the riskiest assets (concentrated in highest-σ names).

**Reference**: Garlappi-Uppal-Wang (2007) §3.

**Math**: Replace μ in objective with μ_robust:
```
μ_robust_i = μ̂_i - κ_robust · σ̂_i
```
where κ_robust calibrates the confidence ball. Higher κ = more
conservative (penalize high-σ names harder).

**Done criteria**:
- We have `qp_robust_mu_kappa` config flag, currently default 0.0
- Set default 0.5 (1-σ confidence band on the mean)
- Unit test: robust μ < raw μ on positive-μ assets
- Holdout: positions less concentrated in high-σ tickers

### G10 — CVaR risk term ☐

**What's wrong**: Σ-only risk term penalizes variance equally
in both tails. Real portfolios care more about left-tail (loss)
than right-tail (upside variance).

**Reference**: Rockafellar-Uryasev (2002).

**Math**: replace γ·w'Σw with γ·w'Σw + λ·CVaR_α(w'r).

**Done criteria**:
- `qp_cvar_lambda` flag exists, default 0.0
- Set default 0.3 (mild tail penalty)
- Unit test: assets with negative skew get downweighted

### G11 — DD-aware γ; current formula amplifies under stress ☐

**What's wrong (subtle)**: `γ_eff = γ_base / (1 - DD/α)` rises as DD
approaches α. But this means in a drawdown, the QP becomes MORE
risk-averse → cuts positions → realizes losses → deeper DD → more
risk-averse. Procyclical. Standard prescription is opposite: in a
drawdown, EITHER hold (if you trust signal) or systematically reduce
(if you don't), but don't let γ explode.

**Done criteria**:
- Cap γ_eff ≤ 2·γ_base
- OR use Grossman-Zhou (1993) explicit DD-floor strategy
- Unit test: γ_eff doesn't explode at DD = α-ε

---

## Execution order (P0 → P3)

| Tier | Item | Why first |
|---|---|---|
| **P0 ✅** | G1 Brown-Smith tax | Direct fix to v2 disposition-effect failure |
| **P0 ✅** | G2 Loss harvesting | Same; paired |
| **P0** | G8 Re-entry blackout | The APP killer (specific case, big impact) |
| **P1** | G7 Tax lot tracking | Required for G1 to be FULLY correct |
| **P1** | G3 Almgren-Chriss TC | Kills micro-trim mathematically |
| **P1** | G4 Smoothed fixed cost | Same goal as G3 |
| **P2** | G5 Ledoit-Wolf Σ | De-noises correlation matrix |
| **P2** | G6 Garleanu-Pedersen φ | Already have flag, just enable + tune |
| **P3** | G9 Robust μ | Nice-to-have |
| **P3** | G10 CVaR | Nice-to-have |
| **P3** | G11 DD γ cap | Defensive |

**Done criteria for the WHOLE refactor**: every gap closed OR explicitly
deferred with docstring + ticket ref. All unit tests pass. Code review
done. THEN run a single B2 sim.

---

## What we DON'T do

- **No mid-refactor sims.** Sim wallclock is 70 min; refactor needs 2-3
  days; running 5 partial sims wastes ~5 hours and produces misleading
  numbers.
- **No skipping unit tests for "speed".** Every formula must have a
  numerical-correctness test (compare vs reference impl on a small case).
- **No "ship and we'll see if Sharpe goes up".** Sharpe is the FINAL
  validator after the design + tests are right.
