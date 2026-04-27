# Boyd-style Convex Rotation — Design Plan (A1 from Tier 2 roadmap)

**Status**: Plan / not yet implemented (2026-04-27).
**References**:
- Boyd, "Markowitz Portfolio Construction at Seventy" (2024 / Markowitz tribute)
- Gârleanu & Pedersen 2013 ("Dynamic Trading with Predictable Returns and Transaction Costs", J. of Finance) — claim: +20% Sharpe vs static one-period optimization via "aim in front of the target + trade partially toward aim"
- `doc/components/portfolio-qp.md` — current Phase 2 greedy-joint-actions sorter implementation
- `doc/roadmap.md` Tier 2 #T2-4

**Expected impact**: +20% Sharpe (G-P empirical claim).

---

## Why this matters

Today's rotation algorithm (`kernel/rotation.py`) is a **greedy joint-action sorter**: it ranks (sell, buy) pairs by `panel_advantage` + `kelly_advantage`, takes the top K, and emits them as orders. This is one-period myopic — it ignores:
- Slow-decay vs fast-decay signal asymmetry (G-P 2013 §III.B)
- Transaction-cost amortization across multi-day signals
- Joint correlation structure between candidate trades (only pairwise sector + correlation guard)

A Boyd-style convex program directly maximizes a **single-period mean-variance utility** with explicit transaction-cost penalty:

```
maximize    μᵀΔw  −  γ · Δwᵀ Σ Δw  −  c · ‖Δw‖₁
Δw

subject to  weight_lo ≤ w + Δw ≤ weight_hi
            ‖Δw‖₁ ≤ turnover_cap
            ‖w + Δw‖₁ ≤ leverage_cap
            sector_caps respected
            correlation guards respected
```

Where:
- `Δw` = trade vector (per-ticker buy/sell quantity, signed; in fractional weight units)
- `μ` = expected returns from NGBoost head (per-ticker μ for the 5-day horizon)
- `Σ` = correlation matrix from `watchlist-correlation.json` (annualized vol-weighted)
- `γ` = risk aversion parameter (operator-tuned)
- `c` = transaction cost coefficient (slippage + tax estimate)

G-P 2013's "aim in front of target" insight: the optimal `Δw` is NOT to fully rebalance to the static-mean target each period. Instead, it's a fraction `(1 − e^{-α·dt})` of the move, where `α` depends on signal decay rate. Slow-decay signals (low-frequency factors) → bigger trades; fast-decay signals (high-frequency reversals) → smaller trades.

---

## Why convex (not MILP)

- **Pure convex** (cvxpy / OSQP solver): solve time ~50-200ms for 99-ticker problem. Live-friendly.
- **MILP (integer share constraints)**: solve time can blow up to seconds or minutes. Not live-friendly.
- **Compromise**: solve as convex (continuous fractional weights), then round to whole shares with a heuristic (largest-fractional-first, respecting cash). Loses ~0.5% of theoretical lift but keeps inference fast.

---

## Implementation plan (4 phases)

### Phase A — `kernel/rotation_convex.py` (new module, ~400 LoC)

```python
import cvxpy as cp
import numpy as np
import pandas as pd

class ConvexRotationSolver:
    """Boyd-style mean-variance optimizer with transaction-cost penalty.

    Solves the single-period rebalancing problem each bar. Outputs a
    Δw vector that the rotation pipeline converts into BUY/SELL orders
    with whole-share quantization.
    """

    def __init__(self, *, gamma_risk: float = 5.0,
                 cost_coef: float = 0.001,
                 turnover_cap: float = 0.40,
                 leverage_cap: float = 1.0,
                 sector_max_pct: float = 0.30,
                 correlation_max: float = 0.70,
                 solver: str = "OSQP"):
        ...

    def solve(self, *,
              current_weights: pd.Series,           # by ticker, sums to ≤ 1
              expected_returns: pd.Series,           # by ticker, μ from NGBoost
              cov_matrix: pd.DataFrame,              # ticker × ticker, σ²-scaled
              sector_map: dict[str, str],
              ) -> pd.Series:
        """Returns Δw (signed, in fractional weight units)."""
        n = len(current_weights)
        delta = cp.Variable(n)
        w_new = current_weights.values + delta

        # Objective: maximize μᵀΔw − γ Δwᵀ Σ Δw − c ‖Δw‖₁
        objective = cp.Maximize(
            expected_returns.values @ delta
            - self.gamma_risk * cp.quad_form(delta, cov_matrix.values)
            - self.cost_coef * cp.norm(delta, 1)
        )

        constraints = [
            w_new >= 0,                                      # long-only
            cp.sum(w_new) <= self.leverage_cap,
            cp.norm(delta, 1) <= self.turnover_cap,
        ]
        # Per-sector caps
        for sector in set(sector_map.values()):
            mask = np.array([sector_map.get(t) == sector for t in current_weights.index])
            constraints.append(cp.sum(cp.multiply(mask, w_new)) <= self.sector_max_pct)

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=self.solver, verbose=False)
        if prob.status != "optimal":
            log.warning("ConvexRotationSolver: %s — falling back to greedy", prob.status)
            return None  # caller falls back to current greedy sorter

        return pd.Series(delta.value, index=current_weights.index)
```

**Tests**:
- Trivial 2-asset problem with known closed-form solution → solver matches within ε
- Solve time < 500ms for 99-ticker problem (CI gate)
- Infeasible problem (constraints conflict) → graceful None return
- Solver OSQP vs SCS vs ECOS — pick fastest with consistent results

### Phase B — Whole-share quantization

```python
def quantize_to_whole_shares(
    delta_weights: pd.Series, prices: pd.Series,
    portfolio_value: float, available_cash: float,
) -> pd.Series:
    """Convert fractional Δw into integer share counts respecting cash budget.

    Heuristic: process by largest |Δw| first; greedy round-up if cash
    permits, round-down otherwise. Track running cash; reject any trade
    that would overdraw.
    """
    notional_delta = delta_weights * portfolio_value
    share_delta = (notional_delta / prices).round().astype(int)

    # Sort by |notional_delta| descending; allocate cash one trade at a time
    cash = available_cash
    out = pd.Series(0, index=delta_weights.index, dtype=int)
    for ticker in notional_delta.abs().sort_values(ascending=False).index:
        s = share_delta[ticker]
        notional = s * prices[ticker]
        if notional > 0 and notional > cash:
            # Buy that doesn't fit — reduce shares to what cash allows
            s = int(cash / prices[ticker])
            notional = s * prices[ticker]
        if notional < 0 and -notional > 0.99 * cash * (-1):  # selling — always OK
            pass
        out[ticker] = s
        cash -= notional
    return out
```

### Phase C — Wire into pipeline

`kernel/rotation.py::RotationJob` gains a config branch:

```python
solver_kind = config["rotation"]["joint_actions"]["solver"]   # "greedy" | "convex"
if solver_kind == "convex":
    from kernel.rotation_convex import ConvexRotationSolver
    solver = ConvexRotationSolver(**config["rotation"]["convex"])
    delta = solver.solve(...)
    if delta is None:
        # Solver failed (infeasible / numeric issue) → fall back to greedy
        log.warning("Convex solver returned None; falling back to greedy")
        solver_kind = "greedy"
    else:
        share_delta = quantize_to_whole_shares(delta, ...)
        # Convert share_delta → ExitSignal (negatives) + buy intentions (positives)
        ...
if solver_kind == "greedy":
    # existing greedy joint-action code
    ...
```

Config:
```json
"rotation": {
  "joint_actions": {
    "solver": "greedy",   // default — convex requires explicit opt-in
    "max_actions": 4
  },
  "convex": {
    "_doc": "A1 Boyd-style mean-variance optimizer (off by default)",
    "gamma_risk": 5.0,
    "cost_coef": 0.001,
    "turnover_cap": 0.40,
    "leverage_cap": 1.0,
    "sector_max_pct": 0.30,
    "correlation_max": 0.70,
    "solver": "OSQP"
  }
}
```

### Phase D — Acceptance + sim validation

- New gate G13 (soft): `solve_time_ms < 500`. Catches solver regressions.
- A/B sim: convex vs greedy on 27-month OOS, measure APY + Sharpe + turnover.
- G-P paper claims +20% Sharpe; we'd accept anything ≥ +5% Sharpe with non-degraded APY.
- If accepted, promote to default in golden config + ntfy alert when convex falls back to greedy in live.

---

## Risk + mitigation

| Risk | Mitigation |
|---|---|
| Solver becomes slow (>500ms per bar) | G13 gate + automatic fallback to greedy |
| OSQP infeasible on degenerate inputs | Try SCS fallback then greedy |
| Theoretical +20% doesn't materialize | A/B sim must clear ≥+5% Sharpe before promote |
| Whole-share quantization erodes lift | Measure post-quantization Sharpe (not pre-) |
| Cost coefficient `c` mis-tuned | Sweep `c ∈ [0.0005, 0.005]` in A/B; pick maximum APY config |
| cvxpy memory leak (known issue) | Wrap solve in `try/finally` with explicit `del` |

---

## Effort estimate

| Phase | Effort | Risk |
|---|---|---|
| A — solver module + tests | 2 days | Medium (cvxpy API) |
| B — whole-share quantizer | 0.5 day | Low |
| C — pipeline wiring | 1 day | Medium (rotation logic is delicate) |
| D — A/B + acceptance | 1 day | Low (compare sim results) |
| Total | ~4.5 days | Medium overall |

---

## Open decisions

1. **Solver**: OSQP (fastest, may have edge cases) vs SCS (more robust, ~3× slower) vs ECOS (most accurate, slowest). Start OSQP, fall through chain if needed.
2. **Risk aversion γ**: 5.0 is a typical default; A/B sweep `γ ∈ {1, 3, 5, 10, 20}`.
3. **Correlation matrix update freq**: weekly via `watchlist-correlation.json` already; sufficient.
4. **Multi-period horizon (G-P 2013 strict form)**: full G-P uses infinite-horizon optimization with signal decay rates per factor. Phase A above is single-period — captures most of the lift but not all. Multi-period is Phase E (future).

---

## What this does NOT do

- Does not implement G-P's "aim in front of the target" multi-period form (single-period is the 80/20).
- Does not change buy/sell candidate generation — the QP only chooses *how* to rebalance among already-scored candidates.
- Does not replace the buy gates (Gate B edge-Sharpe floor, etc.) — those remain as filters BEFORE the QP runs.
