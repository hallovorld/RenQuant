# Rotation Algorithm — Design + References

**Last updated**: 2026-05-20

> **2026-05-20 update**:
> - Active rotation: `joint_actions.enabled=true` + `solver="qp"` (cvxpy + CLARABEL) — Phase 2 IS the default now, NOT opt-in
> - Phase 3 Boyd MPC: SHIPPED via `kernel/portfolio_qp/cvxportfolio_backend.py` (toggle `qp_solver_backend="cvxportfolio"`)
> - Current golden thresholds: `panel_buy_floor=0.30`, `panel_sell_floor=0.20`, `min_expected_advantage_pct=0.06`, `min_rotation_hold_days=7`, `max_rotations_per_bar=2`
> - HIFO lot-selection: `joint_actions.qp_tax_lot_method=hifo` (2026-05-17, was FIFO)
> - Anti-churn `min_reentry_days=5` (2026-05-18) compounds on §1091 wash-sale in `is_wash_sale_blocked` path
> - Legacy 3-pass greedy `JointActionTask` (700 lines) is fallback only (`solver="greedy"`)

## Purpose

The rotation algorithm decides whether a held position should be exited
in favour of a stronger candidate during the same bar — the central
"smart" component of the renquant_104 trading model. User quote (2026-04-25):
*"rotate algo是我的核心资产"*. This document describes the design,
formulation, fixes, and academic references.

## Two-Phase Architecture

| Phase | Module | Purpose | Default |
|-------|--------|---------|---------|
| **Phase 1 — Pairwise** | `kernel/pipeline/task_rotation.py` | Find swap pairs via per-pair net advantage; ER-based selection with thesis/Kelly/symmetric variants. | on (fallback) |
| **Phase 2 — JointAction (QP)** | `kernel/pipeline/task_joint_actions.py` + `kernel/portfolio_qp/` | Unified BUY/SELL/ROTATE Δw vector via cvxpy CLARABEL convex QP; HIFO lot accounting; min_share_floor. | **DEFAULT ON since 2026-05-07** (`solver="qp"`) |
| **Phase 3 — Boyd MPC (cvxportfolio)** | `kernel/portfolio_qp/cvxportfolio_backend.py` | Full `cvxportfolio.SinglePeriodOpt` reference policy with Boyd/Stanford idioms. | opt-in via `qp_solver_backend="cvxportfolio"` |

When the Phase 2 flag is OFF (current default), the legacy chain runs:
`RotationJob → SelectionJob → SizeAndEmitTask → TopUpHeldTask`. When ON,
`JointActionJob` replaces both `RotationJob` and `SelectionJob`.

## Phase 1 — Pairwise Rotation (`task_rotation.py`)

### Decision Tree

For each candidate × held pair:

1. **Score gate (Phase 1 score-double-gate)**:
   - `cand.rank_score >= panel_buy_floor` (default 0.45)
   - `held.rank_score <= panel_sell_floor` (default 0.20)
2. **Held eligibility**: `hold_days >= min_rotation_hold_days` (30); not LT-protected.
3. **Net advantage** (ER mode):
   ```
   net_adv = (cand.ER − held.ER) − transaction_cost − tax_drag(held)
   ```
   where `tax_drag = unreal_pnl × tax_rate(hold_days, ST/LT)`.
4. **Threshold**: `net_adv >= min_expected_advantage_pct` (3%).

### Multiple Modes

The `rotation.mode` config dispatches to different finder primitives in
`kernel/rotation.py`:

- `"er"` (default): ER-based with min_expected_advantage_pct gate
- `"thesis_primary"`: 2-point thesis comparison (held entry vs today)
- `"thesis_symmetric"` (V4): 4-point comparison (A_entry, A_today, B_entry, B_today),
  uses DB lookup for B's score on A's entry date
- Optional Kelly-delta gate, panel-score gate, thesis-degradation gate

### Validation Phase

`ValidatePairsTask` re-checks each surviving pair against:
- `is_wash_sale_blocked` on cand_ticker
- `passes_sector_guard` on virtual post-swap holdings
- `passes_correlation_guard` on virtual post-swap holdings (corr threshold 0.70)

### Emission Phase (`EmitRotationsTask`)

Atomic execution: build buy-leg fully BEFORE committing exit. Pre-fix
(2026-04-24), if buy-leg failed (Kelly=0, bad price, no cash), the
position closed without replacement. Post-fix: skip ENTIRE pair on any
buy-leg failure, with `ctx.rotations_blocked` capturing the reason for
operator visibility (ROT-BLOCKED-NTFY).

**PR1-CASH fix (2026-04-25)**: `cash_remaining` rolls forward across
multiple rotations within `max_rotations_per_bar`, AND credits the
held's mark-to-market value as sell-leg proceeds for buy-leg sizing
(RegT same-bar settlement on paired orders).

## Phase 2 — JointActionTask (`task_joint_actions.py`)

### Algorithm

Build unified action menu (BUY / SELL / ROTATE), filter, sort by
`net_alpha` desc, run **3-pass greedy** with budget enforcement.

```
Pass 1 (SELLs):
  for each SELL action in sorted order:
    if held has higher-net ROTATE candidate AND portfolio ≤ max:
      defer to Pass 3
    else:
      accept SELL → exits, free slot

Pass 2 (BUYs + ROTATEs):
  for each non-SELL action in sorted order:
    if budget exceeded:
      block
    if guards fail (wash, sector, corr, tier):
      block
    size with sell-leg credit (ROTATE) → orders, decrement cash

Pass 3 (deferred SELLs):
  for each deferred SELL:
    if held still un-used (rotation didn't fire):
      accept SELL retroactively
```

### Score-Floor Semantics

Per user 2026-04-25 spec: *"被替换的 portfolio 里的 stock 的 score 要低于一个值，进到 portfolio 的 stock 的 score 要高于一个值"*.

- BUY actions: `cand.rank_score >= panel_buy_floor` (necessary condition)
- SELL actions: `held.rank_score <= panel_sell_floor` (necessary condition)
- ROTATE actions: BOTH gates pass

### Net-Alpha Filter (Bug Q)

Pre-greedy filter: `BUY/ROTATE actions with net_alpha ≤ 0 are dropped`
because they're guaranteed losses after fees+slippage+tax. SELL actions
are EXEMPT — score-driven exit per user spec; net_alpha sign determines
priority but not eligibility.

### Slot Budget (Bug F + Bug Y)

Capacity constraint: `len(virtual_held_after_pass2) ≤ max_concurrent_positions`,
equivalent to `net_position_consumed ≤ open_slots`.

`open_slots = max_positions − len(effective_held)`. May be negative
(over-filled by external path) — formula still holds; over-filled
portfolio can only BUY when prior sells outpace the overflow.

ROTATEs are net-zero on `net_position_consumed`, capped separately by
`max_rotations_per_bar`.

### Pass-1-vs-Pass-2 Conflict Resolution (Bug MM)

When a held has BOTH a SELL action AND a ROTATE action, Pass 1 SELL
fires before Pass 2 sees ROTATE → ROTATE dedup'd. To avoid losing
rotation alpha, defer SELL to Pass 3 if:
- A ROTATE for same held has higher `net_alpha` AND
- `len(virtual_held) ≤ max_positions` (otherwise SELL is needed to
  reduce overfill)

If the deferred ROTATE doesn't materialise (cash, sector, corr fail),
Pass 3 fires the SELL retroactively — never losing a legitimate exit
signal.

This is the **pairwise greedy with dominance pruning** heuristic. True
joint optimum requires Boyd-style QP (Phase 3).

### Cash Accounting (Bug M)

ROTATE buy-leg sizing credits the held's mark-to-market value:
```
cash_for_sizing = cash_remaining + held_shares × held_price × (1 − fees)
```

Without this fix, swapping a $20k held for a $20k cand with $1k cash
on hand would size the buy-leg at ~5 shares = $1k worth — losing $19k
of signal.

## Bugs Fixed (2026-04-25 batch)

| Tag | Severity | Description |
|-----|---------:|-------------|
| **JOINT-NET-POSITIONS (F)** | 🔴 critical | `slot_budget = open_slots + max_rot_bar` allowed BUYs to over-fill past max when rotations didn't materialize |
| **JOINT-OVERFILL-EDGE (Y)** | 🔴 critical | `max(open_slots, 0)` clamp let overfilled state persist after sells |
| **JOINT-ROTATE-CASH (M)** | 🔴 critical | rotate's buy-leg sized off cash without crediting sell-leg proceeds |
| **JOINT-NET-NEG (Q)** | 🟠 high | actions with net_alpha ≤ 0 (lose money after fees) accepted |
| **JOINT-TIER-NEGATIVE (S)** | 🟠 high | tier_idx wrap on negative slots_consumed |
| **JOINT-GREEDY-SELL-LATE (L)** | 🟠 high | sells sorted last → blocked BUYs can't see freed slots |
| **JOINT-PASS1-SELL-VS-ROTATE-CONFLICT (MM)** | 🟠 high | SELL Pre-empted ROTATE dominance |
| **JOINT-ROT-QUOTA-ZERO (edge MM)** | 🟡 med | Bug MM stranded SELL when max_rot_bar=0 |
| **PR1-CASH** | 🔴 critical | Phase 1 EmitRotations sized each pair off bar-start cash |
| **JOINT-CORR-NONE (D)** | 🟡 med | crash on missing watchlist-correlation.json |
| **JOINT-TIER-ESC (C)** | 🟡 med | tier_idx=0 used for all slots in joint mode |
| **JOINT-SLOT-SHARE (B)** | 🟡 med | rotate didn't consume shared slot budget |
| **JOINT-PRUNE-USED-HOLDS (DD)** | 🟢 low | TopUpHeldTask could re-buy a sold ticker |

All ship with regression tests in `tests/test_joint_actions.py` (30 tests)
and `tests/test_rotation_atomic.py`.

## References — Foundations

### Joint Multi-Period Trading

1. **Garleanu N., Pedersen L. H. 2013.** "Dynamic Trading with Predictable
   Returns and Transaction Costs." *Journal of Finance* 68 (6): 2309–2340.
   Closed-form aim portfolio: trade toward target, partial-execution
   damped by transaction costs. Foundation for "rotation as continuous
   trade size" rather than discrete swap.

2. **Boyd S., Busseti E., Diamond S., Kahn R., Koh P., Nystrup P., Speth J.
   2017.** "Multi-Period Trading via Convex Optimization." *Foundations
   and Trends in Optimization* 3 (1): 1–76. The cvxpy formulation — what
   Phase 3 will implement. Open-source code: cvxportfolio.

3. **Almgren R., Chriss N. 2000.** "Optimal Execution of Portfolio
   Transactions." *Journal of Risk* 3 (2): 5–39. Intraday execution
   schedule given a target trade — orthogonal to our strategy-level
   rotation but referenced for completeness.

### Rebalance Heuristics

4. **Markowitz H. 1952.** "Portfolio Selection." *J. Finance* 7 (1): 77–91.
   Single-period MV optimization — gives target weights; transition is
   left to a separate scheduler.

5. **Hartmann F., Marian P. 2010.** "Online Portfolio Rebalancing under
   Transaction Costs." Heuristic-side literature — proposes pairwise
   swap greedy and net-trade greedy when QP is infeasible. Our 3-pass
   joint mode = pairwise swap greedy with dominance pruning.

### Pair-Trading Foundations (rotation_mode = thesis_symmetric)

6. **Avellaneda M., Lee J.-H. 2010.** "Statistical Arbitrage in the U.S.
   Equities Market." *Quantitative Finance* 10 (7): 761–782. Foundation
   for symmetric pair comparison (A_entry, A_today, B_entry, B_today).

7. **Gatev E., Goetzmann W. N., Rouwenhorst K. G. 2006.** "Pairs Trading:
   Performance of a Relative-Value Arbitrage Rule." *Review of Financial
   Studies* 19 (3): 797–827. Empirical validation of pair convergence.

### Tax-Aware Trading

8. **Constantinides G. M. 1984.** "Optimal Stock Trading with Personal
   Taxes." *Journal of Financial Economics* 13 (1): 65–89. ST/LT tax-rate
   asymmetry → defer realisation of gains until LT threshold. Source of
   `lt_protection_days` logic in `is_lt_protected`.

9. **Dammon R. M., Spatt C. S., Zhang H. H. 2001.** "Optimal Consumption
   and Investment with Capital Gains Taxes." *Review of Financial Studies*
   14 (3): 583–616. Proves tax drag should enter rotation net-advantage
   directly — basis for `tax_drag(unreal_pnl, hold_days, ST_rate, LT_rate)`.

### Wash Sale + Regulatory

10. **IRS § 1091** ("Loss from wash sales of stock or securities"). 30-day
    pre/post-window. Implemented in `is_wash_sale_blocked`.

### Multi-Asset Sizing (Kelly)

11. **Kelly J. L. 1956.** "A New Interpretation of Information Rate."
    *Bell System Technical Journal* 35 (4): 917–926. Kelly criterion
    foundation: `f* = μ/σ²` for continuous Gaussian return.

12. **Thorp E. O. 2006.** "The Kelly Criterion in Blackjack, Sports
    Betting, and the Stock Market." *Handbook of Asset and Liability
    Management*. Half-Kelly for finite-sample variance — basis for
    `kelly_sizing.fractional = 0.50`.

### Score Calibration (Used in panel_buy_floor / panel_sell_floor)

13. **Platt J. C. 1999.** "Probabilistic Outputs for Support Vector
    Machines and Comparisons to Regularized Likelihood Methods." *Adv.
    in Large-Margin Classifiers*. Sigmoid calibration for n < 300.

14. **Zadrozny B., Elkan C. 2002.** "Transforming Classifier Scores into
    Accurate Multiclass Probability Estimates." *KDD '02*. Isotonic
    regression for n ≥ 300 — used by `panel-rank-calibration.json`.

## Configuration Reference

```jsonc
{
  "rotation": {
    "enabled": true,
    "mode": "er",                      // "er" | "thesis_primary" | "thesis_symmetric"
    "min_expected_advantage_pct": 0.03,
    "target_horizon_days": 60,
    "transaction_cost_pct": 0.0,
    "min_rotation_hold_days": 30,
    "lt_protection_days": 30,
    "max_rotations_per_bar": 2,

    // Phase 1 score double-gate (2026-04-25)
    "panel_buy_floor": 0.45,
    "panel_sell_floor": 0.20,

    // Phase 2 joint mode (2026-04-25)
    "joint_actions": {
      "enabled": false,                // flag-gated; default off
      "fee_pct": 0.0005,
      "slippage_pct": 0.0005,
      "slot_budget_mode": "shared"
    },

    // Optional gates
    "scoring_mode": "er",              // "er" | "mu_minus_lambda_sigma" | "sharpe"
    "lambda_": 1.0,
    "sharpe_sigma_floor": 1e-4,
    "enabled_regimes": null,           // [list] to restrict; null = all non-BEAR
    "persistence_bars": 0
  },

  "ranking": {
    "panel_scoring": {
      "rotation_advantage": 0.0        // panel-score gate
    },
    "kelly_sizing": {
      "rotation_advantage": 0.0,       // Kelly-delta gate
      "rotation_target_floor": 0.05    // skip Kelly gate if held below this
    },
    "thesis_rotation": {
      "enabled": false,
      "degradation_pct": 0.30,
      "uplift_pct": 0.10
    }
  }
}
```

## Test Coverage

- `tests/test_joint_actions.py` — 30 tests for Phase 2 (BUY/SELL/ROTATE
  menu, score floors, slot budget, two-pass, dominance pruning, cash
  credit, prune-used-holds)
- `tests/test_rotation_atomic.py` — Phase 1 atomic-pair invariants
- `tests/test_rotation_v1_gates.py` — pairwise gate enforcement
- `tests/test_rotation_v2_scoring.py` — scoring_mode dispatch
- `tests/test_rotation_v3_gates.py` — kelly + panel-score gates
- `tests/test_rotation_v4_symmetric.py` — thesis_symmetric V4 mode
- `tests/test_kelly_rotation_gate.py` — Kelly-delta gate
- `tests/test_thesis_rotation.py` — thesis-degradation gate
- `tests/test_runner_state_fixes.py` — STATE-GC + ROT-BLOCKED-NTFY

## Roadmap

### Phase 3 — Boyd Convex MPC (T2-4)

Implement the Boyd 2017 cvxpy formulation:

```python
import cvxpy as cp

x = cp.Variable(n_assets)              # target portfolio weights
trade = x - x_curr                      # trade vector

objective = cp.Maximize(
    expected_return @ x
    - gamma_risk * cp.quad_form(x, Sigma)
    - cp.norm(cp.multiply(trade_cost, trade), 1)  # L1 cost
    - tax_penalty(x_curr, x, hold_days, st_rate, lt_rate)
)

constraints = [
    cp.sum(x) == 1,
    x >= 0,                             # long-only
    x <= max_position_pct,
    sector_constraint(x, sector_map, max_per_sector),
    correlation_constraint(x, corr_matrix, threshold),
]

problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.ECOS)
```

This eliminates Bug L, Bug MM, and any other greedy artifacts.
Trade-off: solving a 99-asset QP per bar adds ~50ms latency vs ~5ms for
greedy. Acceptable for daily bar cadence.

### Optional Enhancements

- Multi-objective tradeoff (after-tax APY vs Sharpe vs max DD)
- Stochastic optimization with NGBoost μ,σ posteriors
- Online learning of cost parameters (slippage from realized fills)
