# Buy Logic — Three Quality Gates + Portfolio QP

**Status (2026-04-26 round-7):**
- ✅ Gate B (Edge-Sharpe floor τ=0.10) — **enabled in production**
- ✅ Sell-side Gate B (symmetric μ/σ guard for model_sell) — **enabled in production**
- ✅ Portfolio QP solver — **enabled in production**
- ⏳ Gate A (distribution floor), Gate C (no-trade band), QP Stages 2/4/5/7 — implemented, default OFF, awaiting sim verification

> **2026-05-07 update**: portfolio QP migrated from
> `scipy.optimize.SLSQP + cvxpy fallback` (700-line hand-rolled solver)
> to **cvxpy + CLARABEL primary** in the Boyd/Stanford
> `cvxportfolio.SinglePeriodOpt` idiom. Hard `min_invested_pct` floor
> became a soft `cash_drag_lambda` penalty (max(0, target − Σwp) added
> to the objective). The `qp_solver_backend = "cvxpy" | "cvxportfolio"`
> config switch lets you opt into Boyd's reference policy classes
> directly. The Almgren-Chriss impact / Brown-Smith tax / RU CVaR /
> Garlappi robust μ stages are all retained as cvxpy DCP terms.
> See [`portfolio-qp.md`](portfolio-qp.md) §0 for the new architecture
> and [`STATUS.md`](../STATUS.md) for the live-vs-research backend split.

Pre-2026-04-26 the pipeline had `panel_buy_floor = null` → any candidate with μ > fee passed. R6/R7 evidence showed buys at edge_sharpe ≈ 0.10 (~random within the panel's noise floor). This doc covers the redesign + operational runbook.

---

## 1. Theory — three gates, each grounded in literature

### Gate A — Edge-Sharpe floor (Lo 2002 / Grinold-Kahn 1999)

> *Don't trade unless predicted instantaneous Sharpe of the edge exceeds a floor.*

```
edge_sharpe = μ / σ        (NGBoost outputs)
admit(cand) ⇔ edge_sharpe > τ_S
```

Calibration: `SR_a_realized ≈ 0.5 × SR_d × √252`. So τ_S=0.20 → SR_a_real ≈ 1.6 (good); τ_S=0.10 → ~0.8 (current production setting); τ_S=0.25 → ~2.0 (aggressive).

**Edge cases:** σ ≈ 0 (NGBoost degenerate) → reject; μ < 0 → reject; μ NaN → reject.

**Reference:** Lo, A.W. 2002. *The Statistics of Sharpe Ratios.* FAJ 58(4): 36–52.

### Gate B — No-trade band (Constantinides 1986 / Davis-Norman 1990)

> *Don't trade unless deviation from optimal weight exceeds the band.*

Under proportional transaction costs τ, the optimal portfolio policy under log utility is **not** to rebalance to the Markowitz weight. There's a no-trade interval [w*-Δ, w*+Δ] around w*; you trade only when the current weight crosses the band. The width Δ scales with τ^(1/3) (Davis-Norman 1990 closed form).

```
target_weight_i = α_i × SR_i / γ                   (Black-Litterman shape)
deviation_i     = target_weight_i - current_weight_i
band_i          = c × (γ × σ_i² × τ_round_trip)^(1/3)    (Davis-Norman 1990)
admit(i)        ⇔  |deviation_i| > band_i
```

For our parameters (γ=3, σ=0.08, τ=0.001): band ≈ 4% of NAV. So a candidate at weight=0 with target weight 3% wouldn't cross the band → no trade. **Naturally prevents "fill empty slots with weak signal."**

**References:** Constantinides 1986 (JPE 94(4): 842–862); Davis & Norman 1990 (Math. of Op. Research 15(4): 676–713).

### Gate C — Cross-sectional persistence (Garleanu-Pedersen 2013)

> *Don't trade a signal whose autocorrelation is too low to recover the cost.*

For autocorrelated signal decaying at rate `1-φ` per bar with quadratic cost coefficient Λ, the optimal position aims **partially** at the Markowitz target:

```
w_t = (1 - a) · w_{t-1} + a · w_target
```

For high-decay (φ≈0): a is small (don't move — signal will be gone before cost is amortized). For persistent (φ≈1): a≈1 (trade aggressively).

**Per-candidate measurement:** `φ_i = correlation(panel_score_t(i), panel_score_{t-1}(i))` over last 60 bars (from `score_distribution` table, just shipped). Reject if `τ_half = -ln(2)/ln(φ)` < min_hold_days × 0.5.

**Reference:** Gârleanu & Pedersen 2013 (JF 68(6): 2309–2340).

### Composition

```
candidates_panel = [c for c in ctx.candidates if pass_panel_score(c)]   # current pre-gate output

# Gates compose; each independently flag-controlled
candidates_A = [c for c in candidates_panel if c.μ/c.σ > τ_S]            # Gate A
candidates_B = [c for c in candidates_A     if past_no_trade_band(c)]    # Gate B
candidates_C = [c for c in candidates_B     if signal_persistent(c)]     # Gate C

ctx.candidates = candidates_C
```

All three gates default OFF preserves bit-for-bit pre-gate behaviour.

---

## 2. Portfolio QP solver — Markowitz with cost + decay + DD

The decision layer (formerly a 3-pass greedy heap) now runs a **Markowitz QP** when `rotation.joint_actions.solver = "qp"`. The QP optimises `Δw` jointly across all assets subject to:
- Linear cost `κ·|Δw|`
- Risk aversion `γ·Δwᵀ Σ Δw`
- Drawdown scaler (γ_eff grows as portfolio approaches `qp_drawdown_limit`)
- Optional: signal decay (G-P 2013), robust μ (Garlappi-Uppal-Wang 2007), CVaR (Rockafellar-Uryasev)

Implementation: `kernel/portfolio_qp/qp_solver.py` + `task_joint_qp.py`. 29 + 12 unit tests.

**Sign convention:** buy positive, sell negative. Wash-sale mask enforces `Δw_i ≤ 0` post-recent-sale. NaN μ → asset stays at current weight (treated as 0).

When `solver = "greedy"` (default fallback), QP task short-circuits → original behaviour preserved.

---

## 3. Code map

```
backtesting/renquant_104/
├── kernel/
│   ├── panel_pipeline/
│   │   ├── job_panel_scoring.py             # chain wires QualityFloorTask
│   │   └── task_quality_floor.py            # Gates A / B / C (~280 LOC)
│   ├── pipeline/
│   │   ├── job_joint_actions.py             # chain wires QP + greedy tasks
│   │   ├── task_joint_actions.py            # legacy greedy (short-circuits if solver=qp)
│   │   ├── task_sell.py                     # sell-side: SellGateBTask + PanelConvictionExit
│   │   ├── task_limit_sells.py              # max_sells_per_bar (round-7)
│   │   └── task_score_distribution.py       # score_db (Gate A input + persistence calc)
│   └── portfolio_qp/
│       ├── qp_solver.py                     # Markowitz QP w/ Stage 2-7 knobs
│       ├── task_joint_qp.py                 # Pipeline Task wrapper
│       └── signal_combiner.py               # Treynor-Black blend (Stage 6)
scripts/
├── validate_buy_logic.py                    # single-variant sim runner
└── run_validation_matrix.sh                 # parallel matrix launcher
```

## 4. Config flags

All in `strategy_config.json` (paired with `strategy_config.golden.json`).

```jsonc
"ranking": {"panel_scoring": {
    "quality_floor": {
        "enabled": true,
        "edge_sharpe_floor": {"enabled": true,  "threshold": 0.10},   // Gate A — current production
        "distribution_floor": {"enabled": false, "percentile": 85},   // Gate ?
        "no_trade_band": {"enabled": false, "risk_aversion": 3.0}     // Gate ?
    },
    "sell_gate_b": {                                                    // round-7 sell-side
        "enabled": true,
        "threshold": 0.10
    }
}}

"rotation": {"joint_actions": {
    "enabled": true,
    "solver": "qp",                          // currently active; "greedy" = legacy
    "qp_risk_aversion": 3.0,                 // γ in Markowitz
    "qp_cost_kappa": 0.0001,                 // κ in linear cost
    "qp_dw_max": 0.50,                       // single-bar slippage cap
    "qp_min_dw_pct": 0.005,                  // dust-trade dropout
    "qp_drawdown_limit": 0.20,               // Stage 4 — DD scaler limit
    "qp_signal_decay": 0.0,                  // Stage 2 — Garleanu-Pedersen φ (off)
    "qp_robust_mu_kappa": 0.0,               // Stage 5 — Garlappi-Uppal-Wang κ (off)
    "qp_cvar_lambda": 0.0                    // Stage 7 — Rockafellar-Uryasev λ (off)
}}
"risk": {
    "max_sells_per_bar": 2                   // round-7 cap (sell side)
}
```

---

## 5. Rollback procedures (rank-ordered by reversibility)

### 5.1 Soft revert — kill QP, keep gates

```bash
python -c "
import json
p='backtesting/renquant_104/strategy_config.json'
c=json.load(open(p)); c['rotation']['joint_actions']['solver']='greedy'
json.dump(c, open(p,'w'), indent=2)
"
# Mirror to strategy_config.golden.json
```

### 5.2 Full revert — kill gates AND QP

```bash
python -c "
import json
p='backtesting/renquant_104/strategy_config.json'
c=json.load(open(p))
c['ranking']['panel_scoring']['quality_floor']={'enabled':False}
c['rotation']['joint_actions']['solver']='greedy'
json.dump(c, open(p,'w'), indent=2)
"
```

### 5.3 Hard revert — git

```bash
git revert <commit_hash>   # buy-logic / portfolio-qp commits are tagged
```

---

## 6. Observability

### 6.1 Live runner / ntfy

Active QP:
```
[INFO] kernel.portfolio_qp.joint_qp: QP_BUY  AAPL  Δw=+0.0823  shares=4 ...
[INFO] kernel.portfolio_qp.joint_qp: JointPortfolioQPTask: solved n=58 buys=2 sells=1 ...
```

Gate B rejection:
```
[INFO] kernel.panel_pipeline.quality_floor: QualityFloorTask: rejected 3/8 ...
```

### 6.2 SQLite (`data/runs.db`)

- `pipeline_runs.{n_buys, n_exits, regime, confidence}` per bar
- `candidate_scores.blocked_by` stamped with `quality_floor:gate_b:edge_sharpe=...` when gate fires
- `ticker_daily_state` (Round-5 user spec) — every watchlist ticker per bar with full decision context

Query last 5 bars' QP behaviour:
```sql
SELECT pr.run_date, pr.regime, pr.n_buys, pr.n_exits,
       cs.ticker, cs.blocked_by
FROM pipeline_runs pr
LEFT JOIN candidate_scores cs ON cs.run_id = pr.run_id
ORDER BY pr.run_date DESC LIMIT 50;
```

### 6.3 Sim validation reports

- `logs/sim_validations/{date}-{tag}.md` — per-variant 27-mo OOS results
- `logs/sim_validations/baseline.json` — current reference baseline

---

## 7. Common failure modes

| Symptom | Likely cause | Diagnose | Fix |
|---|---|---|---|
| All candidates rejected; n_buys=0 for many bars | Gate B threshold too tight | Query `candidate_scores.blocked_by` last 20 bars | Lower `edge_sharpe_floor.threshold` 0.10 → 0.05 |
| QP solver status=failed every bar | Infeasible (cash_reserve > cash) | Log: `JointPortfolioQPTask: solver returned status=failed` | Solver auto-falls-back to greedy; check ctx.cash + DD invariants |
| Position cap drops to 0 unexpectedly | DD scaler firing — γ_effective explodes | Log: `QP: gamma_effective=...` value > 100 | Raise `qp_drawdown_limit` or stop bleed manually |
| Constant rank_score across panel | Calibrator collapse — separate issue | See `calibration.md` | Re-run `scripts/fit_panel_calibrator.py` |
| Δw signs look weird | Sign convention | Buy positive, sell negative; wash-sale enforces Δw≤0 post-sale | Read `qp_solver.py::solve_portfolio_qp` |

---

## 8. Promotion path (per CLAUDE.md §2a — ≥+2pt APY win OR theory-clean parity gain)

| Phase | Status | What |
|---|---|---|
| 1 | ✅ shipped | Gate B@τ=0.10 + QP solver default |
| 2 | ⏳ next | Tighten Gate B to 0.20 if sim doesn't crater APY |
| 3 | ⏳ | Enable Gate A (distribution floor) — score_db needs ≥30d history |
| 4 | ⏳ | Enable Gate C (no-trade band) — closed-form bands prevent dust |
| 5 | ⏳ | QP advanced knobs: signal decay (Stage 2), robust μ (Stage 5), CVaR (Stage 7) |
| 6 | ⏳ | Treynor-Black signal combiner (Stage 6) |

---

## 9. Cross-references

- **Acceptance gates** (round-7) — [`../../backtesting/renquant_104/kernel/model_acceptance.py`](../../backtesting/renquant_104/kernel/model_acceptance.py); auto-rollback on bad retrain
- **Sell-side mirror** — [`sell-logic.md`](sell-logic.md): SellGateB + LimitSellsPerBar
- **QP design** — see this doc §2 + `kernel/portfolio_qp/qp_solver.py`
- **Calibrator collapse** — [`calibration.md`](calibration.md)
- **Score distribution DB** — [`calibration.md`](calibration.md) §4 (percentile-based gate)
- **Pre-gate baseline** — [`../ops/golden-config.md`](../ops/golden-config.md)
- **Architectural rule** — CLAUDE.md §1b (every logical unit is a Task/Job/Pipeline)

## 10. References (~14 papers)

Core:
- Constantinides 1986 — *Capital Market Equilibrium with Transaction Costs*. JPE 94(4): 842–862.
- Davis & Norman 1990 — *Portfolio Selection with Transaction Costs*. Math of OR 15(4): 676–713.
- Gârleanu & Pedersen 2013 — *Dynamic Trading with Predictable Returns*. JF 68(6): 2309–2340.
- Lo 2002 — *The Statistics of Sharpe Ratios*. FAJ 58(4): 36–52.
- Grinold 1989 — *The Fundamental Law of Active Management*. JPM 15(3): 30–37.

Adjacent (referenced but not yet implemented):
- Black & Litterman 1992 — view-confidence portfolio combination
- Brandt-Santa-Clara-Valkanov 2009 — parametric portfolio policies
- Garlappi-Uppal-Wang 2007 — robust portfolio with multi-prior
- Bouchaud et al. 2018 — square-root impact law (Ch. 12)
- López de Prado 2018 — *Advances in Financial Machine Learning* (Chs. 7, 14, 16)
- Treynor-Black 1973 — security-analysis combination
- Markowitz 1952 — portfolio selection (foundational, every QP step)
