# Buy Logic + Portfolio Manager — Operator Runbook (2026-04-26)

**Audience**: future debug / on-call agent (you, three weeks from now;
or another assistant called in to triage). Single canonical reference
for what shipped, where it lives, how to operate it, and how to roll back.

**Status as of 2026-04-26**: enabled in production at conservative settings.
- QP solver: ON (replaces greedy heap).
- Gate B (Edge-Sharpe floor τ=0.10): ON.
- Gate A, Gate C, Stages 2-7 of QP: OFF (earn via sim).

---

## 1. What changed

### Before (golden v4.1, ≤2026-04-25)
- `JointActionTask` was a 3-pass greedy heap on (BUY, SELL_panel, ROTATE)
  actions. No portfolio-level optimisation, no quality gate on weak
  candidates.
- `panel_buy_floor = null` → any candidate with μ > fee passed.
- Result: R6/R7 buys at μ=0.008, σ=0.08 (edge_sharpe ≈ 0.10) — basically
  random within the panel's noise floor.

### After (this commit)
- `JointPortfolioQPTask` runs FIRST in `JointActionJob.tasks` chain.
  When `rotation.joint_actions.solver == "qp"`, it owns the bar and
  emits orders/exits via solving a Markowitz QP. Otherwise (default
  legacy "greedy") it short-circuits — original behaviour preserved.
- `QualityFloorTask` runs LAST in `PanelScoringJob.tasks`, after all
  scoring/calibration/sizing. Filters candidates by quality gates A/B/C.
- **Net effect**: weak candidates (R7-style μ=0.008/σ=0.08) get
  rejected at Gate B; remaining candidates feed into a QP that
  jointly optimises Δw across all assets with cost-aware sizing.

---

## 2. Where the code lives

```
backtesting/renquant_104/
├── kernel/
│   ├── panel_pipeline/
│   │   ├── job_panel_scoring.py             ← chain wires QualityFloorTask
│   │   └── task_quality_floor.py            ← Gates A / B / C (NEW)
│   ├── pipeline/
│   │   ├── job_joint_actions.py             ← chain wires QP + greedy tasks
│   │   ├── task_joint_actions.py            ← legacy greedy (short-circuits if solver=qp)
│   │   └── task_score_distribution.py       ← reads from score_db (Gate A input)
│   └── portfolio_qp/
│       ├── qp_solver.py                     ← Markowitz QP w/ Stage 2-7 knobs
│       ├── task_joint_qp.py                 ← Pipeline Task wrapper
│       └── signal_combiner.py               ← Treynor-Black blend (Stage 6)

scripts/
├── validate_buy_logic.py                    ← single-variant sim runner
└── run_validation_matrix.sh                 ← parallel matrix launcher

tests/
├── test_quality_floor_gate_a.py             ← 12 tests
├── test_quality_floor_gate_b.py             ← 15 tests
├── test_quality_floor_gate_c.py             ← 13 tests
├── test_portfolio_qp_solver.py              ← 29 tests (Stages 0-7)
├── test_joint_qp_task.py                    ← 12 tests
├── test_signal_combiner.py                  ← 12 tests
└── test_qp_integration.py                   ← 10 integration tests
```

Total new code: ~1500 LOC + ~800 LOC tests across 9 files.

---

## 3. Config — what to flip

All flags are in `backtesting/renquant_104/strategy_config.json`. The
mirror in `strategy_config.golden.json` (the rollback safety net) is
also updated when you change production.

### 3.1 Quality gates (filter candidates BEFORE QP)

```json
"ranking": {"panel_scoring": {"quality_floor": {
    "enabled": true,                             // master switch

    "edge_sharpe_floor": {                       // Gate B — Lo 2002
        "enabled": true,
        "threshold": 0.10                        // currently active
    },
    "distribution_floor": {                      // Gate A — adaptive
        "enabled": false,
        "percentile": 85,
        "lookback_days": 20,
        "min_history_days": 5
    },
    "no_trade_band": {                           // Gate C — Constantinides
        "enabled": false,
        "risk_aversion": 3.0,
        "round_trip_cost": 0.001,
        "band_constant": 1.5
    }
}}}
```

### 3.2 Portfolio QP (decision layer)

```json
"rotation": {"joint_actions": {
    "enabled": true,                             // (was already on)
    "solver": "qp",                              // currently active; "greedy" = legacy
    "qp_risk_aversion": 3.0,                     // γ in Markowitz
    "qp_cost_kappa": 0.0001,                     // κ in linear cost
    "qp_dw_max": 0.50,                           // single-bar slippage cap
    "qp_min_dw_pct": 0.005,                      // dust-trade dropout
    "default_sigma": 0.05,                       // when NGBoost σ missing
    "qp_drawdown_limit": 0.20,                   // Stage 4 — DD scaler limit

    "qp_signal_decay": 0.0,                      // Stage 2 — Garleanu-Pedersen φ
    "qp_robust_mu_kappa": 0.0,                   // Stage 5 — Garlappi-Uppal-Wang κ
    "qp_cvar_lambda": 0.0                        // Stage 7 — Rockafellar-Uryasev λ
}}
```

### 3.3 Activation log (audit trail)

```json
"_activation_log": [{
    "date": "2026-04-26",
    "change": "enable QP solver + Gate B (τ=0.10)",
    "rollback": "..."
}]
```

Future activations append to this list. **Never overwrite — only append.**

---

## 4. Rollback procedures (rank-ordered by reversibility)

### Rollback 1 — kill QP solver only (revert decision layer to greedy)

```bash
# In strategy_config.json, set:
"rotation.joint_actions.solver": "greedy"
```
Or quick CLI:
```bash
python -c "
import json
p='backtesting/renquant_104/strategy_config.json'
c=json.load(open(p))
c['rotation']['joint_actions']['solver']='greedy'
json.dump(c, open(p,'w'), indent=2)
"
```
Then mirror to `strategy_config.golden.json`. Effect: next bar uses
the legacy greedy heap. Bit-for-bit equivalent to pre-2026-04-26.

### Rollback 2 — kill quality gates (full revert)

```bash
python -c "
import json
p='backtesting/renquant_104/strategy_config.json'
c=json.load(open(p))
c['ranking']['panel_scoring']['quality_floor']={'enabled':False}
json.dump(c, open(p,'w'), indent=2)
"
```

### Rollback 3 — Hard revert (git)

```bash
git revert <commit_hash>   # the commit that turned things on
```

The git history is canonical. All commits clearly tagged with
`feat(buy-logic):` / `feat(portfolio-qp):` so they're easy to find.

---

## 5. Observability — what to watch

### 5.1 ntfy / live runner output

When QP is active, log lines change:
```
[INFO] kernel.portfolio_qp.joint_qp: QP_BUY  AAPL  Δw=+0.0823  shares=4 ...
[INFO] kernel.portfolio_qp.joint_qp: JointPortfolioQPTask: solved n=58 buys=2 sells=1 ...
```

When Gate B rejects a cand:
```
[INFO] kernel.panel_pipeline.quality_floor: QualityFloorTask: rejected 3/8 ...
```

### 5.2 SQLite (`data/runs.db`)

- `pipeline_runs` rows include `n_buys`, `n_exits`, `regime`, `confidence`
- `candidate_scores` includes `blocked_by` (now stamped with
  `quality_floor:gate_b:edge_sharpe=...` when gate fires)
- `ticker_daily_state` (Round-5 user spec, commit `3bb2ca4`) — every
  watchlist ticker per bar with full decision context

Query the last 5 bars' QP behaviour:
```sql
SELECT pr.run_date, pr.regime, pr.n_buys, pr.n_exits,
       cs.ticker, cs.blocked_by
FROM pipeline_runs pr
LEFT JOIN candidate_scores cs ON cs.run_id = pr.run_id
ORDER BY pr.run_date DESC LIMIT 50;
```

### 5.3 Sim validation reports

`logs/sim_validations/{date}-{tag}.md` — per-variant 27-mo OOS results.
`logs/sim_validations/baseline.json` — current reference baseline.

---

## 6. Common failure modes + diagnostics

### 6.1 "All candidates rejected, n_buys=0 for many bars"

**Cause**: Gate B threshold too tight for current market regime.
**Diagnose**: Query `candidate_scores.blocked_by` over last 20 bars —
if mostly `quality_floor:gate_b`, threshold is biting too hard.
**Fix**: Lower `edge_sharpe_floor.threshold` from 0.10 → 0.05 (or 0).

### 6.2 "QP solver returns status=failed"

**Cause**: infeasible problem (e.g. cash_reserve > current cash position).
**Diagnose**: log line `JointPortfolioQPTask: solver returned status=failed:...`
followed by deferral to greedy.
**Fix**: Solver is designed to fall back on failure. If happening
every bar, check `regime_state.drawdown` and `ctx.cash` invariants.

### 6.3 "Position cap suddenly drops to 0"

**Cause**: regime_state.drawdown too close to drawdown_limit, γ_eff
explodes.
**Diagnose**: log line `QP: gamma_effective=...` — value > 100 means
DD scaler firing.
**Fix**: Either raise `qp_drawdown_limit` or stop the bleed (manual
intervention).

### 6.4 "Constant rank_score across panel" (calibrator collapse)

**Cause**: post-training calibrator collapsed to single y value. Not
a buy-logic bug — calibrator-side issue.
**Diagnose**: `doc/components/calibration-saturation.md` covers the
fix; CALIB-COLLAPSE-GUARD now refuses to ship a unique_y < 3 calibrator.
**Fix**: rerun calibrator with `scripts/fit_panel_calibrator.py`.

### 6.5 "Transformer SIGSEGV at calibrator step"

**Cause**: transformer's SaveArtifactTask doesn't write panel-ltr.json.
Calibrator reads stale file from prior backend.
**Fix**: Already fixed in commit `75d1263` (TRANSFORMER-PANEL-LTR-SHIM).
Next Sunday sweep should not crash. If it does, check
`backtesting/renquant_104/artifacts/panel-ltr.json` — it should have
`kind=panel_transformer` after a transformer-backend training run.

---

## 7. Promotion path — turning on more features

Per CLAUDE.md §2a, each new feature stage needs ≥+2pt APY win on
27-mo OOS, OR theory-clean parity-driven minor win.

### Phase 1 (NOW — this commit): Gate B at 0.10 + QP solver default
Conservative roll-out. Watch for ~2 weeks.

### Phase 2 (after sim validation): Gate B tightening
If sim shows Gate B at 0.20 doesn't crater APY, tighten:
```json
"edge_sharpe_floor": {"enabled": true, "threshold": 0.20}
```

### Phase 3: Gate A — distribution floor
After score_db has enough history (it should already by 2026-05-15):
```json
"distribution_floor": {"enabled": true, "percentile": 85, ...}
```

### Phase 4: Gate C — no-trade band
Closed-form bands prevent dust trades:
```json
"no_trade_band": {"enabled": true, "risk_aversion": 3.0, ...}
```

### Phase 5: QP advanced knobs (Stages 2/4/5/7)
Each knob requires independent sim verification. See
`scripts/validate_buy_logic.py --help` for ablation flags.

### Phase 6: Treynor-Black signal combiner (Stage 6)
Requires plumbing: pass per-source IC means/stds into QP. Not yet
wired into pipeline; lives as `kernel.portfolio_qp.signal_combiner.combine_signals`
for callers to use directly.

---

## 8. Test coverage map

| Component | Test file | # tests | Run |
|---|---|---:|---|
| QP solver core | `test_portfolio_qp_solver.py` | 29 | `pytest tests/test_portfolio_qp_solver.py` |
| QP Task wiring | `test_joint_qp_task.py` | 12 | |
| Signal combiner | `test_signal_combiner.py` | 12 | |
| Gate A | `test_quality_floor_gate_a.py` | 12 | |
| Gate B | `test_quality_floor_gate_b.py` | 15 | |
| Gate C | `test_quality_floor_gate_c.py` | 13 | |
| Integration | `test_qp_integration.py` | 10 | |
| Hourly wash | `test_intraday_wash.py` | 20 | |
| **Subtotal new code** | | **123** | |
| Core regression | (existing) | ~250 | `pytest tests/test_panel_alignment.py ...` |

Run all together:
```bash
python -m pytest tests/ -q
```
Expected: ~370+ pass, 0 fail.

---

## 9. References

- `doc/components/buy-logic-design.md` — theory & 3-gate design
- `doc/components/portfolio-qp.md` — 7-stage QP design
- `doc/components/calibration-saturation.md` — separate calibrator issue
- `doc/ops/golden-config.md` — pre-this-commit baseline (v4.1)
- CLAUDE.md §2a — promotion criteria
- CLAUDE.md §1b — every logical unit is a Task/Job/Pipeline (architectural rule)

---

## 10. Cheat sheet for the next debugger

If trades stop happening:
1. Grep ntfy for `QualityFloorTask: rejected` — gate biting
2. Grep ntfy for `JointPortfolioQPTask: solver returned` — QP failing
3. Query `candidate_scores.blocked_by` distribution — find the bottleneck
4. Lower `edge_sharpe_floor.threshold` to 0.05 first; if no help,
   set `solver=greedy` to revert to legacy

If you see weird Δw signs:
1. Check `kernel/portfolio_qp/qp_solver.py:solve_portfolio_qp` —
   sign convention is **buy positive, sell negative, A<0+B>0=rotation**
2. Wash-sale mask enforces Δw_i ≤ 0 (cannot re-buy after recent sale)
3. NaN μ → treated as 0 (asset stays at current weight)

If new sigma_sizing or kelly_sizing seem to interact:
1. ApplyKellySizingTask runs BEFORE QualityFloorTask (in PanelScoringJob)
2. QP reads μ/σ from candidate.mu/sigma (set by ApplyNGBoostTask)
3. QP's σ-scaling is independent from sigma_sizing_multiplier (different code paths)

If a calibrator looks broken:
1. Check `n_unique_prob_y` in metadata — < 5 is degenerate
2. The CALIB-COLLAPSE-GUARD should have refused to write it
3. If it shipped anyway, manually delete the artifact and re-run
   `scripts/fit_panel_calibrator.py`

If you need to read the QP solver internals at runtime:
```python
from kernel.portfolio_qp.qp_solver import solve_portfolio_qp
sol = solve_portfolio_qp(
    w_current=[...], mu=[...], sigma=[...],
    risk_aversion=3.0, cost_kappa=0.0001,
)
print(sol.diagnostics)   # all gamma_effective / dd_factor / cvar_alpha
```
