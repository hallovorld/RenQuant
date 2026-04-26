# Session Summary — 2026-04-26

**Duration**: ~7 hours of focused work
**Commits**: 40+
**Real bugs fixed**: 19
**Test count**: 452+ → growing

## Headline outcomes

### 🚀 Production turn-on
- **Buy logic Gate B (Edge-Sharpe τ=0.10) + Portfolio QP solver** turned ON in production
- Validated via 27-mo OOS sim: **+26.91% APY, +1.474 Sharpe, 11.07% MaxDD, 78% win rate**
- Activation log + ops runbook documented

### 🛠️ Code shipped (all stages)
- **Buy logic 3 quality gates** (A: distribution, B: Edge-Sharpe, C: no-trade band) — all 7 stages of `unified_portfolio_action_design`
- **Portfolio QP solver** with 7 stages (Markowitz + linear cost + G-P decay + DD scaler + robust μ + Treynor-Black + CVaR)
- **Hourly transformer** Stages A (wash), B (cyclic encoding), C-1 (panel scaffold), C-2 (PanelTrainingPipeline wiring), C-3 (training launched)
- **Watchlist expansion**: LITE + COHR added (AI-infra optical comms)

### 🔧 Infrastructure
- `record_ticker_daily_state` + 23-col schema (per-watchlist-ticker decision audit)
- `validate_buy_logic.py` — 27-mo OOS sim runner
- `monitor_training_resources.py` — CPU/RSS sampler
- `plot_training_resources.py` — 8-panel post-training chart
- `enable_hourly_transformer.py` — Stage C-3 entry point
- `SNAPSHOT-OVERRIDE-WARN` guardrail in `sim/runner.py`

### 🔍 Mega-Audit (all 6 phases)
- **~70 files / 30k LOC audited**
- **0 P0 bugs found** in production critical path
- **5 P1 bugs filed** (signal_combiner dead, _apply_overrides property test,
  28 untested scripts, 43 unindexed docs, 18 mock-pattern tests)
- **7 P2 bugs filed** (3 fixed)

## Bugs fixed today (19 real bugs)

| # | Bug | Severity | File |
|---|---|---|---|
| 1 | NGB-OVERFLOW | 🟠 | ngboost_head.py |
| 2 | CALIB-PER-DATE-IC | 🔴 | global_calibrator.py |
| 3 | CALIB-COLLAPSE-GUARD | 🔴 | global_calibrator.py |
| 4 | SC-NEG-IC | 🔴 | signal_combiner.py |
| 5 | TRANSFORMER-SIGSEGV (panel-ltr.json shim) | 🔴 | pp_panel_training.py |
| 6 | QP-MATMUL-WARN | 🟡 | qp_solver.py |
| 7 | CACHE-DIR-SNAPSHOT (sim 0/101 fundamentals) | 🔴 | pp_panel_training.py |
| 8 | QP-REGIME-STATE-DUCK | 🔴 | task_joint_qp.py |
| 9 | NGB-INPUT-VALIDATION | 🟢 | ngboost_head.py |
| 10 | VALIDATE-BUYS-CALL (50min×4 sim lost) | 🔴 | validate_buy_logic.py |
| 11 | VALIDATE-SNAPSHOT-OVERRIDE | 🔴 | validate_buy_logic.py |
| 12 | VALIDATE-BASELINE-OFF | 🔴 | validate_buy_logic.py |
| 13 | NOTEBOOK-SAME-BUG (cells 14, 15) | 🔴 | renquant_104.ipynb |
| 14 | SNAPSHOT-OVERRIDE-WARN guardrail | 🟡 | sim/runner.py |
| 15 | COMPARE-PANEL-OVERRIDE | 🟠 | compare_panel_vs_baseline.py |
| 16 | AB-HARNESS-OVERRIDE | 🟠 | ab_harness.py |
| 17 | DB-PATH-WRONG-KEY (sqlite_path → sim_db_path) | 🔴 | validate_buy_logic.py |
| 18 | HARDCODED-PYTHON | 🟡 | sunday_panel_sweep.py |
| 19 | C2-FEATURE-COLS-EMPTY (Stage C-3 crash) | 🔴 | pp_panel_training.py |

**Pattern**: Bugs #6, #7, #8, #11, #12, #13, #14, #15, #16, #17, #19 all
share root cause **"new code I shipped didn't account for production-side
type/state shape"**. Need stronger integration tests. Prevention: each new
operator script gets ≥1 smoke test before commit.

## Documents shipped (10 docs)

| Doc | Purpose |
|---|---|
| `db_design_decision_factors_2026-04-26.md` | TDS schema + Tier-2 model registry |
| `buy_logic_redesign_2026-04-26.md` | 3-gate quality floor (14 refs) |
| `unified_portfolio_action_design_2026-04-26.md` | 7-stage QP design (35+ refs) |
| `calibrator_saturation_2026-04-26.md` | Pool IC vs scorer IC investigation |
| `buy_logic_portman_ops_2026-04-26.md` | Operator runbook |
| `transformer_hourly_stage_c2_design.md` | C-2 wiring design |
| `alpaca_crypto_btc_feasibility_2026-04-26.md` | BTC trading plan |
| `session_self_audit_2026-04-26.md` | Self-audit doc |
| `mega_audit_plan_2026-04-26.md` | 6-phase audit plan |
| `mega_audit_phase{1,2,3,4,5,6}_findings_2026-04-26.md` | Phase findings |
| `sim_ab_results_2026-04-26.md` | Production A/B verdict |
| `session_summary_2026-04-26.md` | THIS doc |

## Tomorrow (Mon 2026-04-27)

### 1:55 PM PT — first live run with new logic
- daily_104.sh fires
- Uses new strategy_config.json: solver=qp + Gate B@0.10
- LITE + COHR in watchlist but universe floor will skip (no model yet)
- Panel scoring + QP solver decide all trades
- Monitor: ntfy for trade summary; watch for QualityFloorTask warnings

### Stage C-3 hourly transformer training (in progress now)
- ETA results: ~16:50 PT today (~30 min from launch 16:15 PT)
- Promotion gate: OOS IC ≥ +0.10 to clear ship gate (vs daily transformer -0.003)
- If passes: Sunday sweep next week trains both daily + hourly, choose better

### Tue retrain cycle (per training.cadence)
- Tue 1:55 PT will retrain models (cadence: Tue/Thu/Sun)
- LITE + COHR get per-ticker models for the first time
- Sharpe ≥ 1.0 floor will gate them in or out of the universe

## Risk register (open items)

### 🟠 Open P1
- **signal_combiner not wired into pipeline** (Stage 6 of unified QP)
- **`_apply_overrides` needs property-based test** for keys-in/keys-out
- **28 scripts have ZERO tests** — pattern that bit us 5× this session
- **43 of 59 doc/ files NOT indexed in CLAUDE.md** — future agents won't find them
- **18 test files use mock patterns** that may diverge from prod dataclasses

### 🟠 Operational risk
- **Calibrator pool IC = 0.0011** (vs scorer 0.048) — 44× collapse, may degrade rank quality. See `calibrator_saturation_2026-04-26.md`.
- **Tomorrow's first live run is untested** with QP+Gate B path. Rollback per `buy_logic_portman_ops_2026-04-26.md` §4.1 (30-second CLI edit).

## Net session value

**Code-quality**: +substantial (19 bugs fixed, 7 audit docs, 70 files reviewed)
**Production risk**: higher (new logic active, untested live until tomorrow) but mitigated (rollback documented + sim validated)
**Development velocity**: high (40 commits, all stages of buy/QP shipped)
**Documentation debt**: identified and quantified (43 unindexed docs)

End-of-day status: **READY FOR TOMORROW'S LIVE RUN**.
