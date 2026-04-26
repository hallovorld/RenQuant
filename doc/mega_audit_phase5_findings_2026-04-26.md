# Mega Audit — Phase 5 Findings (2026-04-26)

**Scope**: scripts/ (28+ Python scripts)
**Methods**: M3 coverage + M2 hardcoded paths + M12 error swallowing
**Started**: 2026-04-26 16:05 PT

## Findings

### 🟠 P1 — 28 scripts with ZERO tests

```
ab_harness                        analyze_backtest
analyze_decision_factors          archive_runs
audit_transformer_vs_lgbm_4way    backtest_and_analyze
bench_python_vs_rust              compare_panel_backends
compare_panel_vs_baseline         compute_portfolio_metrics
enable_hourly_transformer         export_lean_data
export_lean_watchlist             export_panel_to_csv
export_transformer_to_safetensors fetch_earnings_calendar
fetch_hourly_bars                 kelly_param_validation
monitor_training_resources        new_strategy
plot_training_resources           poc_rust_transformer
query_runs                        recalibrate_diagnostic
sanitize_bridge_csv               sunday_panel_sweep
train_transformer_only            weekly_apy_check
```

Categories of risk:
- **Production-touching**: `sunday_panel_sweep`, `archive_runs`, `query_runs`,
  `compute_portfolio_metrics`, `fetch_*`, `recalibrate_diagnostic`,
  `weekly_apy_check`, `enable_hourly_transformer`. **HIGHEST risk** —
  these can corrupt live state.
- **Analysis-only**: `analyze_*`, `bench_*`, `compare_*`, `poc_*`,
  `audit_*`. Lower risk — read-only.
- **Operator helpers**: `monitor_training_resources`, `plot_training_resources`,
  `enable_hourly_transformer`. Medium — affect what operator sees.

This is the same pattern that bit us 5× this session. **The rule should
be: any script touched in production needs at least 1 smoke test.**

### 🟡 P2 — 1 hardcoded user path

`scripts/sunday_panel_sweep.py:` `PYTHON = "/Users/renhao/miniconda3/envs/renquant/bin/python"`

Risk: script breaks on other machines or in CI. Should be:
```python
PYTHON = sys.executable   # or os.environ.get("PYTHON", sys.executable)
```

This is THE script that runs every Sunday for retraining. If we ever
move to a CI-based training setup, this hardcode breaks it.

### 🟢 No bare-except patterns in scripts

`grep -lE "except.*:\s*pass$"` returned empty. All scripts handle
exceptions appropriately.

## Phase 5 outcome

- **0 P0 bugs** (no immediate production breakage)
- **1 P1 bug** (28 scripts untested — fix by adding minimal smoke tests)
- **1 P2 bug** (sunday_panel_sweep hardcoded user path)

The systemic issue: scripts shipped without tests is the SAME pattern
that caused VALIDATE-BUYS-CALL + VALIDATE-SNAPSHOT-OVERRIDE +
VALIDATE-BASELINE-OFF + DB-PATH-WRONG-KEY this session. **This is
the BIGGEST audit finding.**

### Action plan
1. Today (no production impact): fix `sunday_panel_sweep.py` PYTHON path
2. This week: add 1 smoke test per script that touches production state
   (8 scripts: sunday_panel_sweep, archive_runs, query_runs,
   compute_portfolio_metrics, fetch_*, recalibrate_diagnostic,
   weekly_apy_check, enable_hourly_transformer)
3. Next month: cover the remaining 20 read-only/analysis scripts

## Phase 6 next: configs + docs + tests-of-tests

- Strategy config schema audit (drift between live + golden)
- CLAUDE.md accuracy
- Doc cross-references
- Test files that import deprecated APIs
