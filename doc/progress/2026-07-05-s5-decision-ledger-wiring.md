# S5: Wire umbrella runner gate verdicts to decision ledger

**Date:** 2026-07-05
**PR:** TBD (umbrella)
**Task:** S5 decision-ledger wiring (roadmap s2-wire-gate-ledger)

## What

Bridge the live runner's gate verdicts (already written to `runs.db.gate_verdicts`
by the pipeline GateRegistry) to the orchestrator's cross-run `decision_ledger.db`.

After the existing `record_gate_verdicts` call in `RunnerAdapter.commit()`, a
second best-effort write persists the same verdicts to
`~/renquant-data/decision_ledger.db` via
`renquant_orchestrator.decision_ledger.write_verdicts()`.

## Why

The `gate_verdicts` table in `runs.db` is per-run — good for in-run diagnostics.
The decision ledger is append-only across ALL runs, making "why was this run
sell-only on date X?" a single SQL query instead of log archaeology. This is the
S5 payoff: 100% decision provenance with cross-run queryability.

## Design choices

- **Best-effort, never blocks the bar**: wrapped in try/except, logged as warning
  on failure. Same contract as the existing `record_gate_verdicts` block.
- **No new dependencies**: `renquant_orchestrator` is already on PYTHONPATH
  (daily_104.sh line 112, subrepo assembly env.sh).
- **Idempotent**: `write_verdicts` uses INSERT OR IGNORE on (run_id, scope, gate)
  — re-running the same bar is a no-op, never a duplicate.
- **Format bridge**: pipeline's `GateRegistry.ledger_rows(run_id=...)` output is
  compatible with orchestrator's `write_verdicts` (reads scope/gate/verdict/reason/
  inputs keys; ignores extra run_id in each row).

## Not in scope

- Preflight-to-GateRegistry bridge (preflight checks → gate verdicts)
- Sim adapter wiring (decision_ledger is live-only; sim uses runs.db)
- Backfill of historical gate_verdicts → decision_ledger
