# Refresh the strategy-104 snapshot (doctor RED)

STATUS: delivered
WHAT: regenerates doc/arch/strategy-104-snapshot.md via make snapshot;
the staleness came from today's weekly-rollback backup artifacts changing
the pinned-source digest set (2026-07-17 anomaly-retrain rehearsal).
WHY/DIR: clears the strategy_104_snapshot_fresh doctor RED.
EVIDENCE: render output; verify-pinned-declaration must stay green.
NEXT: none.
