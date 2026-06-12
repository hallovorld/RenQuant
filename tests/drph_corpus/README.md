# DRPH golden corpus

Design: renquant-orchestrator `doc/research/2026-06-12-engineering-architecture-deep-plan.md`
§IV + S2 item 5. Tooling: `scripts/drph_capture.py` (live cases),
`scripts/drph_replay.py` (sim gate cases).

## Case kinds

- **sim_replay** (`capture_meta.kind = "sim_replay"`): refactor GATES.
  Re-runnable: `scripts/drph_replay.py verify --case <dir>` re-executes the
  frozen day and byte-compares. Worktree runs MUST symlink the canonical
  data tree first: `ln -s <repo>/data <worktree>/data`.
- **live** (no `kind`, captured by `drph_capture.py` from `runs.alpaca.db`):
  FORENSIC ANCHORS. Not replayable as gates (live surfaces include broker
  state); `drph_replay.py verify` refuses them by design.

## Inventory

| Case | Kind | Regime | Why it's here |
|---|---|---|---|
| `sim_2026-06-10` | sim_replay | BULL_CALM | first gate case (#314/#315) |
| `sim_2026-06-11` | sim_replay | BULL_CALM | second gate case — the false-BEAR day; note sim says BULL_CALM while the three live runs that day saw BEAR / CHOPPY / BULL_CALM (regime-instability evidence) |
| `2026-06-11_false_bear` | live | BEAR | the false-BEAR incident run (f68231b0) |
| `2026-06-11_live_2f0ce396` | live | CHOPPY | same day, second run — regime flapped |
| `2026-06-11_live_fbb8c140` | live | BULL_CALM | same day, third run — completes the trio |

## Rules

- Cases are content-addressed; `drph_capture.py check --case <dir>` (CI) and
  `tests/test_drph.py` verify integrity. Never hand-edit a case.
- A refactor PR is behavior-identical iff EVERY sim_replay case verifies
  PARITY OK on the PR head.
- DATA-DRIFT (exit 3) means the OHLCV store was restated — audit the
  restatement, then re-capture; never "fix" a case to make a PR pass.
