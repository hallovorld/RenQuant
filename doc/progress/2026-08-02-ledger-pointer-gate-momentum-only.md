# 2026-08-02 — restrict the ledger-pointer gate exception to the momentum contract

STATUS: complete (fix + 2 fail-closed regressions)

WHAT: `scripts/check_config_artifact_paths.py::_check_ledger_pointer` admitted
ANY shadow `artifact_path` ending in `.jsonl`, and an unresolved file plus any
`*_pending_first_artifact` key passed as INFO — unrelated future JSONL shadow
models or typos could bypass the resolvable-artifact identity gate. The branch
now admits only the momentum contract (declared entry kind
`momentum_residual` + the exact published reference
`artifacts/momentum/momentum_artifact_ledger.jsonl`); everything else fails
closed with a reason citing #550. Widening the set is a reviewed change by
design.

WHY/DIR: RenQuant#550 (codex post-merge review of #549) — the
enumerated-exception-leaves-a-fail-open-default class. One of the three gates
blocking the Grant C re-run (with pipeline#254 and orchestrator#758).

EVIDENCE:
- artifact: this PR's diff; tests/test_check_config_artifact_paths.py —
  33 passed, 1 skipped after the change (31 before; +2 regressions:
  non-momentum `.jsonl` with a pending marker fails closed; momentum kind
  with a typo'd ledger path fails closed)
- prod or exp: gate script + tests only; no production surface touched
- existing data: the s104#77 entry (kind `momentum_residual`, exact ledger
  reference) remains admitted — the existing valid/pending/tamper tests all
  still pass unchanged
- best-known?: yes — the failure mode is reproduced by the new regressions,
  not asserted
- scope: `_check_ledger_pointer` admission only; chain verification,
  tail-artifact identity, marker semantics unchanged

NEXT: rides the Grant C gate set — after #254 + this + #758 land, re-run the
step (c) machine sequence per the corrected order (record: orch#759).
