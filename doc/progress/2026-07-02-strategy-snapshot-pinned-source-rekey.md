# Re-key the generated strategy-104 snapshot to the PINNED subrepo config (M9/A6)

STATUS: delivered
WHAT: The first generated snapshot (PR #429, `scripts/render_strategy_104_snapshot.py`)
read `backtesting/renquant_104/strategy_config.json` — the umbrella WORKING-COPY
config. That is not what the daily run consumes: production config comes from the
PINNED subrepo checkout at `.subrepo_runtime/repos/renquant-strategy-104/configs/`
(pin-aligned to the `renquant-strategy-104` commit in `subrepos.lock.json`). The
working copy went stale across the 2026-06-23 XGB re-promotion, so the "generated"
snapshot faithfully rendered `kind="hf_patchtst"` while production ran
`kind="xgb"` — the A6 rot class recurring one level down. This change re-keys the
renderer to the pinned configs (active + `strategy_config.shadow.json`), and
extends the snapshot with: calibrator identity (active + shadow-e2e), key policy
knobs (`conviction_gate.mu_floor`, `panel_buy_top_n`, Kelly, position/QP caps,
per-regime caps, wf_gate relax flags), subrepo pin identities from
`subrepos.lock.json`, per-source sha256 fingerprints, a machine-readable block,
and an explicit warning row whenever the umbrella working copy disagrees with the
pinned config. Missing files/fields render as explicit
`unknown (field absent)` / `unknown (file missing...)` — never crash, never
invent. Guards: `make snapshot-check` (byte-exact re-render diff on the operator
machine; `Generated-at` excluded), `--selftest` (fixture-based), and a reworked
`.github/workflows/strategy-104-snapshot-fresh.yml` (renderer selftest + unit
tests always; `--verify-pinned-declaration` semantic check whenever
`subrepos.lock.json` changes — clones strategy-104 at the new pin, compares the
snapshot's machine block; a pin bump that changes the production declaration
without a regenerated snapshot goes RED).
WHY/DIR: unified 107 master plan M9 (renquant-orchestrator,
`doc/design/2026-07-02-unified-107-master-plan.md`, Term PROCESS) and amendment
A6 (`doc/design/2026-07-01-104-105-design-review-amendments.md`) — the stale
"PatchTST primary" claim invalidated #210 R1 and cost a full review round; the
snapshot must be keyed to the source production actually consumes.
EVIDENCE:
- Live pinned checkout `.subrepo_runtime/.../configs` HEAD = `c019b256` (read as a
  plain file, no git command against the live tree), byte-identical to
  `renquant-strategy-104@c019b256` on GitHub; declares `kind="xgb"`,
  `artifacts/prod/panel-ltr.alpha158_fund.json`, PatchTST seed44 as in-run shadow
  + shadow-e2e — exactly what the committed snapshot now states.
- Umbrella working-copy config still declares `kind="hf_patchtst"` → the rendered
  snapshot carries an explicit `UMBRELLA WORKING-COPY DRIFT` warning row.
- `--verify-pinned-declaration` run against strategy-104@`1fe312b4` (the
  origin/main lock pin, pre-re-promotion) fails with 5 mismatches including
  `snapshot active kind 'xgb' != pinned config kind 'hf_patchtst'` — the guard
  demonstrably catches the exact incident class.
- `--selftest` 14/14 checks green; `tests/test_render_strategy_104_snapshot.py`
  12 passed (whitelist/no-leak and absolute-path-redaction guarantees from the
  PR #429 Codex review preserved and re-tested).
NEXT: wire `make snapshot-check` + regeneration into the landing loop —
recommended: a post-promote hook at the end of `weekly_wf_promote.sh` (promotes
are the exact moment the declaration changes), with the weekly ops job as a
backstop; operator decision on which.
