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

## Round 3 (Codex review: staleness closure was incomplete — only a manual
## make target, no event-driven enforcement anywhere)

**Finding.** The claimed staleness closure was false: `make snapshot-check` was
a manual target only, wired into nothing. No promote/rollback path, no
`system_doctor`, no scheduled job ran it — the exact M9/A6 operational gap
this PR was supposed to remove. Also: PIN DRIFT was a warning baked into the
rendered doc, never a refusal, so a drifted runtime could produce a "canonical"
snapshot claiming pin alignment. And the `Generated-at:` wall-clock line
churned the committed doc on every bare regeneration with zero semantic
change.

**Fix — genuinely event-driven this time, verified against the actual
integration points, not assumed:**
- `scripts/promote_pin.py`: `bump --apply` and `revert --apply` now run
  `check_snapshot_freshness()` by DEFAULT after a successful sync+verify —
  regenerates the snapshot to a scratch path, compares against the committed
  doc, and if they differ, prints an actionable diff preview and exits 1. It
  NEVER auto-commits the regenerated content and NEVER reverts the pin change
  for a stale-snapshot finding alone (the pin change may be entirely correct;
  only the doc needs a human to run `make snapshot` and commit). Opt out with
  `--skip-snapshot-check` (documented as an explicit escape hatch, not the
  default).
- `scripts/system_doctor.py`: new `check_strategy_snapshot()`, always-on (not
  opt-in like the branch check), added to `run_all()`. Verified this is
  actually exercised in production, not just by `make doctor` run by hand:
  `scripts/daily_104.sh` already invokes `system_doctor.py` as a daily,
  non-fatal "system health heartbeat" (ntfy-alerts on RED, never blocks
  trading) — so any out-of-band artifact/calibrator edit that never goes
  through `promote_pin.py` at all (e.g. a `weekly_wf_promote.sh` model
  promotion) now surfaces as a same-day ntfy alert, not silent indefinite
  rot. This is the genuinely-closed half of the M9/A6 gap.
- `scripts/render_strategy_104_snapshot.py`: PIN DRIFT now fails closed for
  BOTH default generation and `--check` (new `--allow-pin-drift` diagnostic
  escape hatch, documented as never-commit-this-output). The wall-clock
  `Generated-at:` line is replaced with a deterministic `Source fingerprint:`
  — a sha256 over the sorted per-file source hashes, so it changes iff actual
  source CONTENT changes, never on a bare regeneration; `--check`'s
  comparison is now plain byte-equality (no more stripping needed).
- Tests: 25 new/updated across `tests/test_render_strategy_104_snapshot.py`
  (pin-drift fail-closed in generation + check + diagnostic-allow modes,
  fingerprint determinism), `tests/test_system_doctor.py` (green-when-fresh,
  RED on an out-of-band artifact-metadata edit, RED on an out-of-band
  active-calibrator edit, skip-when-renderer-absent — using a real fixture
  repo + the real renderer, not mocks), `tests/test_promote_pin.py` (default-on
  behavior via a mocked check, opt-out flag, revert path coverage, PLUS two
  fully real end-to-end tests against a real fixture+renderer proving
  `check_snapshot_freshness` itself is genuinely content-sensitive and never
  auto-commits). 37/37 pass across the three files.
- `.github/workflows/strategy-104-snapshot-fresh.yml`'s header comment
  (previously the source of the false "wired into the post-promote/weekly ops
  loop" claim this whole round exists to fix) now names the actual mechanism
  precisely instead of asserting it existed.

**Honest STATUS on the M9/A6 closure claim:** genuinely closed for (a) subrepo
pin bump/revert via `promote_pin.py` (event-driven, default-on, verified) and
(b) same-day detection of ANY drift via the pre-existing daily
`system_doctor.py` heartbeat in `daily_104.sh` (verified this heartbeat
already exists and runs daily, non-fatal/ntfy-alerting — did not need to add
new scheduling for this, it already ran). NOT wired: `weekly_wf_promote.sh`'s
own model-promotion flow does not call `promote_pin.py` (confirmed via grep —
it's a separate promotion mechanism entirely) and has no INLINE
regenerate-and-verify step of its own; a model promotion's drift is still
caught, but via the NEXT daily heartbeat, not synchronously inside
`weekly_wf_promote.sh` itself. If synchronous-at-promotion-time detection is
required for model promotions specifically (not just same-day), that needs a
separate follow-up PR touching `weekly_wf_promote.sh` directly — flagged
honestly here rather than claimed as done.

**Also:** did NOT regenerate the committed `doc/arch/strategy-104-snapshot.md`
in this worktree — this worktree's `.subrepo_runtime` is not fully
materialized (confirmed: a real render here degrades to mostly
`unknown (field absent)` rows, which would be actively wrong to commit as the
canonical snapshot). The doc format change (`Generated-at:` →
`Source fingerprint:`) means `make snapshot-check` will correctly report
STALE against the currently-committed doc until the operator runs
`make snapshot` for real on the live tree (which has the fully materialized
runtime) and commits the regenerated result — this is expected, not a bug,
and is the FIRST real exercise of the new event-driven backstop once this PR
lands.
