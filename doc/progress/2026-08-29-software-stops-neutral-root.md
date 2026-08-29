# 2026-08-29 — Software-stop registry at the NEUTRAL root; the sell wrapper seeds before the runner; orchestrator pin advanced past orch#1078 (follow-ups 1 + 2)

**Bottom line:** the live runner resolves the software-stop registry under
the neutral runtime-state root (`~/.renquant/runtime/software-stops`,
override `RENQUANT_RUNTIME_STATE_ROOT`) — the `--data-root` the liveness
pager and the orchestrator seeder/readiness classifier read — instead of the
process cwd. `scripts/intraday_sell_104.sh` calls the orchestrator seeder
unconditionally before the runner with the same `--broker alpaca` the runner
receives; a failed seed is logged + paged but NEVER blocks the sell pass.
The renquant-orchestrator pin is advanced IN THIS PR to `64238032`
(contains orch#1078 seeder/READY + #1080), so the command the wrapper
invokes exists on the pinned PYTHONPATH the moment this lands — proven by an
assembly-level regression against the pinned runtime module. With
`execution.software_stops.enabled` absent/false (every config in this repo)
the wiring returns `None` before importing, resolving, logging or touching
anything: the pre-change inert path is preserved literally. No flag is
flipped; nothing installed; live tree and `-run` checkout untouched by this
branch (the pin promotion itself was applied by the operator via
`promote_pin.py --apply` and copied in).

Revision 2 (Codex CHANGES_REQUESTED on RenQuant#613, 2026-08-29T08:55Z):
(1) landing order — the pin advances here, not in a later PR, and the
"pinned module emits SEEDED/EXISTS" fact is a test; (2) the first revision's
"flag-off byte-identical" claim was false — it imported the orchestrator
contract before the pipeline gate, so a missing contract logged an ERROR
even when the layer was disabled, and the tests pinned that. Fixed: the
gate is read first; the tests now assert NO import / NO log / NO disk when
disabled.

## 1. Landing order: the pin (`subrepos.lock.json`, `doc/arch/strategy-104-snapshot.md`)

- `subrepos.lock.json`: renquant-orchestrator `75dd9c70` → `64238032744fa3752ddfda0ccec4daecb5c5d934`
  (the only changed line of the lock, verified by `git diff`). The
  snapshot was re-rendered by the promotion (source fingerprint + the
  orchestrator row + the lock's sha). The runtime assembly
  (`.subrepo_runtime/repos/renquant-orchestrator`) is at `64238032`
  `[VERIFIED git rev-parse]`; the pinned module has `seeded_registry_path`
  and the `seed`/`readiness` CLI.
- Assembly-level regression `tests/test_software_stops_neutral_root.py::TestPinnedRuntimeAssemblySeeder`:
  resolves the runtime root the way the wrappers do (`RENQUANT_SUBREPO_ROOT`
  → `.subrepo_assembly/current.env` export → `<repo>/.subrepo_runtime/repos`;
  skip WITH reason when absent), asserts the runtime checkout's HEAD equals
  the lock's commit, then runs `python -B -m renquant_orchestrator.software_stops_registry_contract
  seed --broker alpaca --data-root <tmp>` as a subprocess on
  `<root>/renquant-orchestrator/src` + the pinned pipeline `src`
  (`PYTHONDONTWRITEBYTECODE=1`, never writes into the pinned checkout):
  first run stdout has `^SEEDED: `, the file
  `<tmp>/data/rq105/software_stops.alpaca.json` exists with `stops: {}` /
  `last_evaluated_at: null`; second run `^EXISTS: `, bytes unchanged; exit
  0 both times; nothing under the cwd. A negative control
  (`test_pre_seeder_revision_would_fail`) runs a module with only the
  LOCATION functions — the shape of the old pin — and shows it exits 0 with
  EMPTY stdout, i.e. exactly what the SEEDED assertion rejects.

## 2. The path unification

| who | resolution | where |
|---|---|---|
| liveness checker (execution) | `registry_path_for(Path(data_root)/DEFAULT_REGISTRY_PATH, broker)`, `data_root` = plist value `~/.renquant/runtime/software-stops` | `renquant-execution/src/renquant_execution/software_stops_liveness.py:314-336` |
| seeder / readiness (orchestrator, #1078) | `seeded_registry_path(data_root, broker)` = `Path(data_root)/"data/rq105/software_stops.json"` broker-tagged; default `data_root` = `software_stops_registry_root(runtime_state_root())` | `software_stops_registry_contract.py:103-126` (root), `:194` (`DEFAULT_REGISTRY_REL`), `:263-286`, `:308`, `:624-630` |
| live runner (this repo), before | `from_config(config, broker_name=…)` with NO `repo_root` → relative `registry_path` against cwd = the umbrella checkout (`scripts/intraday_sell_104.sh:84` `cd "$REPO_DIR"`) | old `adapters/runner.py:233-236` |
| live runner, now | enabled gate first; then `from_config(config, broker_name=…, repo_root=software_stops_registry_root(runtime_state_root()))` | `adapters/runner.py:223-238` → `adapters/software_stops_wiring.py:100-141` |

The pipeline composes `Path(repo_root) / registry_path` then broker-tags it
(`renquant-pipeline/src/renquant_pipeline/software_stops.py:316-337`
`from_config`; `:141-154` `registry_path_for`; `DEFAULT_REGISTRY_PATH`
`:121`). With `repo_root` = the neutral data root and the default
`registry_path`, the runner's file is
`~/.renquant/runtime/software-stops/data/rq105/software_stops.alpaca.json` —
byte-equal to the checker's and the seeder's composition
(`TestFlagOnNeutralRootParity`: `registry.path == seeded_registry_path(root,
broker)` against the REAL orchestrator module; an `ensure_registry_seeded`
file is read back by the runner's registry as armed + empty). A non-default
*relative* `execution.software_stops.registry_path` would still diverge
from the checker's `DEFAULT_REGISTRY_REL`; no config sets one.

## 3. Inertness — the disabled path is the old path, literally

`software_stops_wiring.py`:

- `software_stops_enabled(config)` `:60-72` — the pipeline exposes no public
  enabled-check, so this is a verbatim mirror of `from_config`'s gate
  (`software_stops.py:326-328`: `((config or {}).get("execution") or
  {}).get("software_stops") or {}` then `.get("enabled", False)` truthiness).
  Pure dict read: imports nothing, logs nothing, touches no disk.
- `build_software_stop_registry` `:114-115` returns `None` on that gate
  BEFORE the pipeline import, the orchestrator import, the root resolution
  and any log line. `from_config` still re-applies its own gate afterwards
  (`:133-134` defers to a `None`), so the pipeline stays authoritative — the
  mirror can only make the wiring MORE inert, never less.
- Parity of the mirror with the real gate is pinned for 10 OFF shapes
  (`None`, `{}`, `execution: None/{}`, `software_stops: None/{}`,
  `enabled: False/None/0/""`) and 3 ON shapes (`True/1/"yes"`): OFF →
  `from_config` returns `None` with a `repo_root` whose `__fspath__`/
  `__str__` raise; ON → a registry under a tmp root, no file written
  (`TestEnabledGate::test_parity_with_the_real_pipeline_gate`).
- `TestFlagOffByteInert`: for every OFF shape, with the orchestrator
  contract monkeypatched to RAISE on import, the result is `None`, the
  pipeline constructor is never called, `caplog.records == []` at DEBUG,
  the neutral root does not exist and the cwd is empty; the same holds with
  the pipeline module itself unimportable. Through the real
  `RunnerAdapter.__init__` (preflight): `_software_stops is None`, no
  software-stop log line, contract never imported.
- Source-order pin (`test_gate_is_read_before_any_import_in_source`): in the
  function body (docstring stripped) no `import`, no `log.`, no
  `from_config`, no root resolution precedes the gate.

Only an ENABLED layer imports the orchestrator LOCATION contract
(unimportable → `NeutralRootUnavailable` `:88-96` → ONE ERROR + `None`
`:126-132`; `from_config` never called, no cwd fallback —
`TestContractImportFailureFailsClosed`, ON shapes only), resolves the root,
and calls `from_config(..., repo_root=root)` `:123-125`.

## 4. Seed-before-runner in the sell wrapper, and why it never blocks

`scripts/intraday_sell_104.sh:109-144`: after the pinned `PYTHONPATH`
export (`:58`), `cd "$REPO_DIR"` (`:84`), the pin-alignment preflight
(`:90`) and the runtime sanity gate (`:100-107`) — so it runs on the audited
pins — and immediately before the runner (`--sell-only --intraday`):

```
SEED_OUT=$("$PYTHON" -m "$SEED_MODULE" seed --broker alpaca 2>&1)
SEED_RC=$?
```

- exit 0 + a `SEEDED:`/`EXISTS:` line → one log line, continue (the
  expected state on the pinned assembly);
- exit 0 without a verdict line → `WARNING … runtime assembly drifted below
  the lock?` — a drift alarm, not an expected state now that the pin
  carries the seeder; continue;
- any non-zero (1 usage, 2 existing file CORRUPT — left untouched, 3
  pipeline module not importable) → `ERROR … continuing with the sell pass`
  + `notify` page, continue.

Why it can never block: `set -uo pipefail` (`:19`, no `-e`), no `exit` and
no `return` in the block, the runner invocation follows unconditionally.
This loop is the live book's EXIT path; a seed is bootstrap plumbing whose
failure is page-worthy, not a reason to skip selling. Same broker by
construction: both lines carry the literal `--broker alpaca`, so the
header's documented paper-restore `sed` rewrites both
(`test_same_broker_as_the_runner`). The failure `notify` is per run (every
~12 min) while the seed keeps failing — a persistent failure should page.

## 5. Test evidence `[VERIFIED pytest, RenQuant/.venv py3.10.20, -p no:cacheprovider]`

`tests/test_software_stops_neutral_root.py` (54 tests; third step of
`.github/workflows/live-broker-fractional-contract.yml`, path filters cover
`adapters/runner.py`, `adapters/software_stops_wiring.py`,
`scripts/intraday_sell_104.sh`, the test file):

- against the PINNED runtime assembly (`RENQUANT_SUBREPO_ROOT=<live
  repo>/.subrepo_runtime/repos`, orchestrator `64238032`, `-B` +
  `PYTHONDONTWRITEBYTECODE=1`; the runtime checkout stayed clean):
  **54 passed, 0 skipped** — both real-sibling parity tests and the three
  assembly-level tests ran.
- default isolated worktree (no assembly; lock `local_path` orchestrator is
  a research checkout without the seeder): **50 passed, 4 skipped**, every
  skip naming its reason.
- lean CI simulation — throwaway venv with ONLY `pytest` (py3.9.6),
  `-o addopts=''` as the workflow runs it: **47 passed, 7 skipped** (2
  parity, 3 pandas, 2 assembly); the gate/inertness/fail-closed/wrapper
  cases ran against the stubs.
- the four software-stops files (`test_software_stops_neutral_root`,
  `test_s_frac_stage0_commit_contract`, `test_software_stops`,
  `test_runner_preflight_adapter`) against the pinned assembly:
  **124 passed, 0 skipped, 0 failed**.
- the runner-importing set (25 files importing `adapters.runner` +
  `test_software_stops.py` + the new file) under the repo `pytest.ini`
  (`-n auto`) from the isolated worktree: **601 passed, 4 skipped, 18
  failed, 1 collection error**; the FAILED/ERROR set is **byte-identical**
  to the pristine `origin/main` (`f9d696a`) run recorded in revision 1
  (empty `diff` against the saved list). Those are environmental in an
  isolated worktree (`renquant_pipeline` not on the path →
  `test_software_stops.py` collection; sibling-relative resolution in
  `test_live_multirepo_entrypoints` x3, `test_adapter_context_contract` x7,
  `test_panel_alignment` x4, `test_feature_cache` x2,
  `test_cusum_cooldown_v2` x1, `test_state_store` x1). The committed
  `backtesting/__pycache__/__init__.cpython-310.pyc` is unmodified.
- shell: `bash -n` clean; the extracted seed block executed under bash for
  all six seeder outcomes reaches the runner line every time and pages only
  on non-zero.

## 6. Chain position

```
orch#1078 (MERGED) bootstrap: seeder + readiness classification + installer requires READY
  → [this PR] umbrella: enabled-gate-first wiring, runner from_config(repo_root=neutral),
    wrapper seeds before the runner, orchestrator pin → 64238032 (seed exists on the pinned path),
    assembly-level regression on the pinned module
  → observe READY at the canonical path (`… readiness --broker alpaca`) during a live loop
  → SLA drill (test-fire STALE)
  → install_stops_pager.sh install --apply  (guard: VALID and READY)
  → stage-3 sign-off → fractional flip under its own LONG-ledger row
```

Not in this PR: `daily_104.sh` constructs the same `RunnerAdapter` (so it
also resolves the neutral root when enabled) but does not seed — the
sell-only loop is the designated writer; the seed is idempotent. No memory
tier file covers this line (LONG row 2b text); this doc is the durable
record.

## 7. Files

- `backtesting/renquant_104/adapters/software_stops_wiring.py` (new, stdlib-only imports)
- `backtesting/renquant_104/adapters/runner.py:223-238`
- `scripts/intraday_sell_104.sh:109-144`
- `tests/test_software_stops_neutral_root.py` (new)
- `.github/workflows/live-broker-fractional-contract.yml`
- `subrepos.lock.json`, `doc/arch/strategy-104-snapshot.md` (pin promotion)
