# 2026-08-29 — Software-stop registry at the NEUTRAL root; the sell wrapper seeds before the runner (orch#1078 follow-ups 1 + 2)

**Bottom line:** the live runner now resolves the software-stop registry
under the neutral runtime-state root (`~/.renquant/runtime/software-stops`,
override `RENQUANT_RUNTIME_STATE_ROOT`) — the `--data-root` the liveness
pager and the orchestrator seeder/readiness classifier read — instead of the
process cwd, and it FAILS CLOSED (registry `None`, one ERROR line, capability
gate unarmed) when the orchestrator LOCATION contract is not importable; it
never falls back to the cwd. `scripts/intraday_sell_104.sh` calls the
orchestrator seeder unconditionally before the runner with the same
`--broker alpaca` the runner receives, and a failed seed is logged + paged
but NEVER blocks the sell pass. No flag is flipped: `execution.software_stops`
is configured nowhere in this repo and `enabled` stays false; the flag-off
path is byte-identical (`from_config` returns `None` before it reads
`repo_root`). Nothing installed, no live tree or `-run` checkout touched.

Effective-in-production caveat `[VERIFIED]`: the runtime assembly pins
renquant-orchestrator at `75dd9c70` (`subrepos.lock.json`;
`.subrepo_runtime/repos/renquant-orchestrator` HEAD = `75dd9c70`), which
PREDATES the seeder (orch#1078). At that pin the contract module has
`runtime_state_root` / `software_stops_registry_root` (so the runner-side
change works as soon as this lands) but no `seed` CLI: `python -m … seed
--broker alpaca` exits 0 with no output (measured on the pinned module
source). The wrapper reports that case as a WARNING ("exited 0 WITHOUT a
SEEDED/EXISTS verdict"), not as green. The seed becomes effective when the
orchestrator pin advances past #1078 (the landing PR of the chain below).

## 1. The path unification

| who | resolution before | resolution now | where |
|---|---|---|---|
| liveness checker (execution) | `registry_path_for(Path(data_root)/DEFAULT_REGISTRY_PATH, broker)`, `data_root` = plist value `~/.renquant/runtime/software-stops` | unchanged | `renquant-execution/src/renquant_execution/software_stops_liveness.py:314-336` (`resolve_registry_path`) |
| seeder / readiness (orchestrator, #1078) | `seeded_registry_path(data_root, broker)` = `Path(data_root)/"data/rq105/software_stops.json"` broker-tagged; default `data_root` = `software_stops_registry_root(runtime_state_root())` | unchanged | `renquant-orchestrator/src/renquant_orchestrator/software_stops_registry_contract.py:103-126` (root), `:194` (`DEFAULT_REGISTRY_REL`), `:263-286` (`seeded_registry_path`), `:308` (`ensure_registry_seeded`), `:624-630` (CLI default data root) |
| live runner (this repo) | `SoftwareStopRegistry.from_config(config, broker_name=…)` with NO `repo_root` → relative `registry_path` resolved against cwd = the umbrella checkout (`scripts/intraday_sell_104.sh:84` `cd "$REPO_DIR"`) | `from_config(config, broker_name=…, repo_root=software_stops_registry_root(runtime_state_root()))` | `backtesting/renquant_104/adapters/runner.py:223-238` → `adapters/software_stops_wiring.py:54-76` (root), `:79-113` (construction) |

The pipeline composes `Path(repo_root) / registry_path` then broker-tags it
(`renquant-pipeline/src/renquant_pipeline/software_stops.py:316-337`,
`from_config`; `:141-154` `registry_path_for`; `DEFAULT_REGISTRY_PATH =
"data/rq105/software_stops.json"` `:121`). With `repo_root` = the neutral
data root and the default `registry_path`, the runner's file is
`~/.renquant/runtime/software-stops/data/rq105/software_stops.alpaca.json` —
byte-equal to the checker's and the seeder's composition. Pinned by
`tests/test_software_stops_neutral_root.py::TestFlagOnNeutralRootParity`
against the REAL orchestrator module (`seeded_registry_path`) and the REAL
pipeline registry: the runner's `registry.path == seeded_registry_path(root,
broker)`, and a file written by `ensure_registry_seeded` is read back by the
runner's registry as armed + empty. (A non-default *relative*
`execution.software_stops.registry_path` would still diverge from the
checker's `DEFAULT_REGISTRY_REL`; no config sets one. Not addressed here.)

## 2. Inertness argument (why production behaviour is unchanged today)

1. `execution.software_stops` is set in no config in this repo (grep of
   `config/`, `backtesting/renquant_104/config*`, the pinned strategy-104
   configs: zero hits) → `enabled` is absent everywhere.
2. `from_config` gates on `enabled` BEFORE it reads `registry_path` or
   `repo_root` (`software_stops.py:327-328` returns `None`; `:329-333` is
   where `registry_path` and then `repo_root` are first used). Proved
   behaviourally, not by reading:
   `test_from_config_returns_none_before_touching_repo_root` passes a
   `repo_root` whose `__fspath__`/`__str__` raise, for `enabled` absent,
   false, `{}` and `None` — `from_config` returns `None` without raising.
3. Therefore `RunnerAdapter._software_stops` is `None` exactly as before,
   `commit_contract.software_stops_armed(None)` is `False`, no file is
   created under the neutral root or the cwd
   (`TestFlagOffByteInert::test_registry_none_and_from_config_got_the_neutral_root`
   asserts both directories stay empty), and the stage-0 capability gate
   keeps refusing fractional BUY intents.
4. The one observable delta with the flag off is the resolution of the
   neutral root itself (a pure path computation from env/default; nothing
   is created) and, ONLY if the orchestrator contract is unimportable, one
   ERROR line — which is the designed loud failure, and cannot occur on the
   pinned runtime because `scripts/intraday_sell_104.sh:58` (and
   `daily_104.sh`) put the pinned orchestrator `src` on `PYTHONPATH`
   before the runner, and the two names the wiring imports exist at the
   current pin (`75dd9c70`).
5. Fail-closed rather than fall-back: when the contract is missing the
   wiring raises `NeutralRootUnavailable` (`software_stops_wiring.py:67-75`)
   and `build_software_stop_registry` turns it into `None` + ERROR
   (`:100-106`); the pipeline constructor is never called — a registry
   resolved against the cwd is a writer the pager cannot see, i.e. a dark
   pager, so refusing is safer than guessing
   (`TestContractImportFailureFailsClosed`, flag off AND on: `from_config`
   call list is empty, cwd stays empty).

## 3. Seed-before-runner in the sell wrapper, and why it never blocks

`scripts/intraday_sell_104.sh:128-140`: after the pinned `PYTHONPATH`
export (`:58`), the `cd "$REPO_DIR"` (`:84`), the pin-alignment preflight
(`:90`) and the runtime sanity gate (`:100-107`) — so it runs on the audited
pins — and immediately before the runner (`:142-143`,
`--sell-only --intraday`):

```
SEED_OUT=$("$PYTHON" -m "$SEED_MODULE" seed --broker alpaca 2>&1)
SEED_RC=$?
```

- exit 0 + a `SEEDED:`/`EXISTS:` line → one log line, continue;
- exit 0 without a verdict line (pinned orchestrator predates #1078) →
  `WARNING … registry NOT confirmed locatable`, continue;
- any non-zero (1 usage, 2 existing file CORRUPT — left untouched, 3
  pipeline module not importable) → `ERROR … continuing with the sell pass`
  + `notify` page, continue.

Why it can never block: the script runs `set -uo pipefail` (`:19`, no
`-e`), the block contains no `exit` and no `return`, and the runner
invocation follows unconditionally. The rule is deliberate: this loop is
the live book's EXIT path (stop-loss / trailing / SDL / max-hold sells on a
12-minute cadence); a seed is bootstrap plumbing whose failure is a
page-worthy fact about the pager's bootstrap, not a reason to skip selling.
Same broker by construction: the seed and the runner both carry the literal
`--broker alpaca`, so the header's documented paper-restore `sed` rewrites
both (`test_same_broker_as_the_runner` pins that the two are equal and are
the only two executable occurrences).

The `notify` on failure is not rate-limited; it fires per run (every ~12 min)
while the seed keeps failing. Today the pinned orchestrator cannot fail
(no CLI → exit 0 → WARNING only, no page); after the pin advances a
persistent failure is exactly what should page.

## 4. Test evidence `[VERIFIED pytest, RenQuant/.venv py3.10.20, -p no:cacheprovider]`

New file `tests/test_software_stops_neutral_root.py` (26 tests; wired into
`.github/workflows/live-broker-fractional-contract.yml` as a third step
`:71-74`, path filters extended to `adapters/runner.py`,
`adapters/software_stops_wiring.py`, `scripts/intraday_sell_104.sh`, the
test file `:31-38`/`:43-50`):

- with the merged orchestrator (`renquant-orchestrator-run/src`, HEAD
  `177446c0`) and the pinned pipeline/execution/common siblings on
  `PYTHONPATH`: **26 passed, 0 skipped** (both real-sibling parity tests ran).
- default isolated-worktree environment (lock `local_path` for the
  orchestrator is the research checkout, which lacks the seeder): **24
  passed, 2 skipped**, each skip naming the reason ("lacks
  ['seeded_registry_path', 'ensure_registry_seeded'] (pinned checkout
  predates the seeder, orch#1078)").
- lean CI simulation — throwaway venv with ONLY `pytest` (py3.9.6),
  `-o addopts=''` exactly as the workflow runs it: **22 passed, 4 skipped**
  (the 2 parity tests + the 2 `RunnerAdapter.__init__` end-to-end tests
  that need pandas); the wiring/flag-off/import-failure/wrapper cases ran
  there against the stubs — not vacuous.
- the four software-stops files together (`test_software_stops_neutral_root`,
  `test_s_frac_stage0_commit_contract`, `test_software_stops`,
  `test_runner_preflight_adapter`) with the real siblings: **96 passed, 0
  skipped, 0 failed**.
- the whole runner-importing set (the 25 test files that import
  `adapters.runner`, + `test_software_stops.py` + the new file) under the
  repo `pytest.ini` (`-n auto`) from the isolated worktree: **575 passed,
  2 skipped, 18 failed, 1 collection error**; a pristine `origin/main`
  (`f9d696a`) worktree run the same way: **551 passed, 18 failed, 1 error**
  and the FAILED/ERROR set is **byte-identical** (empty `diff`; the +24
  passed / +2 skipped are this file). The pre-existing failures measure the
  isolated worktree, not this branch: `renquant_pipeline` is not on the
  path (`test_software_stops.py` collection error), sibling-relative
  resolution in `test_live_multirepo_entrypoints` x3,
  `test_adapter_context_contract` x7, `test_panel_alignment` x4,
  `test_feature_cache` x2, `test_cusum_cooldown_v2` x1, `test_state_store`
  x1 (asserts a `save_live_state_atomic(...)` call text that main's
  `runner.py` does not contain). The pristine worktree was removed
  afterwards. The committed `backtesting/__pycache__/__init__.cpython-310.pyc`
  is unmodified.

Shell-level: `bash -n scripts/intraday_sell_104.sh` clean; the extracted
seed block executed under bash with a stub seeder for all six outcomes
(SEEDED / EXISTS / silent-0 / exit 1 / 2 / 3) reaches the line after the
block every time, logs the expected verdict, and pages only on non-zero
(`test_block_executes_and_continues_under_every_seeder_outcome`).

## 5. Chain position

```
orch#1078 (MERGED) bootstrap: seeder + readiness classification + installer requires READY
  → [this PR] umbrella: runner from_config(repo_root=neutral) + wrapper seeds before the runner
  → landing PR: advance the renquant-orchestrator pin past #1078 (subrepos.lock.json + runtime
    assembly) so `seed` exists on the pinned PYTHONPATH — the wrapper's WARNING branch goes quiet
  → observe READY at the canonical path (`… readiness --broker alpaca`) during a live loop
  → SLA drill (test-fire STALE)
  → install_stops_pager.sh install --apply  (guard: VALID and READY)
  → stage-3 sign-off → fractional flip under its own LONG-ledger row
```

Not in this PR: `daily_104.sh` constructs the same `RunnerAdapter` (so it
also resolves the neutral root once this lands) but does not seed — the
sell-only loop is the designated writer; the seed is idempotent and one
caller suffices. No memory tier file covers this line (LONG row 2b text);
this doc is the durable record.

## 6. Files

- `backtesting/renquant_104/adapters/software_stops_wiring.py` (new, stdlib-only imports)
- `backtesting/renquant_104/adapters/runner.py:223-238`
- `scripts/intraday_sell_104.sh:109-140`
- `tests/test_software_stops_neutral_root.py` (new)
- `.github/workflows/live-broker-fractional-contract.yml`
