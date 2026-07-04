# Software stop layer — ratchet-only registry + sell-only loop wiring (sprint D2)

DATE: 2026-07-03
SCOPE: S-FRAC stage-3 CORE (the protection layer stage 0 fail-closes on) + the
Stage-2 live-executor dependency. Design: renquant-orchestrator
`doc/design/2026-07-02-s-frac-fractional-v2.md` §3.2–§3.4, §6 stage 3.
FLAGS: `execution.software_stops.enabled` — default OFF everywhere; no config
anywhere enables it. Flag-off is byte-inert (no registry object, no file, no
behavior change; pinned by test).

## What shipped

1. **`adapters/software_stops.py`** — `SoftwareStopRegistry`: persisted stop
   registry (default `data/rq105/software_stops.json`, broker-tagged like all
   live state files) with per-position schema
   `{symbol, qty (float-capable), stop_price, armed_at, source: z9|manual|
   fractional-auto, history[]}` plus file-level `last_evaluated_at` heartbeat
   and `max_staleness_minutes` watchdog budget.
   - `is_armed()` — satisfies the stage-0 `commit_contract.software_stops_armed`
     probe (PR #439 seam consumed, not reimplemented).
   - **Never-loosen**: `register()` only ratchets `stop_price` UP; a lower
     proposal is refused and recorded (`ratchet_refused`). Loosening requires
     `rewrite_stop(symbol, price, reason=...)` — explicit, non-empty reason,
     logged + persisted in the entry history. (Deliberate difference from Z9's
     broker-stop `min()` convention, documented in the module.)
   - `evaluate(live_quotes)` → market-exit intents for the FULL registered qty
     when price ≤ stop; gap-down-through-stop fires however deep the print is,
     with the gap size measured, logged, and carried on the intent (§3.3:
     slippage accepted, never hidden).
   - **Corruption fail-closed**: unreadable/schema-invalid registry ⇒ NOT
     armed (stage-0 gate blocks every new fractional BUY), all writes raise,
     the corrupt bytes are preserved, and every evaluate pass logs ERROR.
2. **Wiring (flag-gated, default-OFF)**:
   - `RunnerAdapter.__init__` constructs the registry from config (broker-
     tagged path); `make_context` attaches `ctx.software_stops`.
   - `z9_stops.place_or_replace_stop`: when stage-0 routing selects
     `"software"`, the stop is REGISTERED (same `reference × (1 − pct)` math
     as the broker path) instead of logged-and-dropped. Entry commit + top-up
     both flow through this one router.
   - `z9_stops.cancel_stop` + full liquidation: registry entry disarmed;
     STATE-GC disarms entries for externally-disposed positions.
   - `SellOnlyPipeline` (the 12-min intraday loop's pipeline) runs the new
     `kernel/pipeline/task_software_stops.SoftwareStopExitTask` AFTER
     MetaLabelVetoTask + LimitSellsPerBarTask — a broker-resident stop can't
     be vetoed or capped, so its software mirror isn't either.
   - `kernel/exit_types.py`: `software_stop` added to PATH_RULE_SYNONYMS and
     POST_STOP_COOLDOWN_TRIGGERS (a software-stop breach IS a price stop:
     veto-bypass, cap-exempt, post-stop re-entry blackout).
3. **Failure-mode handling (§3.3/§3.4)**:
   - Gap-down-through-stop ⇒ market exit at next pass, gap logged.
   - Loop-dead ⇒ `scripts/check_software_stops_liveness.py` (ops file only,
     nothing installs it): exit 0 OK / 1 STALE (armed stops + heartbeat older
     than budget during a market session) / 2 CORRUPT; optional `--ntfy-topic`.
   - A breached stop stays registered until the exit is broker-confirmed —
     a failed SELL re-fires next pass instead of silently unprotecting.

## Tests (`tests/test_software_stops.py`, 36 tests)

Ratchet-only invariant; trigger correctness incl. fractional qty; gap-through
pricing; flag-off inertness on the sell-only loop AND on commit; registry
round-trip; corruption fail-closed proven on the REAL commit path (corrupt
registry ⇒ `fractional_capability_gate_failed:software_stop_layer`, no order
reaches the broker); staleness watchdog arithmetic + CLI exit codes; commit
e2e through the stage-0 seam (fractional BUY registers → breach → sell-only
task queues → commit sells full fractional qty, stamps wash-sale + post-stop
blackout, disarms registry, zero residual); top-up never-loosens through the
Z9 route; external-disposition GC.

Ran (scratchpad clone, live venv, subrepo PYTHONPATH): the new suite +
stage-0 suite (`test_s_frac_stage0_commit_contract`) + Z9 suites
(`test_z9_stops`, `test_broker_side_stops`, `test_runner_z9_integration`,
`test_z9_catastrophe_pct`) + exit-type consumers (`test_exit_types_module`
[one pin updated for the taxonomy addition], `test_limit_sells_per_bar`,
`test_post_stop_blackout_g8`, `test_meta_label_veto`, `test_panel_rank_veto`)
+ commit-path suites (`test_runner_ext_sell`, `test_partial_sell`,
`test_buy_sell_audit_fixes`, `test_broker_sync`, …) + every suite driving
SellOnlyPipeline. A/B vs pristine `origin/main`: identical pass/fail sets —
the only failures (1 in `test_meta_label_veto` pipeline guard, 3 in
`test_audit_2026_04_24_fixes`) pre-exist on main in this environment.

## What downstream stages consume from this

- **S-FRAC stage 2 (fractional sizing)**: with this layer enabled the stage-0
  capability gate's `software_stop_layer` requirement is satisfiable — the
  machine-verifiable precondition for any fractional BUY. Stage 2 needs no
  stop code: entry commit already registers via the Z9 router.
- **S-FRAC stage 3 enablement packet**: the registry + evaluator + watchdog
  are the §6 stage-3 mechanics; remaining for enablement = shadow evidence,
  `fractional_max_book_pct` cap, optional DAY-stop belt, pager demonstration
  (§3.4 SLA) — all still default-OFF decisions for the operator.
- **Stage-2 live executor (105)**: a fractional (or otherwise broker-
  unprotectable) position acquired intraday gets loop-resident protection via
  the same `register()`/`evaluate()` surface; `source: fractional-auto|manual`
  are reserved for those writers.

## Rollback

Flag stays OFF (nothing to roll back live). If enabled later and reverted:
disable the flag — existing registry entries remain readable, exits are never
blocked, and the watchdog goes quiet once entries are disarmed.

## Round 2 (codex review — subrepo boundary)

Codex flagged this PR for editing `backtesting/renquant_104/...` source
directly without "the corresponding lock/declaration update surface you'd
expect from a subrepo-consumption change." Investigated rather than assumed:

- `CLAUDE.md` §3.5 documents a **Phase-1 invariant**: production execution
  flows through the umbrella's `kernel.*` imports, text-mirrored
  byte-for-byte with `renquant-pipeline`'s `kernel/pipeline/` — confirmed by
  diffing both directories' file listings, which match file-for-file except
  this PR's new `task_software_stops.py` (present here, missing there).
  That's the actual gap codex's review was pointing at: not "code in the
  wrong repo" in general (dozens of pre-existing `task_X.py`/`job_X.py`
  files already live in both places by design), but an **incomplete paired
  landing** — this PR shipped only the umbrella half of the mirror.
- Filed the missing half: **`renquant-pipeline#165`** —
  `kernel/pipeline/task_software_stops.py` (byte-identical, adjusted for this
  repo's import convention), the same two `kernel/exit_types.py` frozenset
  additions, and the equivalent `SellOnlyPipeline` wiring in
  `pp_inference.py`. 1180/1180 pipeline tests pass.
- `adapters/software_stops.py` (the registry class) is confirmed umbrella-only
  by design — no `adapters/` directory exists in `renquant-pipeline` or
  `renquant-execution` at all, and `task_software_stops.py` never imports the
  registry directly (duck-types via `getattr`), so no mirror is needed there.
- Checked `#439` (referenced as a prior similar-shaped landing) for the same
  pattern: it touches only `adapters/` and `live/`, neither of which has a
  subrepo mirror — not an instance of this gap.
- No code removed from this PR: Phase-1's invariant requires BOTH copies to
  exist, so the umbrella's `task_software_stops.py`/`exit_types.py`/
  `pp_inference.py` changes stay as-is; `renquant-pipeline#165` is the
  missing companion, not a replacement.

## Round 3 (kernel-alias analysis — which copy actually RUNS on the live loop)

Round 2's framing ("production execution flows through the umbrella's
`kernel.*` imports") is only true for the LEGACY runner mode. Verified against
the actual live sell-only entry chain, read end-to-end:

- `scripts/intraday_sell_104.sh` (the 12-minute launchd loop) defaults to
  `RQ_DAILY_RUNNER=multirepo` → `renquant_orchestrator live-bridge` →
  `renquant_orchestrator/live_bridge.run_bridge` →
  `bootstrap_multirepo(repo_root)`, which walks the PINNED renquant-pipeline
  checkout's `kernel/` and force-installs
  `sys.modules["kernel.<mod>"] = renquant_pipeline.kernel.<mod>` for every
  kernel module — including `kernel.pipeline` (the whole package, so all its
  submodules resolve from the pinned tree) and `kernel.exit_types` — BEFORE
  `live.runner` is imported.
- `live/runner.py` then does `from kernel.pipeline import ... SellOnlyPipeline`
  (≈ line 419), which resolves to the pinned renquant-pipeline copy, NOT
  `backtesting/renquant_104/kernel/`.

Consequence: this PR's umbrella `kernel/` additions are SHADOWED on the live
sell-only loop. Had the pipeline companion not landed, `SoftwareStopExitTask`
would have been deployed-but-dark on the exact loop it was built for (the
same aliasing precedent documented on #436). The live authority for the
task + wiring + exit-type additions is **renquant-pipeline#165 (MERGED
2026-07-04T01:45Z)**; the umbrella copies in this PR are parity mirrors and
now carry header comments saying so.

Ownership story (tightened per round-2 review):

- **renquant-pipeline** — `kernel/pipeline/task_software_stops.py`,
  `pp_inference.py` SellOnlyPipeline wiring, `kernel/exit_types.py`
  additions: the LIVE authority on the multirepo path. Its tests exercise
  the task against a duck-typed fake registry — the task's only coupling to
  the umbrella is the `ctx.software_stops` CONTEXT CONTRACT (`getattr`,
  absent/None ⇒ no-op); no umbrella import crosses the repo boundary.
- **umbrella `backtesting/renquant_104/kernel/`** — Phase-1 parity mirror:
  executed by sim/backtest and the legacy `RQ_DAILY_RUNNER=umbrella`
  fallback, pinned by this repo's test suite.
- **umbrella `adapters/` + `live/` + `scripts/`** — the bridge/state layer
  (registry, Z9 router, RunnerAdapter wiring, liveness watchdog):
  umbrella-owned; `adapters/` has no subrepo mirror anywhere.

Merge order + deploy path:

1. `renquant-pipeline#165` — merged first (done).
2. This PR (umbrella mirrors + adapters layer).
3. Deployment when the flag is ever enabled: bump the renquant-pipeline pin
   in `subrepos.lock.json` past #165 (current pin `df7bc07` predates it) +
   local runtime sync. Until then the pinned SellOnlyPipeline simply has no
   software-stop pass — and if the flag were enabled before the pin bump,
   registered stops would go unevaluated but NOT silently: the heartbeat
   never stamps, so `check_software_stops_liveness.py` pages STALE (exit 1).
   Flag-off inertness holds in BOTH repos regardless of pin state (pinned
   1180/1180 pipeline suite green at merged main; umbrella suites re-run
   after the mirror-marker edits).

## Round 4 (codex review — "the ownership story isn't tightened enough yet")

Codex's follow-up: the pipeline companion fixes the missing mirrored task, but
"the remaining heavy surface here is still umbrella-side implementation: a new
stop registry, ops watchdog, runner wiring, and tests... far beyond a pin/update
or thin integration change." Re-investigated each piece skeptically rather than
re-asserting round 3's conclusion:

- **`adapters/software_stops.py` (589 lines) — checked for a duplicate concept
  in renquant-execution/renquant-pipeline first.** Neither repo has any
  `SoftwareStop`/`stop_registry` concept (`grep -rl "SoftwareStop\|stop_registry"`
  — zero hits in both). renquant-execution's only stop-shaped API
  (`supports_broker_side_stops`/`place_stop_order` in `broker.py`,
  `alpaca_broker.py`, `readonly_broker.py`) is the OPPOSITE mechanism this
  registry exists because it's unavailable: broker-native GTC stops, which this
  module's own docstring states fractional Alpaca orders cannot use
  (TIF=DAY only). Not a duplication — a genuinely distinct capability with no
  home elsewhere to duplicate.
- **Checked `adapters/`'s existing size/contents for precedent** (this had not
  been done in rounds 2-3): the directory already holds 30 files / 9,562 lines
  of exactly this class of code — `runner.py` (2,186 lines), `sim.py` (2,475
  lines), `runner_tax_lots.py`, `sim_cash.py`, `runner_execmath.py`, and
  directly relevant, the PRE-EXISTING `z9_stops.py` (215 lines, unmodified by
  this PR except its integration point) — the stop ROUTER this registry
  plugs into. `software_stops.py` is not a new pattern; it's the same kind of
  file as its own direct sibling, in the same directory, at a comparable size.
- **Re-read the actual `z9_stops.py`/`runner.py` diffs line-by-line** (not
  just their existence) to check codex's specific claim that "runner wiring"
  is itself heavy logic, not integration. It isn't: `z9_stops.py`'s diff is
  `getattr(software_stops, "register"/"deregister", None)` duck-typed calls
  with fail-closed error logging if the interface is missing or raises —
  no stop-evaluation decision logic lives here. `runner.py`'s diff
  constructs the registry from config (try/except → None on any failure) and
  threads the object through `ctx.software_stops` / `cancel_stop` — again,
  wiring, not business logic. The actual stop-evaluation/ratchet/fail-closed
  logic lives entirely in `software_stops.py` itself.
- **Checked `scripts/check_software_stops_liveness.py` against existing
  precedent**: this repo's `scripts/` already has 10+ umbrella-top-level
  `check_*.py`/`.sh` ops scripts of the identical shape (`check_config_drift.py`,
  `check_launchagents.py`, `check_lock_pins_ci_green.py`,
  `check_ops_deployment_ready.py`, `check_retrain_triggers.py`,
  `monitor_panel_health.sh`, …). Not a new umbrella-resident pattern.

**Conclusion, held after genuine re-investigation, not re-asserted:** round 3's
ownership table stands. `adapters/software_stops.py` is a new instance of an
already-established, precedented class of umbrella-resident execution logic
(consistent with 9,562 existing lines in the same directory, including the
sibling it integrates with); `z9_stops.py`/`runner.py`'s changes are genuinely
thin wiring, not relocated business logic in wiring's clothing; the ops
watchdog matches 10+ existing scripts. No new relocation target was found —
if codex's underlying objection is actually to the `adapters/` directory's
existing size/shape as a whole (i.e., the general principle "new code belongs
in the owning repo" applied retroactively to already-established, pre-existing
umbrella architecture), that is a pre-existing architectural question spanning
dozens of files well beyond this PR's diff, not something this PR introduced
or should be blocked on unilaterally resolving.

## Round 5 (codex review, unchanged) — actually relocated this time

Codex's round-4 response: the evidence "strengthens the narrative" but does not
change the conclusion — "not on lack of documentation." The ask was explicit:
relocate the substantive stop-layer logic out of the umbrella, or narrow this
PR to a thin integration/pin surface. Rounds 3-4 answered "does a duplicate
already exist elsewhere" (no) rather than "should this be authored in an
owning repo regardless" (codex's actual question) — the wrong test. This round
actually moves the code, following the exact precedent that worked on the
first try for `renquant-orchestrator#291`'s `AlpacaBrokerPort` extraction.

**Moved:** `SoftwareStopRegistry` + `compute_staleness` + `registry_path_for`
+ `SoftwareStopRegistryCorrupt` + `DEFAULT_MAX_STALENESS_MINUTES` — the
entire `adapters/software_stops.py` module — to
`renquant_pipeline.software_stops` (renquant-pipeline#167). Chosen over
renquant-execution because:

- The registry's only external dependency (`kernel.state_paths._safe_broker`)
  already exists byte-identically in renquant-pipeline as the Phase 1 mirror.
- `renquant_pipeline` is already an established, working cross-repo import
  source for this umbrella's live-runner tree (`adapters/runner.py` already
  imported `renquant_pipeline.kernel.gate_registry` before this change, via
  the exact same lazy-import pattern the old local `adapters.software_stops`
  import used). `renquant_execution` has no existing import wiring from this
  tree at all — that would have been first-of-its-kind, not established.

**Umbrella side, reduced to thin wiring:**

- `backtesting/renquant_104/adapters/software_stops.py` — deleted entirely
  (no shim kept; both real import sites updated directly).
- `adapters/runner.py:175` — `from adapters.software_stops import
  SoftwareStopRegistry` -> `from renquant_pipeline.software_stops import
  SoftwareStopRegistry`. Everything else in this file (the flag-gated
  construction, the try/except fail-closed-on-construction-failure logic) is
  unchanged — it was already thin wiring around the constructor, not registry
  logic itself.
- `scripts/check_software_stops_liveness.py` — same import change, plus
  removed the now-unnecessary `sys.path` manipulation (the module is consumed
  as an installed package now, not a local umbrella module).
- `z9_stops.py`/`kernel/pipeline/task_software_stops.py` — docstring/comment
  references to the old `adapters/software_stops` path updated; no functional
  change (both already consumed the registry via duck-typing/dependency
  injection, never a direct import).

**Test split:** the registry's own unit-test coverage (round-trip,
ratchet-only, trigger correctness, gap pricing, corruption schema-validation,
staleness arithmetic) moved to renquant-pipeline#167's own
`tests/test_software_stops.py`, alongside its already-existing
`SoftwareStopExitTask` wiring tests from renquant-pipeline#165 (kept as-is —
that task never imported the registry class directly, so relocating the
registry doesn't affect those tests). This umbrella's own
`tests/test_software_stops.py` keeps only what's genuinely umbrella-only:
flag-off byte-inertness, the stage-0 capability-gate integration test (a
corrupt registry blocking a REAL `RunnerAdapter.commit`), the ops watchdog
CLI script's own exit-code test, and the full commit-path E2E.

**Verified:** 15/15 in this repo's trimmed `test_software_stops.py`, 111/111
across the broader `software_stop`/`s_frac`/`z9` test surface (1 unrelated
pre-existing collection error in `test_per_regime_sigma_wire.py`, confirmed
reproducing identically on unmodified `origin/main`). 29/29 +
1249/1249 in renquant-pipeline#167.
