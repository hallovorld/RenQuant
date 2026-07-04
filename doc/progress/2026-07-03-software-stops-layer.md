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
