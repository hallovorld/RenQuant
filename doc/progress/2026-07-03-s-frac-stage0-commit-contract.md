# S-FRAC stage 0 — fractional-capable RunnerAdapter.commit contract + active-path audits

STATUS: implemented, default-inert. Design:
renquant-orchestrator `doc/design/2026-07-02-s-frac-fractional-v2.md` (merged RFC),
§2.2 stage-0 deliverable + §2.3 audit tests + §6 stage-0 AC.

## Why stage 0 exists (the v1 lesson, inverted)

The v1 fractional chain (execution#19 / pipeline#153 / strategy#36, closed 2026-06-30)
built capability on a NON-ACTIVE path (`ExecutionPipeline`/`FakeBackend`) while the ACTIVE
live path — the umbrella `RunnerAdapter.commit` — still int-truncated fractional fills.
v2 inverts the build order: the active-path contract lands FIRST, before any subrepo
capability work restarts. Stage 0 makes the commit path fractional-SAFE, not
fractional-ACTIVE: no flag is enabled anywhere; whole-share behavior is regression-pinned
byte-identical.

## What changed

1. **Quantity contract (§2.2.1)** — the truncation kill. The verified line
   `shares = int(execution["filled_qty"] or shares)` (runner.py buy path; the exact line
   Codex cited to block pipeline#153 — a 0.435578 fill became 0 shares in orders_placed,
   live_state, journal, cash accounting, and the Z9 stop qty) is replaced by
   `commit_contract.normalize_fill_qty`: broker `filled_qty` is authoritative and
   float-preserved end-to-end; eps-integral fills snap to `int` (the ONE sanctioned
   whole-share branch → flag-off bytes identical). Display-side `%d` / `%.0f` qty casts in
   commit/Z9 logs became `fmt_qty` (byte-identical for whole shares).
2. **Stop routing contract (§2.2.2)** — `supports_broker_side_stops(symbol, qty)` is now
   qty-aware on `BaseBroker` / `AlpacaBroker` / `PaperBroker` / `ReadOnlyBrokerWrapper`
   (no-arg legacy answer unchanged; fractional qty ⇒ False — Alpaca fractional is TIF=DAY
   only, no GTC). `z9_stops.place_or_replace_stop` routes protection per held quantity via
   `commit_contract.route_stop_protection`, re-evaluated at every placement against the
   CURRENT qty (restart-safe, never cached): whole-share → broker GTC stop (unchanged);
   fractional + armed software layer (stage 3 seam: `RunnerAdapter._software_stops`) →
   software; fractional + no layer → loudly UNPROTECTABLE, never a truncated stop.
   A fractional BUY intent fail-closes at entry (`fractional_intent_flag_off` /
   `fractional_entry_unprotectable_no_stop_layer`) before any broker interaction.
3. **Capability gate (§2.2.3, the strategy#36 prose-gate blocker closed)** —
   `commit_contract.fractional_capability_gate`: `execution.fractional_shares.enabled=true`
   requires (a) the broker fractional contract (`is_fractionable` + no-submit classifier,
   from execution#19) and (b) the software-stop layer reporting armed. Either missing ⇒
   ALL BUY emission fail-closes with a dedicated `orders_skipped` audit reason
   (`fractional_capability_gate_failed:<missing>`); exits are never blocked.
4. **Active-path liveness proof (§2.3)** — `RunnerAdapter.commit` stamps
   `ctx.commit_path_fingerprint` (contract tag `fractional-v2-stage0` + sha256 of the
   executed runner source) on every commit; `build_run_bundle` records it in the persisted
   run bundle. "The live runner exercises the contract-carrying path" is now a per-run
   recorded fact — the direct anti-regression for merged-is-not-deployed /
   deployed-but-dark.
5. **Static truncation audit** — `scripts/check_commit_path_no_int_truncation.py`
   (AST-based, same pattern as the orchestrator's `check_model_bundle_consistency.py`):
   no `int(...)` cast on fill quantities across runner.py / runner_execmath.py /
   runner_ext_sell.py / broker_sync.py / z9_stops.py / commit_contract.py outside the
   sanctioned `normalize_fill_qty` / `fmt_qty` branches. Reintroduction under any spelling
   fails the audit.

## Files

- `backtesting/renquant_104/adapters/commit_contract.py` (new — single authority)
- `backtesting/renquant_104/adapters/runner.py` (commit: fingerprint stamp, gate,
  fail-closed entry, truncation kill, fmt_qty logs, software-stop seam)
- `backtesting/renquant_104/adapters/z9_stops.py` (qty-aware routing)
- `backtesting/renquant_104/kernel/artifact_contract.py` (bundle records fingerprint)
- `live/broker.py`, `live/alpaca_broker.py`, `live/paper_broker.py`,
  `live/broker_readonly.py` (qty-aware `supports_broker_side_stops`)
- `scripts/check_commit_path_no_int_truncation.py` (new — static audit)
- `tests/test_s_frac_stage0_commit_contract.py` (new — 44 tests)

## §2.3 audit-test verdicts (all green)

| # | Audit | Result |
|---|---|---|
| 1 | E2E real commit path: 0.435578 BUY round-trip (orders_placed/live_state/journal/exact-float cash) + fractional SELL → zero residual dust, wash-sale stamped, Z9 stop cancelled | PASS |
| 2 | Truncation audit: 6 commit-path modules clean; auditor proven to catch the legacy line + alternate spellings + missing-module | PASS |
| 3 | Active-path liveness: daily_104.sh/intraday_sell_104.sh → live.runner → `adapter.commit` chain walk; fingerprint stamped with executed-source sha; run bundle records it; sim ctx carries no false liveness claim | PASS |
| 4 | Flag-off regression: whole-share order dict == legacy `int()` semantics (values, int types, JSON bytes, no new fields); Z9 whole-share stop placement + "× 5 shares" log byte-identical | PASS |
| 5 | Partial fill held at exact float, order not marked terminal; cancel-replace legs float-sum (union of fills), entry state never overwritten | PASS |
| 6 | Stop-reconciliation-on-restart: capability probed against CURRENT qty (5.435578), never the cached pre-restart 5.0; fail-closed entry re-derived post-restart | PASS |
| 7 | Fail-closed entry ⇒ stage-0 outage-window loss budget $0 by construction: flag-on-without-deps blocks ALL buys pre-broker; flag-off fractional intent never submits; exits never blocked | PASS |

Plus: float-fill round-trip on a fake `broker_order_execution` result (string floats from
broker JSON preserved verbatim; fractional partial classification with dust epsilon) and
unit pins for `normalize_fill_qty` / `fmt_qty` / routing / gate fail-closed probes.

## Test evidence

- `tests/test_s_frac_stage0_commit_contract.py`: 44/44 pass.
- Touched-module regression suites: `test_broker_side_stops`, `test_z9_stops`,
  `test_runner_z9_integration`, `test_z9_catastrophe_pct`, `test_broker_nan_guards`,
  `test_state_ext_sell_fill_attribution`, `test_partial_sell`, `test_artifact_contract`,
  `test_runner_state_fixes`, `test_runner_commit_save_state_config_arg` (186 pass) +
  `test_adapter_context_contract`, `test_persistence`, `test_no_trade_monitor`,
  `test_sim_live_parity`, `test_runner_trade_ntfy`, `test_live_multirepo_entrypoints`
  and the wider audit suites — green.
- Pre-existing failures (4: `test_audit_2026_04_24_fixes` ×3, `test_feature_cache` ×1)
  verified IDENTICAL with and without this diff (git-stash A/B, serial run) — sibling-
  checkout skew vs pins (e.g. scorer `ctx` kwarg drift), unrelated to this change.
- `python scripts/check_commit_path_no_int_truncation.py` → OK (exit 0).
- `py_compile` on all touched modules → clean.

## What stage 1 consumes (the seams this PR reserves)

- `supports_broker_side_stops(symbol, qty)` — the umbrella caller now exists (the #19
  round-2 "unconsumed capability" blocker is closed from the consumer side); the rebased
  execution#19 branch lands its broker-side impl against this exact signature.
- `fractional_capability_gate`'s broker probe (`is_fractionable` + `classify_broker_result`
  / `is_no_submit_status`) — the #19 contract surface, expected on the live broker adapter.
- `normalize_fill_qty` semantics for DAY-expiry / float requested-vs-filled comparison
  (§6 stage 1 scope; the terminal-canceled-with-first-sight-of-fill classification is
  explicitly stage-1, noted in the cancel-replace test).
- Stage 3 attaches the software-stop registry at `RunnerAdapter._software_stops` with
  `is_armed() -> True`; the gate and the Z9 router already consume it.

## Kill/defer check (§6 stage 0)

The commit-path diff stayed small and reviewable (runner.py commit body: one killed line,
one stamped fingerprint, one gate + one fail-closed check in the buy loop, log formatting;
everything else lives in the new single-authority module + tests). The 06-30 "larger,
higher-risk undertaking" concern did not re-materialize.
