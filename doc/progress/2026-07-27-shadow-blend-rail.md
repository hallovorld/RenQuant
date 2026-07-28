# shadow_blend rail — parameterized readonly tag + daily Step 5

STATUS:    delivered (rail lands DORMANT by design — auto-activates when the
blend strategy profile appears in the pinned configs; see NEXT)
WHAT:      Umbrella-side rail for a `shadow_blend` full-funnel shadow lane,
cloned from the legacy PatchTST shadow (daily_104.sh Step 4). Three changes:
(1) `live/broker_readonly.py` — ReadOnlyBrokerWrapper's broker tag is now
parameterized: explicit ctor arg > `RENQUANT_READONLY_TAG` env >
default `"alpaca_shadow"` (byte-identical legacy). Tags are validated fail
closed (`[A-Za-z0-9_]+` AND startswith `"alpaca_shadow"`); an invalid tag
raises ValueError instead of silently writing into the legacy lane's state.
State/DB routing derives from the tag via the existing kernel/state_paths
convention: tag `alpaca_shadow_blend` → `live_state.alpaca_shadow_blend.json`
+ `data/runs.alpaca_shadow_blend.db` (added to ALLOWED_BROKERS in
`backtesting/renquant_104/kernel/state_paths.py`).
(2) `live/runner.py` — the readonly title-prefix check generalized from
`== "alpaca_shadow"` to a `_readonly_label_prefix` helper: legacy tag keeps
the byte-identical `[READONLY]` prefix; any other `alpaca_shadow*` tag gets
`[READONLY][<TAG-UPPER>]` (e.g. `[READONLY][ALPACA_SHADOW_BLEND]RENQUANT-104`)
so blend vs legacy stay distinguishable in ntfy titles while
`_notify_decision`'s `is_shadow = label.startswith("[READONLY]")` contract
still classifies both lanes as shadow/hypothetical. The shadow
preflight-strictness branch (`shadow_strict`, default non-strict) likewise
generalized to `startswith("alpaca_shadow")` so the blend lane mirrors the
legacy lane's preflight policy exactly.
(3) `scripts/daily_104.sh` — new Step 5 mirroring Step 4 verbatim
(non-fatal, own `${DATE}_shadow_blend.log`, wrapper ntfy on FAIL/TIMEOUT
with distinct `SHADOW-BLEND-FAIL` / `SHADOW-BLEND-TIMEOUT` titles, same
`RENQUANT_SHADOW_ALERT_NTFY` kill-switch, same buy-side preflight-block
suppression pattern under its own `SHADOW_BLEND_BUY_SIDE_PREFLIGHT_PATTERN`
variable), invoking live-bridge with `--broker readonly-alpaca
--strategy-config-name strategy_config.shadow_blend.json` and
`RENQUANT_READONLY_TAG=alpaca_shadow_blend` on the invocation env. GATE:
Step 5 only runs when `strategy_config.shadow_blend.json` exists in the
pinned strategy configs dir — `if BLEND_STRATEGY_CONFIG="$(
renquant_strategy_config "$SUBREPO_ROOT" strategy_config.shadow_blend.json
)"; then … else` an INFO skip line. This lets the rail land BEFORE the
strategy profile exists without breaking the daily.
WHY/DIR:   Option-A rail per the 2026-07-27 operator directive: the blend
candidate must be shadowed exactly like prod minus order submission
(full funnel incl. sizing/admission/exits against the LIVE account state,
PatchTST-shadow parity, sized picks visible in ntfy) — not an offline
score-only comparison. Tag threading mechanism: env var
`RENQUANT_READONLY_TAG` (NOT a `--broker readonly-alpaca:TAG` CLI syntax)
because the orchestrator live-bridge validates `--broker` against a fixed
ALPACA_BROKERS set and gates Alpaca credentials on the literal broker
string — env threads through the bridge subprocess boundary with ZERO
orchestrator changes, keeping this a single umbrella PR.
EVIDENCE:  artifact: tests/test_broker_readonly_tag.py (17 new tests:
default-identity, env/ctor threading, fail-closed validation incl.
path-traversal probes, three-lane state-path disjointness, ntfy prefix
contract for both tags, end-to-end `_notify_decision` title
`[READONLY][ALPACA_SHADOW_BLEND]RENQUANT-104 [full] SHADOW-DECISION`) +
5 new Step-5 guards in tests/test_daily_104_shadow_notify.py (gate line +
INFO skip, tag env on the invocation line, own log file, alert-by-default
with distinct titles, preflight-block suppression, Step-5-after-Step-4 +
non-fatal). prod or exp: ops rail, no model/performance claim. existing
data: scoped slice (`-k "runner or shadow or state_paths or daily_104 or
readonly or ntfy or broker"`) = 828 passed / 4 failed / 2 collect-errors
on this branch vs 801 passed / SAME 4 failed / SAME 2 collect-errors on
clean origin/main in the same env (test_runner_trade_ntfy live_only
wrapper, test_sell_ntfy_pnl, test_side_config_artifact_paths shadow
param, test_state_store delegate; errors: test_correlation_guard,
test_per_regime_sigma_wire) → zero regressions attributable to this
change; +27 = exactly the new tests. bash: `bash -n scripts/daily_104.sh`
OK; Step-5 gate exercised both ways against a temp SUBREPO_ROOT
(absent → skip branch taken; present → path returned). best-known?: n/a.
scope: run-surface behavior with the env unset and the profile absent is
byte-identical (legacy tag default; Step 5 emits one INFO line and skips).
NEXT:      Sequencing (this PR is rail-only): (a) pipeline-side composite
blend scorer lands separately in renquant-pipeline — NOTE renquant-pipeline
duplicates the ALLOWED_BROKERS allowlist
(`src/renquant_pipeline/state_paths.py`); if any blend-lane component
resolves state paths through the pipeline copy, `"alpaca_shadow_blend"`
must be added there in that PR (duplicated-kernel class, triple-impl
playbook); (b) the `strategy_config.shadow_blend.json` profile lands in
renquant-strategy-104 `configs/` + pin advance — the moment the pinned
profile exists on this machine, daily Step 5 auto-activates (no further
umbrella change, no launchd change; daily_104.sh is the already-scheduled
surface). Until then the lane is dormant and the daily is unchanged.
