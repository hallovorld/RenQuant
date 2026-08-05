# 2026-08-04 — TestSourceLevel re-anchor: two tests rotted by the Step-4 retirement

Found tonight while validating the fleet-callsigns branch: two
`tests/test_runner_trade_ntfy.py::TestSourceLevel` tests fail identically on
clean main —

1. `test_daily_shadow_wrapper_suppresses_inner_preflight_ntfy` anchored on the
   heading "Step 4: Shadow e2e", RETIRED 2026-08-03; `src.find` returned −1 and
   the test failed for the wrong reason ever since. Re-anchored to the
   maintained pattern: the retirement heading itself + EVERY Step-5-family leg
   (5/5b/5c/5d/5e) must set `RENQUANT_SUPPRESS_PREFLIGHT_NTFY=1` immediately
   before its heredoc — the guard now covers five lanes instead of one dead
   one.
2. `test_live_only_wrapper_does_not_duplicate_runner_success_ntfy` asserted
   body text ("Wrapper success ntfy suppressed") that left when
   `live_only_104.sh` became an exec shim to `intraday_sell_104.sh`.
   Re-anchored to the properties that must HOLD: the shim delegates via exec
   and composes no success ntfy of its own.

Suite: 63 passed (was 61 passed + 2 wrong-reason failures).
