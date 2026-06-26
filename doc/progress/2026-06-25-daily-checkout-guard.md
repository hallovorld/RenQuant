# Daily preflight: fatal live-checkout guard before pin-align

2026-06-25. Follow-up to the agent-reset incident (#412) + prod re-stamp (#413).

## Why
daily_104.sh already runs a `system_doctor` heartbeat that ntfy's on RED — but it runs AFTER
`preflight_pin_align`. The 2026-06-25 incident class (a stray git op leaves the live umbrella
checkout on a feature branch, whose committed `subrepos.lock.json` pins differ) would be
**silently deployed**: pin-align re-aligns the runtime to the stray branch's pins, then the
heartbeat sees runtime==lockfile (both wrong) → green → no alert → the run trades the wrong model.

## Fix (this PR)
A **fatal guard BEFORE pin-align**: if the umbrella checkout is not on `main`, ntfy ✗ and
**abort** — trading on a stray branch's pins is worse than not trading. Self-contained
(`git rev-parse --abbrev-ref HEAD`), doesn't depend on the system_doctor `check_live_checkout_branch`
guard added in #412 (which still covers the manual-`make doctor` path). The post-align doctor
heartbeat is unchanged (catches runtime drift / backup hygiene).

## Verified
`bash -n` OK. The guard fires (exit 1 + ntfy) when HEAD≠main; passes silently on main.
