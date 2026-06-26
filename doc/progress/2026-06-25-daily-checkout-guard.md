# Daily preflight: live-checkout guard before pin-align (fatal + operator override)

2026-06-25. Follow-up to the agent-reset incident (#412) + prod re-stamp (#413).

## Why
daily_104.sh already runs a `system_doctor` heartbeat that ntfy's on RED — but it runs AFTER
`preflight_pin_align`. The 2026-06-25 incident class (a stray git op leaves the live umbrella
checkout on a feature branch whose committed `subrepos.lock.json` pins are internally valid but
WRONG) would be **silently deployed**: pin-align materializes the branch pins, then the heartbeat
sees runtime==lockfile (both wrong) → green. The later hard gates (P-CONFIG-FP) prove consistency
WITH the checkout, not that it is the stable `main` interface.

## Fix (this PR)
A guard BEFORE pin-align: if the umbrella checkout is not on `main`, **abort (exit 1) by default**
so a stray branch's pins can't deploy — WITH an explicit operator escape hatch
`RENQUANT_ALLOW_NONMAIN_CHECKOUT=1` to proceed anyway. So it never permanently halts you (you can
always force a run), while refusing to silently deploy a stray branch's pins. Both paths ntfy.
The post-align doctor heartbeat is unchanged.

## Verified
`bash -n` OK. On `main`: passes silently. Off `main`: aborts + ntfy, unless
`RENQUANT_ALLOW_NONMAIN_CHECKOUT=1` (then ntfy-override + continue).

## Note
Earlier revisions of this PR flip-flopped fatal↔non-fatal: the operator worried a hard abort could
halt all trading; Codex showed a pure heads-up doesn't actually protect (pin-align still deploys
branch pins). The fatal-default-with-override resolves both — safe by default, always escapable.
