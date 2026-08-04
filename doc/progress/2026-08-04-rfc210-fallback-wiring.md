# weekly_wf_promote consumes the RFC#210 fallback (operator P0 wiring)

**Date:** 2026-08-04 · `RenQuant` (umbrella) · backtesting#101/#102

STATUS:    script wiring; ARMS ONLY after the backtesting runtime pin
           advances past #102 (until then Step 4b prints UNAVAILABLE and
           behaves exactly as today's REJECT — fail-closed, loud).
WHAT:      Step 4b in scripts/weekly_wf_promote.sh: on gate REJECT, consult
           `renquant_backtesting.wf_gate.freshness_fallback --stamp`.
           REFUSE → today's behavior verbatim (REJECT ntfy, exit 1).
           FALLBACK_PROMOTE → pair-promote (same incoming/replace dance as
           Step 5) licensed by the promotion_basis stamp (and requiring the
           stamped passed=False — the gate-passed license check must not
           run on this path); Steps 6/7 run for both paths; final emitter
           line "weekly_wf_promote FALLBACK-PROMOTED (rfc210)" + its own
           ntfy title (paired orchestrator PR teaches the silent-refusal
           sentinel that this line is an ACTION).
WHY:       Operator P0 (2026-08-03): the placebo-deadlocked gate starves
           prod (42d+ stale, 4 identical Sunday REJECTs). Policy decided on
           backtesting#101 (amended), implemented+merged as #102 with the
           real 08-02 reject dry-running to FALLBACK_PROMOTE.

EVIDENCE:

```
bash -n clean; the REFUSE path is byte-equivalent to today's REJECT branch
plus two echo lines; module-absent path measured (current pin 8f6700ab
predates #102 → UNAVAILABLE message + REFUSE).  [本次实测]
scope:  "scripts/weekly_wf_promote.sh + this doc; nothing else. Arming =
         the separate backtesting pin-advance PR + runtime sync grant."
```

## Revert

git revert; the REJECT branch returns to unconditional exit 1.
