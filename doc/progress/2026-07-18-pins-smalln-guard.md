# Progress: pin bump — small-n guard (pipeline #205 + strategy-104 #60)

Date: 2026-07-18

## What

`subrepos.lock.json`: renquant-pipeline 7108f514 → a6e15db6 (adds the
VetoWeakBuys small-n relax-only guard, RFC pipeline#204, 51 tests) and
renquant-strategy-104 0d45d960 → d3730080 (activation keys
buy_floor_min_n=12 / buy_floor_absolute_smalln=0.50, mirrored in
golden). Evidence orchestrator#543/#544; sentinel rule orchestrator#545
already deployed on the run surface.

## Effect when landed on the machine

Small-n scans (n<12 finite-scored) get the relax-only floor
max(0.20, min(status-quo floor, 0.50)) — one-sided, fixes the
07-16/07-17 all-veto freeze (5/5 vetoed, floor > max score). Normal-n
sessions bit-identical (replay-anchored to the recorded 07-10 n=85
floor). Expected side effect, flagged in the #205 review: the 105
intraday shadow tick's entry-intent counts DROP (its twin task gains
real adaptive floors, previously 0.0) — parity fix, not a regression.

## Landing protocol (separate from this PR)

Machine sync to these pins + two-arm harness epoch-5 refreeze + one
shadow-verify session = ask-first operator batch per the landing-actions
rule; G1 pilot registration (task #77) queues AFTER the epoch-5
refreeze. Merging this PR alone changes nothing on the machine
(merged-is-not-deployed).
