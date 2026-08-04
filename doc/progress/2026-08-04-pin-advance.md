# Pin advance 2026-08-04 — the night's reviewed merges reach the deployed surface

**Date:** 2026-08-04 · `RenQuant` (umbrella)

STATUS:    lock-file advance (reviewed surface); the runtime-clone sync is
           the granted machine step after merge and is logged in the session
           grants trail.
WHAT:      Six pins advance to their repos' current mains:
           strategy-104 001ab612→320ed77c (blend delta-6 #80, momentum
           profile #81, full-signal v2 #82, $5 wash-sale floor #83);
           model e1f83f8c→dec37193 (v1_fast frozen params #200);
           pipeline 40ec66df→936869f8 (momentum primary surface #259);
           backtesting 8f6700ab→ea7b014a (Stage-2 lane #100 signed-off,
           RFC#210 fallback #102 — THE ARMING DEPENDENCY of RenQuant#559);
           execution c4163984→5724dc74 (AC6 R2 template, coverage API);
           orchestrator ade07dd7→28931844 (sentries #768, detector fixes
           #770/#773, ack refresh #772).
DELIBERATE HOLDS:
           renquant-artifacts stays c09d66f8 — the F-7 canonical-snapshot
           constraint ("do NOT advance past artifacts#29 until snapshot
           integration + green suites") is untested against #31/#32 tonight;
           held rather than assumed. common/base-data unmoved upstream.
ALSO:      This CLOSES tonight's pin drift: the s104 runtime clone was
           synced ahead of the lock (to 320ed77) under the operator's repair
           grants; the drift scan alarming on it was the designed reminder —
           this PR legitimizes rather than silences it.

EVIDENCE:

```
per-pin one-line deltas reviewed above; every advanced commit is a
squash-merge of a codex-approved PR from tonight's session.  [本次实测]
scope:  "subrepos.lock.json + this doc; machine sync is the separate
         granted step."
```

## Revert

git revert + re-sync runtime clones to the prior pins (all recorded here).
