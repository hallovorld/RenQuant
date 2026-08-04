# Pin advance 2026-08-04 — the night's reviewed merges reach the deployed surface

**Date:** 2026-08-04 · `RenQuant` (umbrella)

STATUS:    lock-file advance (reviewed surface), revised after round-2 review;
           the runtime-clone sync is the granted machine step after merge and
           is logged in the session grants trail.
WHAT:      Four pins advance to their repos' current mains:
           strategy-104 001ab612→320ed77c (blend delta-6 #80, momentum
           profile #81, full-signal v2 #82, $5 wash-sale floor #83);
           model e1f83f8c→dec37193 (v1_fast frozen params #200);
           pipeline 40ec66df→936869f8 (momentum primary surface #259);
           execution c4163984→5724dc74 (AC6 R2 template, coverage API).
           doc/arch/strategy-104-snapshot.md regenerated to match the
           strategy-104 pin bump (round-1 review CI finding: the committed
           snapshot was generated at 001ab612 while the lock moved to
           320ed77c — `verify-pinned-declaration` failed).
DELIBERATE HOLDS:
           renquant-artifacts stays c09d66f8 — the F-7 canonical-snapshot
           constraint ("do NOT advance past artifacts#29 until snapshot
           integration + green suites") is untested against #31/#32 tonight;
           held rather than assumed. common/base-data unmoved upstream.
           renquant-backtesting stays 8f6700ab (was going to ea7b014a) and
           renquant-orchestrator stays ade07dd7 (was going to 28931844) —
           round-2 review: backtesting's ea7b014a carries the RFC#210
           freshness-fallback provider (#102), the arming dependency of
           RenQuant#559's Step 4b consumer; orchestrator's 28931844 still
           predates the FALLBACK-PROMOTED sentinel action-consumer contract
           (renquant-orchestrator#774, open, review required, not merged).
           Advancing either pin now would let a real fallback promotion
           mutate production while the sentinel classifies it as a silent-
           refusal incident. Both stay pinned until the rollout order
           completes: RenQuant#559 merges → renquant-orchestrator#774
           merges → a follow-up pin-advance PR advances backtesting past
           #102 and orchestrator past #774 together.
ALSO:      This CLOSES tonight's pin drift: the s104 runtime clone was
           synced ahead of the lock (to 320ed77) under the operator's repair
           grants; the drift scan alarming on it was the designed reminder —
           this PR legitimizes rather than silences it.

EVIDENCE:
artifact:      subrepos.lock.json (four pins) + doc/arch/strategy-104-snapshot.md
               (regenerated)
prod or exp:   prod — subrepos.lock.json governs which subrepo commit the
               daily/live/weekly runner boots against.
existing data: per-pin one-line deltas above; every advanced commit is a
               squash-merge of a codex-approved PR from tonight's session.
               `python3 scripts/render_strategy_104_snapshot.py --check` and
               `--verify-pinned-declaration` both pass against the pinned
               strategy-104 runtime checkout at 320ed77c (was failing before
               this regeneration). No IC/Sharpe/APY claim is made — this is
               a lock-file + generated-doc change, not a model result.
best-known?:   n/a — pin advance, not a model/data comparison.
scope:         "this is subrepos.lock.json + doc/arch/strategy-104-snapshot.md,
               prod (lock file / generated snapshot), no performance claim —
               backtesting and orchestrator pins are deliberately held back
               per the round-2 review above, not advanced."

NEXT:      merge RenQuant#559 (RFC#210 fallback consumer wiring), then
           renquant-orchestrator#774 (sentinel action-consumer contract),
           then open a follow-up pin-advance PR that moves backtesting past
           #102 and orchestrator past #774 together and re-verifies the
           snapshot.

## Revert

git revert + re-sync runtime clones to the prior pins (all recorded here).
