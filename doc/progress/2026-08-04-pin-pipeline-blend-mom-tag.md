# 2026-08-04 — pin advance 2: pipeline → 5f07a4d2 (blend-mom broker tag) + honest snapshot base

Second single-pin advance today, carrying renquant-pipeline#264:
`ALLOWED_BROKERS` learns `alpaca_shadow_blend_mom` — measured root cause of the
GOAL-8 S1 session-1 crash (state write raised ValueError; details in the
pipeline repo's progress doc).

Also fixes the snapshot's render base honestly: the fakeroot's prod-artifact
MIRRORS were stale copies from before the 11:31 PT RFC#210 pair-swap, which is
the actual root cause of the `strategy_104_snapshot_fresh` doctor RED that
survived RQ#569. The mirrors are now refreshed read-only from the live tree,
so this snapshot carries the promotion footprint (ACTIVE fp `6461b827…`,
bound scorer fp `d7bddf2a…`) AND the new pin row. Supersedes RQ#570 (closed
with a pointer; its diff is a strict subset of this one).

- `subrepos.lock.json`: renquant-pipeline `a3686efb` → `5f07a4d2` (sha read
  back from the merge API output)
- `doc/arch/strategy-104-snapshot.md`: re-rendered against the refreshed base

Deploy after merge (deploy batch 2, grants-logged): live pull + runtime sync,
then a manual Step-5b shadow-blend-mom rerun to produce S1 session 1's shadow
decision record (operator directive: tonight).
