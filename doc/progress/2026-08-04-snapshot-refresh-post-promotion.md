# 2026-08-04 — snapshot refresh: the RFC#210 promotion's reviewed footprint

The daily doctor flagged `strategy_104_snapshot_fresh: STALE`. Root cause is
benign and designed: the 11:31 PT RFC#210 promotion pair-swapped the ACTIVE
scorer + calibrator, changing the artifact content hashes the snapshot embeds;
the snapshot committed with the RQ#569 pin advance was rendered against a
fakeroot whose artifact base predated the swap.

This refresh re-renders against the LIVE tree (read-only) in a worktree. The
12-line diff is exactly the promotion's footprint:

- Artifact metadata file fingerprint `04d7a381…` → `6461b827…` (June-trained
  → 2026-08-02-trained ACTIVE)
- Artifact file fingerprint `d2b4d6ab…` → `bce257d1…`
- Bound scorer content fingerprint `6fc9985e…` → `d7bddf2a…` (the calibrator's
  runtime-legacy stamp, verified matching PanelScorer.load on the live pair)
- Source fingerprint accordingly.

No production files were written; the render read the live tree and wrote only
this worktree's copy. Doctor closure: merge + the live tree's next pull.
