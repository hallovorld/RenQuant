# 2026-08-02 — routine pin advance: health-record config stamp + pending-key cleanup

STATUS: complete (round 2 after the verify-pinned-paths red: gate #557 +
marker #79 landed first, per the review; lock re-pinned + snapshot
regenerated; machine sync post-merge)

WHAT: renquant-pipeline `dff3cbe3` → `40ec66df` (pipeline#257: every
shadow-health record now stamped with `task_config_path` +
`task_config_sha256` — the fix for the shared-sink profile-identity gap,
pipeline#256) and renquant-strategy-104 `ce8ad100` → `001ab612` (s104#78:
the satisfied `_2026_08_02_pending_first_artifact` key retired; s104#79: the
`_2026_08_02_machine_produced_ledger` key declares the true CI-invisible
state, admitted by the RenQuant#557 gate this branch now carries). Both pins CI-green validated; snapshot regenerated via the mirror
assembly — `--verify-pinned-declaration` OK, mirror `--check` exit 0, parity
pre-check vs `40ec66df` zero NEW kernel drift.

WHY/DIR: GOAL-1 follow-through. With the stamp pinned, tomorrow's records
are profile-attributable (the shadow_blend companion runs through the
stamping `live.runner` path — verified on the machine, noted on #257), which
un-blocks the orch#765 sentinel scoping and already disambiguates the sink
for human readers Monday morning.

EVIDENCE:
- artifact: this PR's diff; the three checks above run locally, not asserted
- prod or exp: reviewed surfaces only; machine sync post-merge
- existing data: pins validated by refresh_subrepo_lock's CI-green gate
- best-known?: yes
- scope: two pins + regenerated snapshot; no other lock entry moves

NEXT: post-merge machine sync (umbrella ff-pull + assemble + live `--check`);
orch#765 implementation once orch#764 lands (same sentinel file — sequenced
to avoid a two-PR conflict on one surface).
