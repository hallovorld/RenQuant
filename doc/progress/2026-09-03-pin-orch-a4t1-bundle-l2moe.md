# Pin advance: renquant-orchestrator c3f9d709 -> 0474af18edf9a1190193d0a213de41dad7491178 (A4-T1 bundle checker + L2 MoE mixture book)   (PR #636)

STATUS:    delivered — pin advance + snapshot, no code change in this repo.
WHAT:      `subrepos.lock.json`: renquant-orchestrator c3f9d709 → 0474af18edf9a1190193d0a213de41dad7491178,
           carrying orch#1113 (bundle checker accepts the RFC#210 A4-T1 stamp
           for the missing Sharpe numerics, lockstep with pipeline#308) and
           orch#1114 (L2 MoE mixture book — the Hedge allocation marked daily
           in shadow as a derived view, held-step valuation rule, comparators
           under the same rule). `doc/arch/strategy-104-snapshot.md`
           re-rendered against the new lock (pin row + lock fingerprint only).
WHY/DIR:   `scripts/system_doctor.py` runs the bundle checker from the PINNED
           orchestrator runtime (`.subrepo_runtime/repos/renquant-orchestrator`),
           so the 13:55 PT DOCTOR page's `[RED ] bundle_consistency` on the
           served A4-T1 pair clears only through this pin (the `-run` checkout
           is already at main for the ops jobs). The L2 job itself reads
           `renquant-orchestrator-run/src`, so the mixture book is live from the
           15:45 PT run; this pin keeps the pinned runtime at parity. Direction:
           G-F AC4 (DOCTOR truthful), G-M AC2 (MoE in shadow).
EVIDENCE:  artifact:      `subrepos.lock.json`; ntfy "RenQuant 104 DOCTOR" 2026-09-03 13:55 PT (bundle_consistency RED) [VERIFIED — ntfy cache]
           prod or exp:   prod (pins the daily/doctor runtime)
           existing data: orch#1113 merged c7a71a2f (41 bundle tests), orch#1114 merged 0474af18edf9a1190193d0a213de41dad7491178 (13 L2 tests), both codex APPROVED [VERIFIED — gh pr view]
           best-known?:   n/a — pin advance, no model claim
           scope:         "one pin + the snapshot it changes"
NEXT:      merge → live tree collision check → `git pull --ff-only` →
           `scripts/subrepo_assemble.py --sync --runtime-root .subrepo_runtime/repos`
           → `make doctor` (or the next 13:55 DOCTOR page) shows
           bundle_consistency green for the served pair. Revert: `git revert`
           + `subrepo_assemble --sync`.
