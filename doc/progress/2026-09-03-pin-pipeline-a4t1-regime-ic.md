# Pin advance: renquant-pipeline b1905d5b -> faf1416a (P-REGIME-IC honours the RFC#210 A4-T1 license)   (PR #635)

STATUS:    delivered — pin advance + snapshot, no code change in this repo.
WHAT:      `subrepos.lock.json`: renquant-pipeline b1905d5b → faf1416a342b5ba8ca0967756b61ee98b9fbfe12
           (renquant-pipeline#308). `doc/arch/strategy-104-snapshot.md`
           re-rendered against the new lock (pin row + lock fingerprint only;
           strategy-104 pin unchanged).
WHY/DIR:   The 2026-09-03 13:55 PT daily full run aborted at P-REGIME-IC [HARD]
           on the A4-T1 candidate promoted that morning (no round-trips ⇒ no
           eligible regime), and had been aborting since 08-31 on P-WF-GATE.
           pipeline#308 makes both P-REGIME-IC twins honour the same
           time-limited, artifact-bound RFC#210 A4-T1 license (candidate
           20260831T141820Z, until 2026-09-07, orchestrator receipt required)
           as a SOFT pass whose text says the regime evidence is ABSENT. With
           this pin the 2026-09-04 13:55 PT run can place orders again; the
           license closes itself on 2026-09-07. Direction: G-F AC2 / G-C.
EVIDENCE:  artifact:      `subrepos.lock.json`; `logs/daily_104/2026-09-03.log` 377/398-401 (P-REGIME-IC ✗ HARD → PRE-FLIGHT FAILED) [VERIFIED — read 2026-09-03]
           prod or exp:   prod (pins the daily runtime)
           existing data: renquant-pipeline#308 merged 2026-09-03, merge commit faf1416a342b5ba8ca0967756b61ee98b9fbfe12, codex APPROVED after r1 [VERIFIED — gh pr view]; pipeline suites 83 passed on the reviewed head [VERIFIED]
           best-known?:   no — the licensed artifact is a zero-trade candidate; this pin makes the operator's authorization executable and self-expiring, it claims no signal
           scope:         "one pin + the snapshot it changes; the license itself is scoped to one run id and one week"
NEXT:      merge → live tree collision check → `git pull --ff-only` →
           `scripts/subrepo_assemble.py --sync --runtime-root .subrepo_runtime/repos`
           → confirm `.subrepo_runtime/repos/renquant-pipeline` carries
           `A4T1_LICENSED_RUN_IDS` → the 2026-09-04 13:55 PT daily104 log shows
           `✓ P-REGIME-IC [SOFT] LICENSED (RFC#210 A4-T1) …` and orders placed.
           Revert: `git revert` this merge + `subrepo_assemble --sync`.
