# A4-T1: route --promote-staged through the orchestrator wrapper + pin advance (bt#128, orch#1110)

STATUS:    delivered — umbrella side of RFC#210 A4-T1 (v13 / orchestrator v2).
WHAT:      (1) `scripts/weekly_wf_promote.sh --promote-staged` keeps the ONE
           standing mechanism (`freshness_fallback --prod … --staging … --stamp`)
           and, ONLY when that CLI refuses with
           `stamp_refused = a4t1_candidate_requires_orchestrator_consumption`,
           routes to `.subrepo_runtime/repos/renquant-orchestrator/ops/renquant104/
           a4t1_promote_staged.sh` (identify → committed authorization record →
           atomic O_EXCL consume → stamp, exit 0 iff PROMOTED). Every other
           REFUSE stays a REFUSE, so ordinary staged candidates are untouched.
           `renquant-orchestrator` added to the pinned PYTHONPATH list. New
           source-shape test pins the ordering (CLI → named refusal → wrapper →
           shared pair-promote) and the pinned-runtime lookup.
           (2) `subrepos.lock.json`: renquant-backtesting d081670c → 31b6d6e4
           (bt#128), renquant-orchestrator 34336b16 → c3f9d7096b625339659956f12e43bb7b9786f282 (orch#1110,
           which also carries orch#1109 = the L2 paper-bandit manifest entry).
           (3) `doc/arch/strategy-104-snapshot.md` re-rendered against the new
           lock (pins table + lock fingerprint only; strategy-104 pin unchanged).
WHY/DIR:   Since bt#128 the direct `freshness_fallback --stamp` CLI exits 1 for
           the A4-T1 candidate exception (it has no ledger), so the previous
           `--promote-staged` call is fail-closed on exactly the artifact the
           operator authorized. The orchestrator wrapper is the ONLY path that
           consumes the exception (committed authorization record, O_EXCL ledger
           marker under `logs/weekly_wf_promote/a4t1_ledger/`, proof validated by
           backtesting before the stamp). Direction: G-C (model refresh path
           reaches an honest verdict) — the day-31+ lapse of the served
           2026-08-02 model ends with `--promote-staged 20260831T141820Z`.
EVIDENCE:  artifact:      `scripts/weekly_wf_promote.sh` lines 228 (PYTHONPATH) and 357-362 at d7007e76 (the replaced `--stamp` call); `subrepos.lock.json`
           prod or exp:   prod (weekly promote entry point; pins the daily/promote runtime)
           existing data: bt#128 merged 2026-09-03T15:22:30Z, merge commit 31b6d6e41ed7adae28265316ebc8f8723daf9ddf [VERIFIED — gh pr view]; orch#1110 merge commit c3f9d7096b625339659956f12e43bb7b9786f282 [VERIFIED — gh pr view]; bt tests 100 passed, orch governance tests 22 passed, both re-validated by codex on the reviewed heads [VERIFIED — review bodies 2026-09-03T15:20Z]
           best-known?:   n/a — wiring + pin advance, no model claim; the candidate itself is a zero-trade artifact the standing A4 policy refuses, promoted under a time-limited operator authorization (orchestrator record `ops/governance/a4t1/20260831T141820Z.authorization.json`)
           scope:         "this PR changes the promote entry point and two pins; it does not run the promotion — that is the operator step below"
CORRECTIONS (review r1, codex HIGH): the wrapper was looked up at
           `$SUBREPO_ROOT/repos/renquant-orchestrator/...`, but
           `renquant_subrepo_root()` already returns the directory that CONTAINS
           the checkouts (`.subrepo_runtime/repos`, `.subrepo_assembly/current.env`),
           so the `-x` guard would have refused before the wrapper ran. Fixed to
           `$SUBREPO_ROOT/renquant-orchestrator/...`; the new source-shape test
           now forbids `$SUBREPO_ROOT/repos/` in the mode. The first cut of this
           PR also REPLACED the standing `--stamp` call instead of falling through
           from its named refusal, which broke
           `test_promote_staged_mode_reuses_the_one_mechanism` and would have
           routed ORDINARY staged candidates into a wrapper that refuses them —
           rewired as described in WHAT (1).
NEXT:      live tree: read-only checks → `git pull --ff-only` → `scripts/subrepo_assemble.py --sync --runtime-root .subrepo_runtime/repos` (weekly_wf_promote.sh does NOT source preflight_pin_align.sh, so the runtime must be materialised explicitly before the promote) → `scripts/weekly_wf_promote.sh --promote-staged 20260831T141820Z` → verify `logs/weekly_wf_promote/20260831T141820Z.a4t1_promote.json` = PROMOTED and the `FALLBACK-PROMOTED` ntfy → the 13:55 PT daily104 run serves the promoted model (post-close, completed bars). Literal revert: `git revert` this merge; delete the ledger marker only by a recorded governance decision.
