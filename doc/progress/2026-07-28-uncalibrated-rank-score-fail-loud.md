# Progress: fail loud when the buy floor meets an uncalibrated rank_score

STATUS:   delivered (umbrella fork mirror of renquant-pipeline#219). Canonical
          fix lands there; this keeps the fork from diverging — the class that
          bit twice today already (blend `kind` unknown; `adaptive_quantile`
          buy_floor unsupported here).

WHAT:     Adds a rank_score UNIT DOMAIN marker (`raw` at scoring,
          `probability` after calibration) and a guard in `VetoWeakBuysTask`:
          if the probability-domain buy floor meets a raw-domain rank_score,
          log the mismatch and `_fail_closed_panel_scoring(ctx,
          "rank_score_domain_uncalibrated")` instead of silently vetoing the
          entire cross-section. Absent domain = previous behaviour.
          2026-07-29 codex review (MED) caught that only the
          `score_with_history` branch stamped the `raw` marker; the general
          branch in `job_panel_scoring.py` (alpha158-rebuild path for
          panel_linear/panel_ltr_xgboost/blend without history, plus the
          plain `scorer.score(X)` fallback for any other kind) wrote raw
          scores into `cand.rank_score` around line 915 with no domain
          stamp, so `VetoWeakBuysTask` (in `admission_tasks.py`) never
          tripped for those scorer kinds. Fixed by stamping
          `RANK_SCORE_DOMAIN_RAW` right after that branch's scoring loop
          too, mirroring the existing early-branch stamp — same fix as
          `renquant-pipeline#219`.

WHY/DIR:  `rank_score` is written twice — raw by the scoring stage, calibrated
          probability by the calibration stage. When calibration does not run,
          the raw value survives into a comparison against a [0,1] floor. That
          is a unit error, not a model verdict, and it presents as "no trade".
          The 2026-05-03 fix closed the same confusion from the consumer side
          (read rank_score, not panel_score — see the comment still at the veto
          site naming the production incident); the uncalibrated producer path
          reopened it from the other end.

EVIDENCE: artifact:      local log (this machine, 2026-07-28 21:54)
          `[VERIFIED — direct log read]`, running a fresh PatchTST as primary
          scorer with calibration disabled (its old calibrator failed the
          strict scorer-match contract): `HFPatchTSTPanelScorer.
          score_with_history: scored 82/82 (mean=-0.1071 std=0.0308)` then
          `VetoWeakBuysTask: dropped 75 candidate(s) below rank_score
          floor=max(min=0.20, mean+1.00*std=-0.079) = 0.200 (n=75)` — 75 of
          75, i.e. the whole cross-section, because an all-negative raw
          scale cannot clear a 0.20 probability floor.
          prod or exp:   experiment (manual PatchTST-as-primary-scorer swap
          on this machine, calibration intentionally disabled); no prod
          artifact touched.
          existing data: this worktree's `veto or admission or calibra or
          panel_scoring` test selection: 10 failed / 719 passed, identical
          with and without this diff (baseline measured by stashing the
          diff and re-running) — the 10 are pre-existing environment
          failures, not introduced here. Canonical side
          (renquant-pipeline#219): 2103 passed / 8 skipped, +3 new guard
          tests. General-branch fix re-verification (2026-07-29):
          `tests/test_panel_scoring_job.py tests/test_veto_weak_buys_p0_fix.py
          tests/test_panel_rank_veto.py tests/test_blend_kind_umbrella.py
          tests/test_panel_scoring_specialist_wiring.py` = 108 passed
          (added one regression assertion to
          `TestApplyScoresTask::test_overwrites_candidate_rank_scores`,
          confirmed it fails with `AttributeError` on the pre-fix code);
          `tests/test_panel_scoring_drift.py` (3 failures) and
          `tests/test_meta_label_veto.py` (1 error) fail identically with
          and without this diff — pre-existing, unrelated to this fix.
          best-known?:   n/a — no model/IC/Sharpe number is claimed; this is
          a unit-domain correctness guard, not a performance result.
          scope:         "this is a code fix (fail-loud domain guard)
          verified by direct log read + regression-suite parity across both
          repos, not a model/IC/Sharpe claim — the §4(b) sanity triad does
          not apply."

NEXT:     Merge alongside renquant-pipeline#219, then re-run the PatchTST e2e:
          with the guard in place an uncalibrated swap fails loudly instead of
          reporting a false "no trade", and the calibrated serving artifact
          (effective cutoff 2026-04-27, calibrator fitted in the same WF build)
          gives the model's real answer.
