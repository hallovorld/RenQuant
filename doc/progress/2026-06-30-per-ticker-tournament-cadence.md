# per-ticker tournament retrain cadence

STATUS: delivered (schedule artifact + wrapper; install is OPERATIONAL)
WHAT: adds the MISSING scheduled cadence for the per-ticker tournament (RL
Q-table + RandomForest + per-ticker XGB, in
`backtesting/renquant_104/models/<TICKER>/`) — the model population that gates
UNIVERSE ADMISSION and is produced by `scripts/train_104.py --skip-panel`.
Ships three artifacts plus a marker contract:
  - `scripts/weekly_tournament_retrain.sh` — locked, pinned-venv + subrepo
    PYTHONPATH wrapper that freezes the expected watchlist, runs
    `train_104.py --skip-panel --force`, logs to
    `logs/weekly_tournament_retrain/<date>.log`, FAILS LOUDLY (ntfy) on any
    non-zero exit OR on an uncertified population, and stamps a "last successful
    tournament retrain" marker
    (`backtesting/renquant_104/models/.last_tournament_retrain.json`) ONLY when
    the retrain is certified.
  - `scripts/tournament_retrain_marker.py` — ARTIFACT-DERIVED completion
    evidence (see Codex fix below); the marker logic, extracted so it is
    unit-testable without executing the wrapper
    (`tests/test_tournament_retrain_marker.py`).
  - `scripts/launchd/com.renquant.weekly-tournament-retrain.plist` — Sunday
    06:00 PT weekly schedule, mirroring the existing renquant plists.

WHY/DIR: the tournament had NO scheduled job. `launchctl list | grep renquant`
showed retrain-panel104 (a compat no-op delegating to weekly_wf_promote =
PANEL only), conditional-retrain104 (VIX/SPY anomaly → weekly_wf_promote, must
not call train_104 directly), retrain-alpha158-linear, monthly-meta-label-
retrain, weekly-wf-promote — NONE runs the per-ticker tournament. It silently
aged to 61d and had to be hand-retrained on 2026-06-30, after starving the
universe of admissions (a recurring driver of the no-buys). Phase-1/Pillar-2
of the merged model-freshness-governance design is "restore a reliable
tournament retrain cadence"; this is that cadence.

DESIGN NOTES:
  - `--force` is deliberate. FullTrainingPipeline has a `training.cadence` gate
    (`_cadence_allows_today`) that silently returns (exit 0, no retrain) when
    the configured cadence weekday does not match today, and its weekday
    convention differs from launchd (Python Mon=0..Sun=6 vs launchd
    Sun=0..Sat=6). `--force` makes the launchd schedule the single source of
    truth for WHEN, so the job can never silently no-op the way the missing
    cadence did — the exact failure this PR closes.
  - `--skip-panel` confines the run to the BaselineTournamentJob; panel /
    calibrator stay owned by `weekly_wf_promote.sh`. With `--skip-panel`,
    `train_104.py` auto-disables the ModelAcceptanceGate (no candidate panel
    artifact), so per-ticker exports are direct production writes — the
    intended, WF-ungated refresh path for the tournament.
  - Schedule = Sunday 06:00 PT (Weekday=0). Avoids CPU contention with the two
    Saturday 04:00 core-saturating jobs (weekly-wf-promote ~90 min +
    weekly-fundamental-refresh) and lands before Sunday's lighter jobs
    (retrain-panel104 10:00, weekly-apy104 12:00, screen-watchlist 12:05).
    Market closed all weekend → full compute headroom, no live-trade overlap.
  - Marker completion is ARTIFACT-DERIVED, not process-derived (Codex review,
    #420). `train_104.py` exiting 0 is necessary but NOT sufficient: a partial /
    no-op run can exit 0, and the original marker counted pre-existing `models/*`
    dirs (97/230 on the live tree were stale orphans from prior watchlists) and
    stamped `trained_date = wall clock` — so a partial retrain could publish a
    globally fresh-looking marker. `tournament_retrain_marker.py` now: freezes
    the expected watchlist BEFORE launch; captures `LAUNCH_EPOCH`; for each
    expected ticker requires its `<T>-policy-metadata.json` was REWRITTEN this
    invocation (mtime >= LAUNCH_EPOCH, else `stale`); records the effective DATA
    CUTOFF (`live_train_end`, else `trained_date`) + a sha256 artifact digest;
    partitions attempted/succeeded/failed(stale∪unparseable)/missing SETS; and
    refuses to stamp unless a pre-registered coverage policy is met (fresh-
    coverage >= MIN_COVERAGE=0.90 AND zero stale AND zero unparseable). The
    strong regression guard is zero-stale — any previously-trained watchlist
    ticker not rewritten this run blocks certification regardless of the floor;
    the floor additionally catches a wiped / mass-missing population. The marker
    binds `trained_date` to the MIN effective data cutoff (artifact-derived, the
    tournament is only as fresh as its stalest ticker), reports min/max cutoff +
    explicit PARTIAL status, and keeps `wall_clock_date` as a clearly-labelled
    non-authoritative field. Stamped ONLY when certified; on any failure it is
    left untouched so its cutoff keeps ageing → the freshness monitor
    (orchestrator PR #213) alerts too, in addition to the immediate loud ntfy.
  - Cadence-completion is kept SEPARATE from model-quality/promotion
    (`scope: cadence_completion_only`). This marker certifies only that the
    scheduled retrain covered the population; OOS/shadow/WF evaluation against
    the pinned incumbent (turnover/cost, regime slices) stays owned by
    `weekly_wf_promote.sh`. Process completion never implies model quality.

EVIDENCE: `plutil -lint` OK on the plist; `bash -n` clean on the wrapper;
`python -m py_compile` clean on the marker; `git diff --check` clean.
`tests/test_tournament_retrain_marker.py` — 12 tests PASS (stale-not-fresh,
one-ticker-not-rewritten → fail, exit-0-partial → no marker, orphan dirs
ignored, missing tolerated only under floor, unparseable blocks, cutoff
fallback, CLI end-to-end). End-to-end smoke against the live-tree models dir:
133/142 fresh-covered, 9 missing (ETFs/newly-added), 0 stale → certified
`partial`, cutoff [2026-04-13..2026-04-23] (the honest data-cutoff freshness,
vs the old wall-clock stamp). The training run itself is not executed here (it
runs on the live host).

NEXT: OPERATIONAL install on the live scheduler (this PR does NOT install or
touch the live scheduler):
  cp scripts/launchd/com.renquant.weekly-tournament-retrain.plist ~/Library/LaunchAgents/
  launchctl load ~/Library/LaunchAgents/com.renquant.weekly-tournament-retrain.plist
Then point the orchestrator freshness monitor (PR #213) at
`backtesting/renquant_104/models/.last_tournament_retrain.json`. The separate
renquant-strategy-104 PR #37 (parallel_ticker_timeout_seconds 600→2400) fixes
the timeout bug that made the tournament retrain fail even when run manually.
