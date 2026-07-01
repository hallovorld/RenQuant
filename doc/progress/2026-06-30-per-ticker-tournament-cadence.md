# per-ticker tournament retrain cadence

STATUS: delivered (schedule artifact + wrapper; install is OPERATIONAL)
WHAT: adds the MISSING scheduled cadence for the per-ticker tournament (RL
Q-table + RandomForest + per-ticker XGB, in
`backtesting/renquant_104/models/<TICKER>/`) — the model population that gates
UNIVERSE ADMISSION and is produced by `scripts/train_104.py --skip-panel`.
Ships two artifacts plus a marker contract:
  - `scripts/weekly_tournament_retrain.sh` — locked, pinned-venv + subrepo
    PYTHONPATH wrapper that runs `train_104.py --skip-panel --force`, logs to
    `logs/weekly_tournament_retrain/<date>.log`, FAILS LOUDLY (ntfy) on any
    non-zero exit, and stamps a "last successful tournament retrain" marker
    (`backtesting/renquant_104/models/.last_tournament_retrain.json`) on success.
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
  - Marker is stamped ONLY on exit 0. On failure it is left untouched so its
    `trained_date` keeps ageing → the freshness monitor (orchestrator PR #213)
    alerts too, in addition to the immediate loud ntfy. The marker exposes the
    same `trained_date` field the panel artifact uses, so the monitor can
    compute tournament age without scanning 100+ per-ticker dirs.

EVIDENCE: `plutil -lint` OK on the plist; `bash -n` clean on the wrapper;
`git diff --check` clean. Not executed here (retrain runs on the live host).

NEXT: OPERATIONAL install on the live scheduler (this PR does NOT install or
touch the live scheduler):
  cp scripts/launchd/com.renquant.weekly-tournament-retrain.plist ~/Library/LaunchAgents/
  launchctl load ~/Library/LaunchAgents/com.renquant.weekly-tournament-retrain.plist
Then point the orchestrator freshness monitor (PR #213) at
`backtesting/renquant_104/models/.last_tournament_retrain.json`. The separate
renquant-strategy-104 PR #37 (parallel_ticker_timeout_seconds 600→2400) fixes
the timeout bug that made the tournament retrain fail even when run manually.
