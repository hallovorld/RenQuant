# Training universe: add CRWV, RKLB, SPCX so the served watchlist is trainable   (PR #640)

STATUS:    delivered — config-only change to the umbrella's TRAINING
           watchlist; operator-authorized 2026-09-03.
WHAT:      `backtesting/renquant_104/strategy_config.json` `watchlist` gains
           `CRWV`, `RKLB`, `SPCX` in sorted position (142 → 145 entries;
           three added lines, nothing else in the file changes). This is the
           file `weekly_tournament_retrain.sh` freezes the per-ticker
           tournament universe from and the file the daily run-surface
           drift scan's `check_watchlist_trainability` compares against the
           PINNED served config.
WHY/DIR:   Since 2026-08-19/20 the served strategy-104 watchlist (145) has
           carried three names the training universe (142) did not, so
           every daily run logs them `no_artifact, skipping` and the drift
           scan has paged "watchlist-trainability: 3 served ticker(s) are
           absent from the training watchlist … CRWV, RKLB, SPCX" every day
           (it is the one standing drift item; the scan's own prescribed
           fix is exactly this change: "add them to
           backtesting/renquant_104/strategy_config.json"). A served name
           that can never receive a per-ticker artifact is inert capital
           allocation with one WARNING line as its only trace. The
           invariant the scan enforces is `served ⊆ trained`: be in the
           training universe, then be excluded from it WITH a reason if
           that is the intent (`non_trainable` is derived by intersecting
           with this list, so a name absent from it cannot even be declared
           non-trainable). Direction: G-A/G-D (stop the standing page by
           fixing its cause, not its reporter).
           Authorization: operator, first-hand, Claude operator session
           428feb92, 2026-09-03 between 18:01 and 18:10 PDT, verbatim
           「授权，加速」 ("authorized, speed up") — in direct reply to the
           agent message enumerating the pending operator decisions, which
           named "CRWV/RKLB/SPCX". Recorded as a reply to an enumerated list
           that named this change. This file is the umbrella's training
           config, not the served `configs/strategy_config.json` that LONG
           row 2 protects; no served key changes.
EVIDENCE:  artifact:      `renquant-orchestrator/ops/run_surface_drift_check.py::check_watchlist_trainability` (lines 812–875); daily drift pages 08-24 → 09-03 [VERIFIED — ntfy triage 08-30 + today's log]; live training config byte-identical to origin/main [VERIFIED — cmp, 2026-09-03 between 18:24 and 18:28 PDT]
           prod or exp:   prod config (training universe); affects the next weekly tournament retrain (which trains or explicitly declines the three names) and the drift scan's verdict; no served config, artifact, or pin changes
           existing data: the drift check's FUNCTION (imported from the `-run` checkout, no notify path executed) against the pinned served config: BEFORE this change `served=145 trained=142 unaccounted=3` with the standing problem; AFTER (function pointed at this PR's blob) `problems=[]`, `served=145 trained=145 unaccounted=0` [VERIFIED — 2026-09-03 between 18:24 and 18:28 PDT]; the modified file parses, its watchlist stays sorted, and a JSON-level compare shows the watchlist is the only key that differs [VERIFIED — assertion in the edit script]
           best-known?:   n/a — config coverage fix; no claim that any of the three names has signal. CRWV (IPO 2025-03) has a short history: the tournament may decline it on data length — that outcome is the honest one and will be visible in the tournament log instead of a silent skip
           scope:         "this PR adds three tickers to the training watchlist; it does not touch the served config, the sector map, any artifact, or the tournament's exclusion logic"
NEXT:      live tree: read-only checks → `git pull --ff-only` (this file is
           tracked and currently clean live) → next drift scan run reports
           `unaccounted=0` (the standing page stops) → next weekly tournament
           (`weekly_tournament_retrain.sh`) either trains the three names or
           records them in `non_trainable` with a reason; the daily run's
           `no_artifact, skipping` lines for them disappear after that
           tournament. Revert = `git revert` this merge.
