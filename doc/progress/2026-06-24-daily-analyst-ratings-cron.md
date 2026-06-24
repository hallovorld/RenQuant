# Daily analyst-ratings cron — incremental, fail-closed FMP rating-revision pull

STATUS:   merge-pending (PR). INERT until base-data #24 merges + the base-data pin is bumped —
          until then the wrapper's import guard reports "module unavailable" and exits non-zero
          (no silent pass, no data written). Installing the plist before that is a no-op alert.
WHAT:     `scripts/daily_analyst_ratings_refresh.sh` + `scripts/launchd/com.renquant.daily-analyst-ratings.plist`.
          A DAILY cron pulls a SMALL incremental batch (--max-pull 40, 1s throttle) of FMP
          `/stable/grades-historical` rating distributions for the pinned strategy watchlist into
          `data/analyst_ratings_fmp.parquet`, calling `renquant_base_data.fmp_analyst_ratings_refresh`.
WHY-DIR:  the free BASIC tier caps BOTH daily (250) and per-minute calls; a one-shot watchlist
          backfill tripped the per-minute limit (104/142 missed). So: many small daily batches, not
          one weekly burst. `select_to_refresh` rotates the ~145-name watchlist oldest-`fetched_at`
          first, covering it every ~4 days, always under both caps. Ratings update monthly, so a
          few-day rotation keeps the consensus-REVISION signal (the documented post-revision-drift
          alpha) fresh enough.
FAIL-CLOSED: applies the 2026-06-24 silent-degradation lesson (same bug fixed in
          weekly_fundamental_refresh): a quota hit / bad key / schema break / 0-data run ntfy's ✗ and
          exits non-zero via the CLI's `--min-coverage-pct 75` gate (catches a SYSTEMIC break — all
          errors → 0% — while tolerating one transient 429 in a 40-name batch). quota_error/fetch_err
          counts are surfaced in the ✓ ntfy body too, so a creeping problem is visible before it
          becomes systemic.
EVIDENCE: `bash -n` + shellcheck clean (bar the two SC warnings the sibling weekly script also
          carries); `plutil -lint` OK; env resolution verified live (SUBREPO_ROOT, STRATEGY_CONFIG
          → 145-name watchlist parse, PYTHONPATH); inert-guard verified — import currently
          unavailable (pin not bumped) so the wrapper would alert INERT and exit non-zero, exactly
          as designed. The robustness it relies on (FetchResult status buckets + coverage/error
          gates + `source` provenance) lands in base-data #24 (14 unit tests).
NEXT:     1) merge base-data #24, bump the base-data pin. 2) `cp` the plist to ~/Library/LaunchAgents
          and `launchctl load`. 3) first clean full-coverage run feeds the #184 per-regime WF/placebo
          confirmation ablation (alpha158 vs +rev3, ≥5 seeds, decide on the placebo-clean difference
          — NOT absolute IC). No production model enablement before that.
