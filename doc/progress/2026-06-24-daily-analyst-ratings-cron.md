# Daily analyst-ratings cron — incremental, fail-closed FMP rating-revision pull

STATUS:   merge-pending (PR), tooling-on-HOLD. base-data #24 is MERGED; this stays INERT until the
          base-data PIN is bumped (a separate follow-up PR) — until then the wrapper's import guard
          reports "module unavailable" and exits non-zero (no silent pass, no data written). Do NOT
          install/load the plist from this PR.
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
          exits non-zero via `--fail-on-error` — ANY quota/fetch error fails the run, no per-429
          tolerance — plus `--min-coverage-pct 75` as a secondary floor on the COVERABLE set (a
          systemic break → 0%). The ntfy body now reports `active=%` (with_data/requested) and
          `premium_locked=%` alongside coverable `cov=%`, so a high coverable cov is never misread
          as full active-watchlist coverage (Codex #402). Subset-only infra (free tier ~30% active);
          no production/model decision rides on it.
EVIDENCE: `bash -n` + shellcheck clean (bar the two SC warnings the sibling weekly script also
          carries); `plutil -lint` OK; env resolution verified live (SUBREPO_ROOT, STRATEGY_CONFIG
          → 145-name watchlist parse, PYTHONPATH); inert-guard verified — import currently
          unavailable (pin not bumped) so the wrapper would alert INERT and exit non-zero, exactly
          as designed. The robustness it relies on (FetchResult status buckets + coverage/error
          gates + `source` provenance) lands in base-data #24 (14 unit tests).
NEXT:     This PR is tooling-on-HOLD — do NOT install/load the plist from it. A separate follow-up
          PR bumps the (now-merged) base-data #24 pin and ONLY THEN installs the agent. On the free
          tier this cron is subset-only (~30% active, megacap-biased) and feeds NO production/model
          decision; a panel-wide #184 analyst-revision ablation requires a PAID full-coverage plan
          first — it cannot be run off this free subset.
