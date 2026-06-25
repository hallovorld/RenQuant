# Daily Finnhub analyst cron — broad-coverage, accumulates history

STATUS:   PR. INERT until base-data #25 merges + the base-data pin is bumped (the wrapper's
          import guard reports ✗ "module unavailable" until then). Unlike the FMP cron (HOLD,
          ~30% coverage), THIS one activates for real once deployed — it's the broad-coverage
          analyst source.
WHAT:     scripts/daily_analyst_ratings_finnhub_refresh.sh + the launchd plist (daily 04:25 PT).
          Pulls Finnhub /stock/recommendation for the WHOLE watchlist each day (MAX_PULL=0;
          60/min free fits ~145 in ~2.5 min) via renquant_base_data.finnhub_analyst_ratings_refresh
          → append-merge data/analyst_ratings_finnhub.parquet (dedup by (ticker,period)).
WHY-DIR:  Finnhub free = broad stock coverage (vs FMP free ~30% plan-lock), so the analyst path
          that FMP coverage shelved is back on. The free window is ~4 months, so DAILY pulls
          accumulate the multi-month series the 3-month revision feature needs.
FAIL-CLOSED: --fail-on-error (any quota/fetch error → ✗) + --min-coverage-pct 90 over the
          COVERABLE set (excludes the ambiguous no_coverage — an empty response can be ETF/index
          OR uncovered/delisted/outage; the fetcher can't tell). ntfy body carries
          with_data / coverable-cov / active_coverage_pct / no_coverage_pct / error counts, so a
          high coverable cov is never misread as full active coverage.
EVIDENCE: bash -n OK; plutil -lint OK. base-data CLI validated live (exit 0; live probe 136/145
          returned data, 9 empty surfaced as no_coverage). `[VERIFIED — bash -n + plutil + live CLI]`
NEXT:     merge base-data #25, bump the base-data pin, then `cp` the plist + `launchctl load` to
          start the daily accumulation. The 3-month revision then becomes a feature CANDIDATE that
          must clear its OWN pre-registered per-regime WF/placebo validation before any retrain or
          enablement — NOT orch #190 (that is the conviction-gate live-outcome validator, a
          different control). Data is megacap-tilted to the analyst-covered universe; treat
          accordingly.
