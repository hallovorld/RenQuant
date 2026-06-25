# Daily Finnhub analyst cron — full-coverage, accumulates history

STATUS:   PR. INERT until base-data #25 merges + the base-data pin is bumped (the wrapper's
          import guard reports ✗ "module unavailable" until then). Unlike the FMP cron (HOLD,
          30% coverage), THIS one activates for real once deployed — it's the full-coverage
          analyst source.
WHAT:     scripts/daily_analyst_ratings_finnhub_refresh.sh + the launchd plist (daily 04:25 PT).
          Pulls Finnhub /stock/recommendation for the WHOLE watchlist each day (MAX_PULL=0;
          60/min free fits ~145 in ~2.5 min) via renquant_base_data.finnhub_analyst_ratings_refresh
          → append-merge data/analyst_ratings_finnhub.parquet (dedup by (ticker,period)).
WHY-DIR:  Finnhub free = FULL stock coverage (vs FMP free ~30% plan-lock), so the analyst path
          that FMP coverage shelved is back on. The free window is ~4 months, so DAILY pulls
          accumulate the multi-month series the 3-month revision feature needs.
FAIL-CLOSED: --fail-on-error (any quota/fetch error → ✗) + --min-coverage-pct 90 over the
          COVERABLE (non-ETF) set; ETFs (no analysts) are no_coverage, not errors. ntfy body
          carries with_data/coverable-cov/etf-no-cov/error counts.
EVIDENCE: bash -n OK; plutil -lint OK. base-data CLI validated live (6/6, 100%, exit 0; full
          collection 136/145 = ~all stocks). `[VERIFIED — bash -n + plutil + live CLI]`
NEXT:     merge base-data #25, bump the base-data pin, then `cp` the plist + `launchctl load` to
          start the daily accumulation. Then the 3-month revision becomes a live full-coverage
          feature candidate, validated over time via orch #190.
