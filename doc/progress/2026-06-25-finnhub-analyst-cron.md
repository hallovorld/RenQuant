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
FAIL-CLOSED: --fail-on-error (any quota/fetch error → ✗) + an ACTIVE-coverage floor (88%) the
          wrapper enforces over the FULL requested set (with_data/requested), NOT the coverable
          set — so a real stock silently dropping into the ambiguous no_coverage bucket still trips
          the gate (the CLI's coverable gate is disabled here with --min-coverage-pct 0). An empty
          response is ambiguous (ETF/index, delisted/unsupported, vendor-empty, or no current recs);
          ntfy body carries with_data / active% / coverable-cov / no_coverage% / error counts.
EVIDENCE: bash -n OK; plutil -lint OK. base-data CLI validated live (exit 0; live probe 136/145
          returned data → active ≈94%, 9 empty surfaced as no_coverage). `[VERIFIED — bash -n + plutil + live CLI]`
NEXT (ACTIVATION ORDER): (1) merge base-data #25 + bump the base-data pin; (2) one-shot dry-run of
          the wrapper proving the active/no_coverage metrics + fail-closed on today's watchlist;
          (3) THEN `cp` the plist + `launchctl load` to start the daily accumulation. Do NOT load
          the plist before the dry-run passes. The 3-month revision then becomes a feature CANDIDATE that
          must clear its OWN pre-registered per-regime WF/placebo validation before any retrain or
          enablement — NOT orch #190 (that is the conviction-gate live-outcome validator, a
          different control). Data is megacap-tilted to the analyst-covered universe; treat
          accordingly.
