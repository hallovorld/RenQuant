# Daily Finnhub analyst cron — broad-coverage, accumulates history

STATUS:   PR. base-data #25 is now MERGED, so the producer contract is final on base-data main
          (--min-active-coverage-pct = the fail-closed control; --min-coverage-pct fully
          diagnostic). This wrapper stays INERT until the remaining post-merge activation step —
          bumping the base-data pin in the orchestrator — lands (the wrapper's import guard reports
          ✗ "module unavailable" until the pin is bumped). Unlike the FMP cron (HOLD, ~30%
          coverage), THIS one activates for real once deployed — it's the broad-coverage analyst
          source.
WHAT:     scripts/daily_analyst_ratings_finnhub_refresh.sh + the launchd plist (daily 04:25 PT).
          Pulls Finnhub /stock/recommendation for the WHOLE watchlist each day (MAX_PULL=0;
          60/min free fits ~145 in ~2.5 min) via renquant_base_data.finnhub_analyst_ratings_refresh
          → append-merge data/analyst_ratings_finnhub.parquet (dedup by (ticker,period)).
WHY-DIR:  Finnhub free = broad stock coverage (vs FMP free ~30% plan-lock), so the analyst path
          that FMP coverage shelved is back on. The free window is ~4 months, so DAILY pulls
          accumulate the multi-month series the 3-month revision feature needs.
FAIL-CLOSED (gate lives in the base-data CLI — tested Python, base-data #25 — NOT in the shell):
          The wrapper is THIN: it invokes the CLI and reports the summary; it does NOT re-implement
          a coverage gate in bash (that earlier shell-embedded JSON gate is removed). The CLI owns:
            * --fail-on-error           → exit non-zero on ANY quota/fetch error;
            * --min-active-coverage-pct → THE fail-closed coverage control: floors
              active_coverage_pct (with_data/requested, the FULL requested set), so the ambiguous
              no_coverage bucket counts AGAINST the floor and a real stock silently dropping into it
              still trips the gate (wrapper passes 88 — provisional, see CALIBRATION below);
            * --min-coverage-pct        → DIAGNOSTIC ONLY (coverage over the coverable set, which
              EXCLUDES the ambiguous no_coverage); passed as 0 — it cannot fail-close on a
              widespread-empty run, so it is NOT the safety gate.
          base-data #25 ships Python tests for the active gate: widespread-empty (gate FAILS while
          the coverable metric reads full), a healthy baseline (PASSES), the threshold boundary
          (exactly-at-floor PASSES, just-below FAILS), default-off (the active gate is inert when the
          floor is unset), one main-path failure (the CLI trips a non-zero exit on a widespread-empty
          run), and a diagnostic test proving --min-coverage-pct can NEVER change the exit status. So
          the gate is no longer a shell-embedded JSON check that is easy to regress. An empty response is
          ambiguous (ETF/index, delisted/unsupported, vendor-empty, or a real stock with no current
          recs) — NOT assumed to be ETFs. ntfy ✓ body carries with_data / active% / coverable-cov /
          no_coverage% / error counts.
CALIBRATION (threshold, pre-registered): the 88% active floor is PROVISIONAL — a coarse
          systemic-collapse guard, NOT a tuned threshold and NOT a statistical estimate. It is
          derived from a SINGLE probe (136/145 with data → active_coverage_pct ≈ 93.8%) with no
          baseline window and no variance estimate. The first ~10 daily observations are
          serially/vendor-correlated and far too few to estimate a percentile from, so calibration
          treats them as an empirical RANGE only — no "independent samples", no 5th-percentile fit.
          Pre-registered calibration plan:
            1. BASELINE WINDOW: over the first ~10 daily cron runs, record from the CLI summary
               (logs/daily_analyst_ratings_finnhub/<date>.log) the EMPIRICAL RANGE of
               active_coverage_pct (min/max observed) AND the per-symbol missingness
               (which tickers land in no_coverage, and how persistently) — not a single point.
            2. PROPOSE A FLOOR: from the observed range only — set it below the observed MIN minus
               explicit headroom (e.g. ~3–5 pts), so normal day-to-day vendor jitter does not trip it
               but a real collapse does. This is a range-based proposal, NOT a percentile estimate.
            3. UPDATE RULE: changing MIN_ACTIVE_COVERAGE_PCT requires an EXPLICIT, human-reviewed
               change recorded in this doc (the observed min/max range, the per-symbol missingness,
               and the chosen headroom) — never an automatic fit. Re-review if the watchlist
               composition changes materially or the observed range shifts. Until that reviewed
               change, 88 is a placeholder collapse guard, not a calibrated bound.
EVIDENCE: bash -n OK; plutil -lint OK. base-data #25 CLI validated live (exit 0; live probe 136/145
          returned data → active ≈93.8%, 9 empty surfaced as no_coverage) and carries the gate
          boundary tests above. `[VERIFIED — bash -n + plutil + base-data #25 gate tests + live CLI]`
NEXT (ACTIVATION ORDER — binding, do these IN ORDER):
            1. base-data #25 is MERGED — bump the base-data pin in the orchestrator (the remaining
               post-merge activation step);
            2. one-shot dry-run of the wrapper proving BOTH paths on today's watchlist WITHOUT
               editing committed source — the PASS path
                 `bash scripts/daily_analyst_ratings_finnhub_refresh.sh`
               (healthy active coverage clears the default 88% floor → ntfy ✓) AND the fail-closed
               path
                 `MIN_ACTIVE_COVERAGE_PCT=99 bash scripts/daily_analyst_ratings_finnhub_refresh.sh`
               (a deliberately HIGH floor is breached by a normal ~94% day → ✗, exit non-zero; the
               floor is read from an env var, so no code edit is needed — a simulated
               low-coverage/quota-error response also works);
            3. ONLY THEN `cp` the plist into ~/Library/LaunchAgents/ + `launchctl load` to start the
               daily accumulation. Do NOT load the plist before the dry-run passes both paths.
          The 3-month revision then becomes a feature CANDIDATE that must clear its OWN
          pre-registered per-regime WF/placebo validation before any retrain or enablement — NOT
          orch #190 (that is the conviction-gate live-outcome validator, a different control). Data
          is megacap-tilted to the analyst-covered universe; treat accordingly.
