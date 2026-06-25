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
          base-data #25 added boundary tests for the active gate (0%, exactly-at-floor PASS,
          below-floor FAIL, missing active metric, and quota/fetch-error interactions), so the gate
          is no longer a shell-embedded JSON check that is easy to regress. An empty response is
          ambiguous (ETF/index, delisted/unsupported, vendor-empty, or a real stock with no current
          recs) — NOT assumed to be ETFs. ntfy ✓ body carries with_data / active% / coverable-cov /
          no_coverage% / error counts.
CALIBRATION (threshold, pre-registered): the 88% active floor is PROVISIONAL — a coarse
          systemic-collapse guard, NOT a tuned threshold. It is derived from a SINGLE probe
          (136/145 with data → active_coverage_pct ≈ 93.8%) with no baseline window, no variance
          estimate, and no update rule yet. Pre-registered calibration plan:
            1. BASELINE WINDOW: over the first ~10 daily cron runs (multiple independent daily
               probes, not one), record active_coverage_pct each day from the CLI summary
               (logs/daily_analyst_ratings_finnhub/<date>.log).
            2. SET THE FLOOR: at the observed lower-tail of that window (e.g. the min or 5th-pct of
               active_coverage_pct) MINUS explicit headroom (e.g. another ~3–5 pts), so normal
               day-to-day vendor jitter does not trip it but a real collapse does.
            3. UPDATE RULE: revise MIN_ACTIVE_COVERAGE_PCT only after the baseline window completes,
               recording the window stats + chosen headroom in this doc; thereafter re-review if the
               watchlist composition changes materially or the observed lower-tail shifts. Until the
               baseline completes, 88 is a placeholder collapse guard, not a calibrated bound.
EVIDENCE: bash -n OK; plutil -lint OK. base-data #25 CLI validated live (exit 0; live probe 136/145
          returned data → active ≈93.8%, 9 empty surfaced as no_coverage) and carries the gate
          boundary tests above. `[VERIFIED — bash -n + plutil + base-data #25 gate tests + live CLI]`
NEXT (ACTIVATION ORDER — binding, do these IN ORDER):
            1. merge base-data #25, then bump the base-data pin in the orchestrator;
            2. one-shot dry-run of the wrapper proving BOTH paths on today's watchlist — the PASS
               path (healthy active coverage → ntfy ✓) AND the fail-closed path (force a low floor
               or a simulated quota/fetch error → ✗, exit non-zero);
            3. ONLY THEN `cp` the plist into ~/Library/LaunchAgents/ + `launchctl load` to start the
               daily accumulation. Do NOT load the plist before the dry-run passes both paths.
          The 3-month revision then becomes a feature CANDIDATE that must clear its OWN
          pre-registered per-regime WF/placebo validation before any retrain or enablement — NOT
          orch #190 (that is the conviction-gate live-outcome validator, a different control). Data
          is megacap-tilted to the analyst-covered universe; treat accordingly.
