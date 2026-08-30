# Earnings calendar refreshed on schedule with a staleness rail; earnings-surprise incremental daily refresh after prints   (PR #TBD)

STATUS:    delivered (code + jobs authored and tested; the two plists are
           NOT installed by this PR — installation is an operator landing
           batch, executed together with the orchestrator manifest PR so
           the run-surface drift scan sees job + manifest land as one
           reviewed change; see NEXT).

WHAT:      (1) The earnings-calendar producer becomes a SCHEDULED job.
           `scripts/fetch_earnings_calendar.py` now writes to the CONSUMED
           path `artifacts/prod/earnings-calendar.json` (it wrote to
           `artifacts/`, one level off, since the 2026-05-10 sim/prod
           isolation refactor 238359b), takes `--config` so the watchlist
           comes from the PINNED strategy config, merges with the previous
           calendar (a transient vendor failure cannot erase known dates),
           writes atomically, and exits 2 when the result is still stale
           (`--min-horizon-days`). New wrapper
           `scripts/refresh_earnings_calendar.sh` + plist
           `com.renquant.earnings-calendar-refresh` run it Mon–Fri 05:40 PT
           (every session morning, 90d lookahead — covers the required
           14-day window) and Sat 04:40 PT (weekly anchor).
           (2) Staleness RAIL, fail SOFT + loud: single source of truth
           `backtesting/renquant_104/adapters/earnings_freshness.py`
           (`assess_earnings_calendar_freshness`: stale when the calendar's
           last date < today+5d). Wired into
           `adapters/runner_artifacts.load_context_artifacts` (log.error,
           run continues) and `scripts/daily_104.sh` Step 0c (ntfy ⚠, run
           continues) via the new `scripts/earnings_calendar_rail.py` CLI
           (exit 0 fresh / 3 stale / 4 missing).
           (3) Earnings-surprise incremental daily refresh:
           `scripts/daily_earnings_surprise_refresh.sh` + plist
           `com.renquant.daily-earnings-surprise` (Mon–Fri 06:00 PT)
           refresh ONLY tickers with a print in the last 7 days
           (`select_recent_prints` over the now-fresh calendar; falls back
           to the full watchlist with a ⚠ ntfy when the calendar itself is
           stale). The Saturday full refresh in
           `weekly_fundamental_refresh.sh` stays unchanged.
           (4) Bonus defect closed: `scripts/execute_shadow_orders.py`
           `ValidateEarningsTask` read a `{"earnings": {...}}` wrapper key
           no producer ever wrote — a silent no-op; it now accepts the real
           flat `{ticker: [dates]}` schema (and the legacy dict entries).
           (5) Tests + CI: `tests/test_earnings_calendar_rail.py`
           (stdlib-only by design) named explicitly by the new
           `.github/workflows/earnings-freshness-contract.yml`; fail-soft
           rail integration tests appended to
           `tests/test_runner_artifacts.py` (local full suite).

WHY/DIR:   2026-08-30 data audit: the calendar artifact froze at its last
           manual run and silently disabled a live risk control — the
           pre/post-earnings buffer (buy ±3d, sell −2d/+5d) could not fire
           for any Aug/Sep print; HPE was bought 08-27 into an early-Sep
           print. Root cause is twofold [VERIFIED — this session]:
           (a) the producer was NEVER scheduled — no launchd plist, cron,
               or wrapper references `fetch_earnings_calendar` (grep over
               scripts/, ops/, deploy/, ~/Library/LaunchAgents; only the
               script itself and the shadow consumer mention the artifact);
               the prod file's mtime is its last manual run,
               2026-04-24 22:13, last date 2026-07-24;
           (b) since 238359b (2026-05-10) the script's output path
               (`artifacts/earnings-calendar.json`,
               scripts/fetch_earnings_calendar.py old lines 82–85) diverged
               from the consumed path (`artifacts/prod/…`, main.py:269–270,
               adapters/runner_artifacts.py:57) — even a manual re-run
               would not have refreshed the live artifact.
           And the surprise lane: `data/earnings_surprise/*.parquet` was
           refreshed only by weekly-fundamental-refresh (Sat 04:00)
           [VERIFIED — installed plist + logs/weekly_fundamental_refresh/
           2026-08-{15,22,29}.log], so after a print (NVDA/CRWD 08-26)
           PEAD/SUE stayed median-imputed for up to a week while the panel
           saw the price gap. A stale calendar silently disabling the
           buffer is the defect; the rail makes it loud without adding a
           new kill switch (fail SOFT everywhere on the consumer side).
           Vendor note: both fetchers are yfinance (free, no API key) —
           no FMP/Finnhub quota is consumed; FMP Starter budget untouched.

EVIDENCE:  artifact:      backtesting/renquant_104/artifacts/prod/
                          earnings-calendar.json (prod, live-consumed)
           prod or exp:   prod run surface (scheduled jobs + live runner
                          preflight); no model/data-science claim
           existing data: rail run against the REAL artifact this session:
                          "[STALE] last date 2026-07-24 < today+5d
                          (2026-09-04)", exit 3 [VERIFIED — CLI run];
                          artifact mtime Apr 24 22:13 [VERIFIED — ls];
                          producer unscheduled [VERIFIED — grep sweep];
                          path divergence [VERIFIED — git show 238359b]
           best-known?:   n/a — ops/data-freshness fix
           scope:         "this is the earnings-calendar/PEAD freshness
                          lane only; no strategy config, no model, no
                          production data file is written by this PR"

TESTS:     tests/test_earnings_calendar_rail.py — 26 passed under
           RenQuant/.venv python [VERIFIED — pytest run this session];
           rail integration (fail-soft + loud) in
           tests/test_runner_artifacts.py — passes with the
           renquant-pipeline sibling on PYTHONPATH (same requirement as
           the pre-existing tests in that file);
           bash -n on all three touched wrappers; plutil -lint OK on both
           plists; live smoke: rail CLI exit 3 on the real stale artifact,
           select-recent returns the expected names for --today 2026-05-01.
           CI: earnings-freshness-contract.yml names the new test file
           explicitly (a test no workflow names never runs — 2026-08-20
           lesson).

NEXT:      operator landing batch (ask-first, not performed here):
           1. merge this PR + the orchestrator manifest PR (cross-linked);
           2. deploy to the live tree (git pull on
              /Users/renhao/git/github/RenQuant per L6 authorization);
           3. install:
                cp scripts/launchd/com.renquant.earnings-calendar-refresh.plist \
                   scripts/launchd/com.renquant.daily-earnings-surprise.plist \
                   ~/Library/LaunchAgents/
                launchctl load ~/Library/LaunchAgents/com.renquant.earnings-calendar-refresh.plist
                launchctl load ~/Library/LaunchAgents/com.renquant.daily-earnings-surprise.plist
           Until step 3, the daily run-surface drift scan alarms
           "manifested job missing from disk" — that alarm is the DESIGNED
           reminder to complete the batch, not a defect.
           Revert steps (literal): launchctl bootout gui/$(id -u)/com.renquant.earnings-calendar-refresh
           and …/com.renquant.daily-earnings-surprise; rm the two plists
           from ~/Library/LaunchAgents; git revert this PR and the
           orchestrator manifest PR.
           First-green criteria: next session morning the prod calendar's
           last date ≥ today+5d (rail exit 0 in daily_104 Step 0c) and,
           after the next watchlist print, that ticker's
           data/earnings_surprise/{T}.parquet carries the new announcement
           before the same day's session.
