# `sample_end` was a wall: an unpinned training window must follow the calendar

STATUS:   delivered. `resolve_sample_end()` + both call sites + 7 tests.
          **Code only — this PR does NOT change any config**, so behaviour is
          byte-identical until a config drops its literal `sample_end`. That
          follow-up is a separate strategy-104 PR, deliberately sequenced after
          this so the code supports the value before any config supplies it.

WHAT:     `sample_end` in the served strategy config is the literal
          `"2026-06-30"`. Traced (orch#1015): set ONCE at bootstrap on
          2026-05-25 — **36 days in the future at the time** — and never
          touched again; the only two later commits to that line are whitespace
          reflows. It carries no `_reason` note in a config that documents
          every deliberate choice (`_zblend_floor_note`, `_qp_turnover_max_reason`,
          `_sdl_reason`, …). It was headroom, not policy.

          The calendar overran it on 2026-06-30 and the wall stopped moving.
          `DataFetchJob` fetches `start .. cfg["sample_end"]`, so on the
          2026-08-16 tournament run:

            DataFetchJob: fetching 146 tickers 2016-01-01 -> 2026-06-30
              AAPL: 2637 rows        (the on-disk store held 2672)

          **35 fresh trading days per ticker, fetched away, every week.** With
          the tournament's 5-day label lookahead the feature frame ends
          2026-06-23, so `today - frame_end` grows 7 days a week; it crossed the
          acceptance gate's 45-day cap on 2026-08-09 and **all 142 per-ticker
          candidates have been rejected every week since, incumbents kept**.

          Nothing was broken. The gate was right, the data was fine, and a
          hand-set bound had quietly become a wall.

          FIX: `kernel.data.resolve_sample_end(cfg)` — an EXPLICIT date still
          pins the window (reproducible backtests unchanged); `null` / absent /
          empty now means "follow the calendar". Wired into both readers, the
          tournament path (`pp_training.py`) and the panel path
          (`pp_panel_training.py`), from one shared definition.

          WHY IT RETURNS A DATE AND NOT `None`: `fetch_ohlcv` tolerates
          `end=None`, so `null` alone would appear to work — but
          `ParquetStore.has_range` skips its staleness check entirely when
          `end` is falsy (`if end and df.index.max() < ...`). Passing None
          would therefore fix the wall by DISABLING the cache-freshness guard,
          trading one silent failure for another. A concrete date keeps that
          check alive. `TestItNeverReturnsNone` pins this.

WHY/DIR:  The operator's standing complaint is that the book trades the same few
          names, and this is one of the two upstream causes found (the other,
          the 60% vol cap, is under preregistered test at orch#1017). It is also
          a shape worth killing generally: a static date in a config is a
          time-bomb whose fuse is invisible, because nothing fails at the moment
          it is set — it fails silently N weeks later, somewhere else, as a
          gate rejecting good work.

EVIDENCE:
  artifact:      `backtesting/renquant_103/kernel/data.py` (`resolve_sample_end`),
                 `.../kernel/pipeline/pp_training.py`,
                 `.../training_panel/pp_panel_training.py`,
                 `tests/test_resolve_sample_end.py` (7 tests).
  prod or exp:   **exp** — edited in a scratch copy and pushed via the contents
                 API; the live umbrella working tree was not written. No config
                 touched, so no behaviour changes on merge.
  existing data: measured, not assumed —
                 - `sample_end` value history: one bootstrap commit, two
                   whitespace reflows, never a value change [VERIFIED —
                   `git log -G` on strategy-104]
                 - the fetch line and row counts [VERIFIED —
                   logs/weekly_tournament_retrain/2026-08-16.log]
                 - 2672 on-disk vs 2637 fetched = 35, and exactly 35 trading
                   days exist between 2026-06-30 and 2026-08-19 [VERIFIED]
                 - `2026-06-30` minus 5 trading days = `2026-06-23`, the frame
                   end the gate reports, to the day [VERIFIED]
  best-known?:   yes. The alternative — bumping the literal to a newer date —
                 fixes today and rebuilds the same wall N weeks out, which is
                 how this one was born.
  scope:        the resolution helper and its two callers. Does NOT touch the
                45-day acceptance cap (that gate is correctly refusing stale
                candidates and should keep doing so), any config, or any job.

NEXT:      strategy-104 PR setting `"sample_end": null` with a `_sample_end_reason`
           recording why it is unpinned. Only after that does the tournament
           start fetching through today; this PR alone changes nothing.

REVIEW:    codex (haorensjtu-dev).
