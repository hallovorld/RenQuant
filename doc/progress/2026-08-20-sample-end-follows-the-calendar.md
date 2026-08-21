# `sample_end` was a wall — and the first fix patched a copy nothing runs

STATUS:   delivered. `resolve_sample_end()` + both readers + the config line
          that actually unfreezes the tournament + 9 tests.

WHAT:     `sample_end` in the tournament's config was the literal
          `"2026-06-30"`: set ONCE at bootstrap on 2026-05-25 — **36 days in
          the future at the time** — and never changed since; the only later
          commits to that line are whitespace reflows, and it carries no
          `_reason` note in a config that documents every deliberate choice.
          `kernel/tournament_acceptance.py` admits it in passing: "sample_end
          is bumped manually". It was headroom, not policy.

          The calendar overran it on 2026-06-30 and the wall stopped moving:

            DataFetchJob: fetching 146 tickers 2016-01-01 -> 2026-06-30
              AAPL: 2637 rows        (the on-disk store held 2672)

          35 fresh trading days per ticker, fetched away, every week. With the
          5-day label lookahead the feature frame ends 2026-06-23, so
          `today - frame_end` grows 7 days a week; it crossed the acceptance
          gate's 45-day cap on 2026-08-09 and **all 142 per-ticker candidates
          have been rejected every week since, incumbents kept**.

          Nothing was broken. The gate was right, the data was fine, a
          hand-set bound had quietly become a wall.

WRONG COPY, FIRST TIME. The earlier attempt at this patched
          `backtesting/renquant_103/`. **Nothing imports it.** Resolving
          `kernel.data` and `kernel.pipeline.pp_training` under
          `weekly_tournament_retrain.sh`'s OWN PYTHONPATH lands in
          `backtesting/renquant_104/` — measured, not inferred from the repo
          layout. Those three 103 files are restored to `main` in this branch.
          This is the third time a fix has gone into a non-running twin
          (orch#1018); the only reliable answer is to resolve the module the
          runner's way and read `__file__` back.

A RETRACTED JUSTIFICATION. That earlier draft argued the helper must return a
          date rather than `None` because `ParquetStore.has_range` skips its
          staleness check when `end` is falsy — so `None` would fix the wall by
          disabling a freshness guard. **False in this copy.** The 2026-05-03
          P0 fix removed the `end=None` short-circuit; `has_range` now derives
          `ref = _market_timestamp(end)` and enforces NYSE-aware staleness
          against the wall clock when `end` is None. The claim was true of
          `renquant_103/`. The real reasons are smaller and true:
          `pp_training.py` reads `cfg["sample_end"]` as a hard subscript, so an
          absent key is a KeyError; and a concrete date keeps the fetch window
          legible in the run log instead of printing `None`.

WHY THE CONFIG SHIPS HERE TOO: the code change alone is byte-identical
          behaviour until a config drops the literal. Splitting it would be
          exactly the inert-scaffolding shape — plumbing merged, nothing
          unfrozen, and a follow-up that may never land.

WHY/DIR:  The operator's standing complaint is that the book trades the same
          few names. This is one of the two upstream causes found. It is also
          a shape worth killing generally: a static date in a config is a
          time-bomb whose fuse is invisible, because nothing fails when it is
          set — it fails silently N weeks later, somewhere else, as a gate
          rejecting good work.

EVIDENCE:
  artifact:      `backtesting/renquant_104/kernel/data.py` (`resolve_sample_end`),
                 `.../kernel/pipeline/pp_training.py`,
                 `.../training_panel/pp_panel_training.py`,
                 `.../strategy_config.json`, `tests/test_resolve_sample_end.py`.
  prod or exp:   **exp** — scratch copies pushed via the contents API; the live
                 umbrella working tree was not written.
  existing data: measured, not assumed —
                 - which copy runs [VERIFIED — imported `kernel.data` under the
                   tournament's own resolved PYTHONPATH and read `__file__`]
                 - `sample_end` history: one bootstrap commit, two whitespace
                   reflows, never a value change [VERIFIED — `git log -G`]
                 - the fetch line and row counts [VERIFIED —
                   logs/weekly_tournament_retrain/2026-08-16.log]
                 - 2672 on-disk vs 2637 fetched = 35, and exactly 35 trading
                   days exist between 2026-06-30 and 2026-08-19 [VERIFIED]
                 - `2026-06-30` minus 5 trading days = `2026-06-23`, the frame
                   end the gate reports, to the day [VERIFIED]
                 - end-to-end after the config change, the resolved fetch end
                   moves from 2026-06-30 to today [VERIFIED — run]
                 - both mutations caught: treating null as pinned turns the
                   absence tests red; unwiring `pp_training` turns the
                   call-site test red [VERIFIED]
  best-known?:   yes. Bumping the literal to a newer date fixes today and
                 rebuilds the same wall N weeks out — which is how this one was
                 born.
  scope:        the resolution helper, its two readers, and the one config
                line. Does NOT touch the 45-day acceptance cap (that gate is
                correctly refusing stale candidates and should keep doing so),
                any job, or any other config.

  NOT CLAIMED:  that the next tournament run will now promote. It removes a
                known blocker; whether a candidate then beats its incumbent is
                a separate question this PR does not answer.

REVIEW:    codex (haorensjtu-dev).
