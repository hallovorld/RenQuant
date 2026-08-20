# `None→APH` was not a broken rotation — it was never a rotation

STATUS:   delivered. Renderer split + 13 tests (11 of which fail against main).

WHAT:     The live 2026-08-20 DECISION message opened with

            BLOCKED-ROTATION None→APH (nonpositive_expected_return_no_long)
            BLOCKED-ROTATION None→WELL (...)
            BLOCKED-ROTATION None→CVS (...)
            BLOCKED-ROTATION +58 more (61 total — see run log)

          and the operator asked the two questions it provokes: why is every
          rotation failing, and why is the sell leg NULL.

          NEITHER PREMISE WAS TRUE. No rotation failed, because none was
          attempted for those names. `BuildPairsTask` declines a buy CANDIDATE
          before any sell leg is chosen (#289, 2026-08-17) and the producer
          writes `sell=None` together with `stage="prefilter"` deliberately —
          its own comment reads "no pair exists yet ... so monitors can tell
          the stages apart". All 61 entries that day were prefilter; zero were
          pairs.

          So the producer was correct and self-describing. The defect was
          entirely in this umbrella renderer, which ignored `stage` and
          invented a rotation that never existed — then reported it as broken.

          Two things it got wrong, both worth naming:
          - it never read `stage`, the field added precisely to let a consumer
            tell the stages apart;
          - `rb.get("sell", "?")` could never fire its own default. The key is
            PRESENT with value None, so `.get` returns None and the "?" is
            dead code. A default fires on a missing key, never on a null value.

          FIX: prefilter entries render as one compact segment

            DECLINED-BUY x61 (nonpositive_expected_return_no_long 61) — APH, WELL, CVS +58 more

          and real pairs keep `BLOCKED-ROTATION sell→buy (reason)`. Mixed
          payloads report both, never pooled. Per-reason counts sort by count
          desc then FULL reason string asc — the determinism lesson from #599,
          applied here before review rather than after.

          NOT a cosmetic rename. The message steered the operator toward a
          subsystem that was working, on a day when the actual binding cause
          was a panel score with genuine_ic = -0.032 in BULL_CALM declining
          candidates with positive expected return.

WHY/DIR:  Third in a row of the same class (#598, #599, this): a notification
          naming a cause that is not the cause. The operator reads these on a
          phone and acts on them; two of the three had already moved them
          toward loosening a live risk limit or suspecting rotation. The
          notification surface is a decision input, so a wrong label there is a
          correctness bug, not a display nit.

EVIDENCE:
  artifact:      `live/runner.py` (the blocked-rotation segment),
                 `tests/test_prefilter_is_not_a_rotation.py` (13 tests),
                 `tests/test_runner_trade_ntfy.py` (two fixtures corrected).
  prod or exp:   **exp** — edited in a scratch copy, pushed via the contents
                 API. The live umbrella working tree was not written.
  existing data: measured, not assumed —
                 - the exact live message [VERIFIED — RenQuant
                   logs/daily_104/2026-08-20.log:571]
                 - `"sell": None, "stage": "prefilter"` written deliberately,
                   with the explanatory comment [VERIFIED — the RUNNING tree at
                   `RenQuant/.subrepo_runtime/repos/renquant-pipeline/.../
                   task_rotation.py`, not the dev checkout, which does not
                   contain this code path at all]
                 - candidate rank order: `eligible_candidates = [c for c in
                   ctx.ranked if ...]` [VERIFIED — task_rotation.py:250], and
                   the log's panel scores descend to the decimal (APH 2.434,
                   WELL 2.210, CVS 2.209, ROST 1.934)
                 - 11 of 13 new tests fail against main; the 2 that pass are
                   the preservation tests, which is the correct polarity
                   [VERIFIED — mutation run]
  best-known?:   yes. The alternative — printing the sell leg as "?" — keeps
                 the false claim that a rotation was attempted, which is the
                 actual defect.
  scope:        the ntfy body segment only. No order path, no gate, no sizing,
                no config, no state, and no change to what the pipeline
                produces.

FIXTURE BUG FOUND IN MY OWN EARLIER TEST: `test_blocked_rotations_are_capped_
with_an_honest_remainder` used `"sell": None` while asserting the paired `S→B`
rendering. That payload shape never produces a paired rotation — the assertion
passed only because the renderer was printing `None→B`, i.e. the test was
pinned to the very defect being fixed. Corrected to real sell legs, along with
its sibling flood test.

NEXT:      deploy to the live umbrella tree after merge (merged is not
           deployed — the daily run reads the working tree, not origin/main).

REVIEW:    codex (haorensjtu-dev).
