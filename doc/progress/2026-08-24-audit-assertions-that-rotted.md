# Four assertions rotted four different ways, and nothing was running them

STATUS:   delivered. Four assertions repaired (none deleted), one shared
          structural probe, one workflow. Test-and-CI only; no production code
          touched, nothing deployed.

WHAT:     `tests/test_audit_2026_04_24_fixes.py` (3) and
          `tests/test_round3_audit_fixes_2026_04_25.py` (1) had been failing
          unnoticed. All four now assert structure or behaviour via
          `tests/_source_probe.py`, and a new `audit-regression-suites`
          workflow runs both files.

WHY/DIR:  orch#1022. **None of the four was a regression** — every property is
          intact in production. Each was a stale LOCATOR, and naming the four
          mechanisms separately is the point, because "source-text assertions
          are bad" is too coarse to act on:

          | test | what rotted |
          |---|---|
          | sim partial wash-sale | substring inside a hand-tuned BYTE WINDOW, bumped 4k→8k→10k across three previous repairs; the function grew again |
          | ngboost `hs.rank_score` | class MOVED module, `src.find(...)` returned **-1**, `src[idx:]` became `src[-1:]` — a haystack of ONE CHARACTER |
          | LS-ATOM atomic write | the write was EXTRACTED into `save_live_state_atomic`; the caller stopped containing the literal while behaving correctly |
          | EMA50 missing-SPY block | asserted at the wrong LAYER — the task latches, the job boundary applies |

          The common defect is not the use of source text. It is that a
          **missing target was indistinguishable from a failing property**. The
          probe raises `TargetNotFound` — a distinct type — so a refactor now
          says "the code moved" instead of failing an assertion about nothing.

EVIDENCE:
  artifact:      tests/_source_probe.py (new), the two suites,
                 .github/workflows/audit-regression-suites.yml (new).
  prod or exp:   neither — tests and CI only.
  existing data: the four failures reproduce on main today [VERIFIED
                 2026-08-24], and each property was confirmed intact by
                 reading its CURRENT home before touching the test:
                   sim.py:1697   `if not is_partial:` / `_last_sell_date[...]`
                   ngboost_tasks.py:259  `class ApplyNGBoostTask`
                   state_store.py:146-148  tmp write + `tmp_path.replace(...)`
                   job_gates.py:59  applies the latch OR the registry aggregate
  best-known?:   for these four, yes. A fully behavioural rewrite of the three
                 structural ones would need `SimAdapter` / the panel pipeline,
                 which pull in the sibling subrepos — and this repo's CI is
                 hermetic by design (bare checkout, `pip install pytest`, named
                 files, no subrepo runtime). A test that cannot run in CI is the
                 defect being fixed, so structure-with-loud-absence is the
                 strongest assertion available inside that constraint. The one
                 that CAN be behavioural and hermetic — the atomic write — was
                 made behavioural.
  scope:         four assertions and the workflow that runs them.

VERIFICATION:
  174 passed across both suites (was 4 failed / 169 passed).
  Hermeticity checked under `env -i` — a stripped environment — so the suites
  cannot quietly start depending on a developer's PYTHONPATH: 174 passed.

  Mutation-verified against REAL behavioural regressions, not just moved text:
    remove the `if not is_partial` guard so a PARTIAL sell stamps the
      wash-sale date                                  -> 1 failed
    make the job boundary ignore the degrade-safe latch
      (`_gate_block_pending`)                         -> 1 failed
    restored                                          -> 174 passed
  [VERIFIED 2026-08-24]

  The workflow's own steps were run locally exactly as written, including the
  anti-vacuity step that requires the probe to RAISE on a missing target.

NEXT:     **The measurement that reframes this issue.** `tests/` holds 609 test
          files; workflows name 20. **589 are run by nothing** [VERIFIED
          2026-08-24]. These two files are 2 of those 589, found only because
          RenQuant#601 happened to touch the notification path.

          Closing that is a real decision, not a follow-up commit: a bare
          `pytest tests/` in this CI would be red on arrival, because a large
          subset needs the subrepo runtime this workflow pattern deliberately
          does not assemble. The options — a hermetic-subset job, a job that
          checks out the pinned subrepos, or a staged adoption — differ in cost
          and in blast radius, and none should be picked unilaterally inside a
          PR about four assertions.
