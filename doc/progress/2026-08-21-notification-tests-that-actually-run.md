# The 16 tests I shipped in #600 never ran, and could not have passed

STATUS:   delivered. Opt-out fixture + an explicit non-vacuity assertion + the
          workflow actually running the file + a self-enforcing coverage guard.

WHAT:     #600 merged with "5/5 checks pass" cited as evidence for
          `tests/test_prefilter_is_not_a_rotation.py`. Two independent facts,
          both invisible behind that green check:

          1. **No workflow named the file.** Every CI job in this repo runs a
             specific named test file; the new one was in none of them, so its
             16 tests never executed in CI at all. The green check was true and
             said nothing about them.

          2. **In the real repo they could not have passed.** `pytest.ini` sets
             `RENQUANT_NO_NOTIFY=1` for every test — a deliberate safety after
             the 2026-08-05 incident where a test paged the operator's real
             phone. `_notify_decision` short-circuits on it, so `urlopen` was
             never called and every assertion was vacuous. Run in the deployed
             tree: **16 failed**, all `TypeError: 'NoneType' object is not
             subscriptable` — the mock had no recorded call.

          `tests/test_runner_trade_ntfy.py` has carried the opt-out fixture all
          along. I copied that file's structure without noticing the fixture
          was load-bearing.

          WHY MY HARNESS SAID 97 PASSED: it was a scratch directory with no
          `pytest.ini`. The suppression that defines the real environment was
          simply absent, so the tests exercised a code path that does not
          exist under CI or in the deployed tree. Copying `pytest.ini` into the
          harness reproduces the 16 failures immediately.

          FIX, in three parts, because one would have left a hole:
          - the same autouse fixture, with the reason written down;
          - `_body` now asserts `m.called` FIRST, so a future suppression fails
            with "every assertion downstream of this would be VACUOUS" instead
            of a bare `NoneType` subscript error;
          - the workflow runs all three notification test files, and
            `test_the_workflow_runs_every_notification_test` fails if a file
            that touches `_notify_decision` is not named in the invocation.

          THE GUARD WAS VACUOUS ON ITS FIRST ATTEMPT TOO, and mutation caught
          it: it searched the whole YAML, and the explanatory comment I had
          just written NAMES the file — so the comment satisfied the check.
          Narrowed to parse only the `python3 -m pytest ...` command with its
          line continuations. `test_the_invocation_is_actually_found` pins that
          the parse returns something, since `""` would make every membership
          test pass.

THE GAP WAS 6 FILES, NOT 1. The guard failed on its first CI run naming five
          MORE test files that touch the notification path and that no workflow
          runs. My scratch harness held three test files total, so it could not
          have seen them; the real `tests/` holds 613. Resolution, measured
          against a faithful mirror of the job's minimal environment rather
          than guessed:
          - gated: test_no_trade_priority.py, test_runner_preflight_fail_closed.py
          - excluded, `kernel` not on this job's path (ModuleNotFoundError at
            collection in the mirror; passes 30/30 in the full tree):
            test_broker_readonly_tag.py
          - excluded, pre-existing failures: test_audit_2026_04_24_fixes.py (3)
            and test_round3_audit_fixes_2026_04_25.py (1), all source-TEXT
            assertions that rotted when the code moved into the pipeline
            subrepo — orch#1022. Four tests had been failing with nobody
            watching, which is the same defect one layer over.

          The exclusion map is modelled on the tournament's `non_trainable`
          map, the one mechanism in this system that handles "deliberately not
          covered" well: a thing is either covered or excluded WITH A REASON,
          never absent by accident. Three tests keep it honest — every entry
          needs a non-empty reason, must name a file that still exists, and
          must still touch the notification path.

          THE MIRROR IS THE OTHER LESSON. I validated twice against harnesses
          that were not the repo: first without `pytest.ini` (which is what
          created the vacuity), then with only 3 of 613 test files and no
          `scripts/` (which produced two more phantom failures). The harness
          that finally answered correctly is a copy of the real `tests/`,
          `live/`, `scripts/`, `pytest.ini` and `.github/workflows/`, run with
          the job's exact pytest invocation: **119 passed**.

WHY/DIR:  Production behaviour was never wrong — `live/runner.py` is correct
          and independently verified by replaying the real 2026-08-20 payload
          through the deployed tree. What was wrong is that the safety net
          under it was decoration, while a green check and a merged PR said
          otherwise. That is worse than a missing test, because it is a test
          nobody will re-examine.

EVIDENCE:
  artifact:      `tests/test_prefilter_is_not_a_rotation.py`,
                 `tests/test_runner_trade_ntfy.py`,
                 `.github/workflows/operator-notification-contract.yml`.
  prod or exp:   **exp** — scratch copies, pushed via the contents API. The
                 live umbrella working tree was not written.
  existing data: measured, not assumed —
                 - 16 failed in the deployed tree at b528ca6 [VERIFIED]
                 - `pytest.ini:26` sets `RENQUANT_NO_NOTIFY=1` under `env =`
                   [VERIFIED — read]
                 - no workflow names the file [VERIFIED — grep of every
                   `.github/workflows/*.yml` pytest line]
                 - 119 passed under the job's exact invocation against a
                   faithful mirror (real tests/ + live/ + scripts/ + pytest.ini)
                   [VERIFIED]
                 - every guard mutation-checked: removing the fixture, and
                   removing the file from the invocation, each turn the
                   corresponding test red [VERIFIED]
  best-known?:   yes. Deleting the file would also make CI honest and would
                 discard the only coverage of the rendering split.
  scope:        tests and the workflow. No production code, no config, no
                state. `live/runner.py` is untouched.

REVIEW:    codex (haorensjtu-dev).
