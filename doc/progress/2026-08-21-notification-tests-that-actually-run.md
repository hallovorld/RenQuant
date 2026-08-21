# The 16 tests I shipped in #600 never ran, and could not have passed

STATUS:   delivered. Opt-out fixture + an explicit non-vacuity assertion +
          a marker-based contract set the workflow and the guard share.

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

THE DISCOVERY GUARD WAS NOT A CLASSIFIER [codex on #601]. My first version
          treated any source-text mention of `_notify_decision` /
          `_no_trade_reason` as a notification contract, and CI produced the
          false positives immediately: `test_runner_preflight_fail_closed.py`
          only monkeypatches the helper AWAY (its single mention is
          `monkeypatch.setattr`), and `test_audit_2026_04_24_fixes.py` merely
          mimics a branch. A substring scan cannot encode "asserts on what the
          operator reads", and it would drag an unrelated suite in whenever a
          helper name appeared in a comment.

          I had then started absorbing those false positives into an exclusion
          map — which would have grown into a list of files that were never
          contracts at all, i.e. noise wearing the costume of justification.

          MEMBERSHIP IS NOW AN EXPLICIT MARKER: `pytestmark =
          pytest.mark.notification_contract`, registered in `pytest.ini` and
          applied deliberately by the author. The workflow runs exactly the
          marked files, and the guard asserts both directions — a marked file
          the job does not run is uncovered; a file the job runs without the
          marker means the invocation has drifted from the source of truth.
          Both directions mutation-verified.

          THE EXCLUSION MAP IS GONE, and the reason is worth recording: the
          fifth genuine contract file (`test_broker_readonly_tag.py`, which
          does call `_notify_decision` and assert on `call_args`) needed
          nothing but ONE `PYTHONPATH` entry. `kernel.state_paths` imports
          `pathlib`; `kernel/__init__.py` imports `pkgutil`; no heavy deps.
          Measuring what it actually needed beat excusing it with a reason.

          Final membership, decided by what each file DOES:
          - marked + gated: test_runner_trade_ntfy, test_no_trade_reason_
            rotation_economic, test_prefilter_is_not_a_rotation,
            test_no_trade_priority (12 real `_no_trade_reason(` calls),
            test_broker_readonly_tag
          - not marked: test_runner_preflight_fail_closed (patches it away),
            test_audit_2026_04_24_fixes, test_round3_audit_fixes_2026_04_25
          The four long-red source-text assertions in the last two are real and
          stay filed as orch#1022; they are simply not this contract.

          THE MIRROR IS THE OTHER LESSON. I validated three times against
          harnesses that were not the repo: no `pytest.ini` (which created the
          vacuity), then 3 of 613 test files and no `scripts/` (two phantom
          failures), then no `backtesting/renquant_104/` (a phantom import
          error). The harness that answers correctly copies real `tests/`,
          `live/`, `scripts/`, `backtesting/renquant_104/kernel/`,
          `pytest.ini` and `.github/workflows/`, and runs the job's exact
          invocation: **142 passed**.

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
                 - 142 passed under the job's exact invocation against a
                   faithful mirror [VERIFIED]
                 - the marker guard fails in BOTH directions under mutation:
                   drop a marked file from the invocation, or add an unmarked
                   one [VERIFIED]
                 - every guard mutation-checked: removing the fixture, and
                   removing the file from the invocation, each turn the
                   corresponding test red [VERIFIED]
  best-known?:   yes. Deleting the file would also make CI honest and would
                 discard the only coverage of the rendering split.
  scope:        tests and the workflow. No production code, no config, no
                state. `live/runner.py` is untouched.

REVIEW:    codex (haorensjtu-dev).
