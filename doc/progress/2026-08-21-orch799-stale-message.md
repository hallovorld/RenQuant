# The orch#799 alarm was STALE, and the message said so wrongly

STATUS:   delivered. Two corrections to `weekly_wf_promote.sh`. **No gate logic
          changed**, and deliberately so — the change I was asked for turned out
          to rest on a premise that stopped being true three days earlier.

WHAT:     The run-health scan flagged `retrain-panel104` (11 FAIL) and
          `conditional-retrain104` (25 FAIL), both with:

            ERROR: no PINNED strategy config declares kind=xgb ...
            Decision needed (orch#799 item 'blend-prod reference rule'):
            either derive the xgb reference from the blend's component[0]
            semantics, or gate blend prods on a blend-kind candidate.

          I reported that as a decision nobody had made, recommended option B
          (gate blend prods on a blend-kind candidate), and the operator chose
          B. **B was the wrong thing to build**, and reading the code before
          writing it is what caught that:

          - option A is ALREADY IMPLEMENTED — `_derive_xgb_ref_from_blend` is
            called whenever the pinned kind is `blend`;
          - it WORKS — run by hand against the real pinned config:
            `DERIVED OK, kind = panel_ltr_xgboost`;
          - the 2026-08-20 run used a derived reference and reached a verdict.

          TIMELINE, which is the whole finding:
            shell side  `2f85e0d`      2026-08-16
            backtesting `3ede0de` #112 2026-08-17
            conditional_retrain orch#799 errors: last on 2026-08-17; ZERO on
              08-18, 08-19, 08-20, 08-21
            retrain_panel runs SUNDAYS; last run 2026-08-16 — the day the shell
              fix landed, one day BEFORE its dependency. Next run 2026-08-23,
              which will be its first with the fix present.

          So `retrain-panel104`'s alarm reflects the last run before the fix,
          and the job has not had a chance to run since. Implementing B would
          have changed a capital promotion gate on a stale premise.

FIX (both are about why this cost hours, not about the gate):
          1. THE MESSAGE described a pending decision that had been made and
             shipped. Read literally, its recommended next step is to change
             the gate. It now states that the derivation exists, works, and
             that reaching this line means it was TRIED and FAILED.
          2. THE DERIVATION RAN UNDER `2>/dev/null`. On the one path where it
             matters — failure — the reason was destroyed, and the caller
             printed a generic "no kind-matched reference" that says nothing
             about why. Diagnosing 2026-08-16 required re-running the call by
             hand. Its stderr is now printed, indented, above the message.

WHY/DIR:  A stale diagnostic is worse than none: it does not merely fail to
          help, it actively proposes the wrong repair, and the proposal sounds
          authoritative because it is printed by the system itself. Combined
          with a swallowed stderr, the only surviving explanation was the wrong
          one.

EVIDENCE:
  artifact:      `scripts/weekly_wf_promote.sh` (message + stderr capture).
  prod or exp:   **exp** — scratch copy pushed via the contents API.
  existing data: measured, not assumed —
                 - derivation output against the real pinned blend config
                   [VERIFIED — `derived kind = panel_ltr_xgboost`]
                 - both fix commit dates [VERIFIED — `git log -S`]
                 - per-day orch#799 error counts [VERIFIED — grep of
                   logs/conditional_retrain_104/2026-08-*.log]
                 - the 2026-08-20 run used a derived ref [VERIFIED — grep]
                 - HAPPY-path parity with main: both return exactly one path,
                   the file exists, same derived kind; main's failure path
                   prints NOTHING [VERIFIED — both scripts driven by the same
                   probe under bash]
                 - a bug in my OWN first patch, caught before pushing: a
                   redundant `local err` wrote `err=''` to stdout, and stdout
                   IS the return value, so it would have broken the working
                   path [VERIFIED — probe output]
  best-known?:   yes for these two; the gate itself needs no change.
  scope:         one message and one stderr redirect. No gate logic, no
                 promotion decision, no config.

  NOT DONE:      option B. It was authorised, and I am not building it, because
                 the premise it rests on is stale. If a blend-kind candidate
                 requirement is wanted on its own merits that is a separate
                 discussion with its own evidence.

NEXT:      2026-08-23 is `retrain_panel`'s first run with the fix. If it fails
           again, the stderr this PR surfaces will say why.

REVIEW:    codex (haorensjtu-dev).
