# Shadow lanes are log-only; the tournament verdict names its rejections; alerts already use the patched header encoder (orch#886)   (PR #622)

STATUS:    delivered. Three operator-facing ntfy defects, one PR. No live-tree
           write, no deploy, no pin change; the shadow switch is default-OFF
           and needs no operator action. Decision needed: none beyond review.
           Revision 2 (Codex CHANGES_REQUESTED 2026-08-30T17:28Z): this doc
           was a narrative memo without the C5 literal fields; rewritten
           into the required shape. No code change in revision 2.

WHAT:      (1) `live/runner.py::_notify_decision` — the six daily104
           shadow/readonly lanes no longer push to ntfy by default. They
           compose the identical alert and write it to their own log;
           `RENQUANT_SHADOW_NTFY=1` restores the push.
             * New `_shadow_ntfy_enabled()`: only the literal
               `RENQUANT_SHADOW_NTFY=1` enables (same convention as
               `RENQUANT_NO_NOTIFY`).
             * Gate placed AFTER title/body/priority/taxonomy composition
               and immediately before the single `_post_ntfy_with_retries`
               call, consulted ONLY when `is_shadow`
               (`label.startswith("[READONLY]")`). Log line:
               `ntfy log-only (shadow lane; set RENQUANT_SHADOW_NTFY=1 to push): <title> | <body>`
               — carries the full title so `scripts/check_readonly_e2e.sh:248`'s
               `SHADOW-DECISION` count still finds it.
             * Live `TRADE` / `PENDING` / `FAILED-EXIT` / `ACTION_REQUIRED`
               composition and sending are byte-identical (pinned by a test
               that runs the live path with the switch unset, `1` and `0`
               and requires one call with the same title/priority/body
               each time).
             * Why NOT in `live/alerts.py`: it is a byte-identical twin of
               `renquant-execution/src/renquant_execution/alerts.py`
               (`renquant-orchestrator/scripts/check_twin_parity.py:124`);
               a change there would trip the parity tripwire or require a
               lockstep execution-repo PR. The caller is the right layer
               anyway — it is the one that knows `is_shadow`.
             * No shadow "daily digest" exists in this repo (`grep digest`
               over `scripts/ live/ ops/` finds none for shadow lanes), so
               nothing to preserve. `daily_104.sh`'s own
               `SHADOW-BLEND-FAIL/TIMEOUT` wrapper alerts (lane died) are
               untouched — those are operational, not intent.
           (2) Tournament verdict — `weekly_tournament_retrain.sh`'s final
           message reads the rejection count from a run-bound RECEIPT
           written by `train_104.py`: 0 → `TOURNAMENT-RETRAIN ✓ … CERTIFIED,
           0 rejections`; N>0 → `TOURNAMENT-RETRAIN ⚠ … CERTIFIED WITH 2
           REJECTIONS (APP, SPY)`; receipt missing/stale → `⚠ … UNKNOWN`
           with the reason, never a silent ✓.
             * `scripts/train_104.py`: `_write_tournament_rejection_receipt(...)`
               writes `$RENQUANT_TOURNAMENT_REJECTIONS_OUT` (bound to
               `$RENQUANT_TOURNAMENT_RUN_ID`) right after `baseline_rejected`
               is known — ALWAYS when the tournament ran (zero rejections
               included), NEVER under `--skip-baseline` or when the env is
               unset (ad-hoc runs). Best-effort: a receipt failure logs
               ERROR and does not fail training; the wrapper then reports
               UNKNOWN (⚠).
             * `scripts/tournament_verdict.py` (new, stdlib-only):
               `write_rejection_receipt` (atomic tmp+replace),
               `load_rejection_receipt` (MISSING / UNREADABLE / MALFORMED /
               STALE-by-run_id), `compose_tournament_verdict` →
               `(title, body)`, CLI prints title on line 1, body on line 2.
               Names capped at 10 with `+N more`.
             * `scripts/weekly_tournament_retrain.sh`: exports the two env
               vars next to `LAUNCH_EPOCH` (before launch); the CERTIFIED
               branch runs the CLI with `--run-id "$RUN_ID"` and notifies
               `"$VERDICT_TITLE" "$VERDICT_BODY"`; a composer failure falls
               back to a ⚠ "rejection count UNKNOWN" message. The hardcoded
               ✓ literal is gone (a test asserts its absence). The ✗ branch
               is unchanged.
             * Why a receipt and not a log grep: the `TOURNAMENT ACCEPTANCE
               WARN` text is a curl side effect in `train_104.py:114-140`
               and a WARNING log line; neither is bound to THIS run, and a
               grep would read a same-day rerun's lines. The receipt reuses
               the run-identity idea the marker already uses for no-change
               attestations.
           (3) orch#886 — route `live/alerts.py`'s Title header through
           `renquant_common.notify.encode_header` — was already landed by
           #585 `[VERIFIED — git log -- live/alerts.py: b14cf9e]`. This PR
           does NOT touch `live/alerts.py` (`diff` vs the execution twin
           reports IDENTICAL `[VERIFIED 2026-08-30]`) and instead pins the
           behaviour with four tests in `tests/test_alerts.py`:
           `live.alerts.encode_header is renquant_common.notify.encode_header`;
           `"🚨 rq104 假想前10 — 2026-07-28"` → Title header
           `=?UTF-8?B?…?=` that survives `.encode("latin-1")` and decodes
           back to the original, body bytes unchanged — on the urllib path
           AND the curl fallback (`-H "Title: =?UTF-8?B?…"`); ASCII titles
           pass through unchanged.
           (4) `.github/workflows/operator-notification-contract.yml`:
           `tests/test_tournament_verdict.py` added to its list (the file
           carries `pytestmark = pytest.mark.notification_contract` — it
           asserts operator-read text — so the bidirectional
           marker/workflow guard requires the listing).

WHY/DIR:   Inventory that motivated this `[VERIFIED — read-only pass over
           the fleet's alert log, 08-23..08-30]`: 64 ntfy messages; 22 were
           `[READONLY][<lane>]RENQUANT-104 [full] SHADOW-ACTION: FAILED-EXIT VLO …`
           from the daily104 shadow lanes, plus 4 `[READONLY][V]
           SHADOW-DECISION` and 3 post-close shadow actions — 29/64 = 45%
           of the operator's pages reported a shadow lane's intent.
           "FAILED-EXIT" was true only because the readonly broker sees the
           LIVE run's own pending SELL. `_notify_decision` marks those
           cycles `actionable` (exits_failed) → `force=True` → dedupe
           bypassed → one page per lane per cycle. Separately, `RenQuant
           104 TOURNAMENT-RETRAIN ✓` ("CERTIFIED") fired 1 s after
           `TOURNAMENT ACCEPTANCE WARN: rejected 2 per-ticker candidate(s)`
           (APP, SPY): both true — certification measures
           coverage/freshness/exit code (`tournament_retrain_marker.py`),
           not acceptance — but the ✓ reads as success. Direction: the
           operator's pager carries only live intent and honest verdicts;
           orch#886's remaining ask is closed by test, not by a second
           implementation.

EVIDENCE:  artifact:      `live/runner.py` (`_notify_decision`,
                          `_shadow_ntfy_enabled`), `scripts/train_104.py`,
                          `scripts/tournament_verdict.py` (new),
                          `scripts/weekly_tournament_retrain.sh`,
                          `.github/workflows/operator-notification-contract.yml`,
                          tests: `tests/test_runner_trade_ntfy.py`,
                          `tests/test_broker_readonly_tag.py`,
                          `tests/test_tournament_verdict.py` (new),
                          `tests/test_alerts.py` (+4)
           prod or exp:   neither — alert routing + operator message
                          composition; no model, signal, artifact,
                          fingerprint, or trading-decision path is touched.
                          `live/alerts.py` is byte-identical to its
                          execution twin before and after `[VERIFIED —
                          diff, 2026-08-30]`.
           existing data: the alert-log inventory above (64 messages,
                          29 shadow-intent, 08-23..08-30, read-only) and
                          `git log -- live/alerts.py` showing #585
                          (`b14cf9e`) already routes the Title header
                          through the shared encoder.
           best-known?:   yes — the gate sits at the single send site that
                          knows `is_shadow`; the verdict is bound to the
                          run by receipt rather than inferred from logs.
           scope:         "this is the umbrella live runner + tournament
                          wrapper, prod code path, alert routing only —
                          no performance claim; vs existing best: the
                          identical live alert payloads, now minus the
                          shadow-lane pages."

TESTS:     `[VERIFIED — /Users/renhao/git/github/RenQuant/.venv/bin/python, 2026-08-30]`
           * `.github/workflows/operator-notification-contract.yml` command:
             `RENQUANT_NO_NOTIFY=1 PYTHONPATH=backtesting/renquant_104 python -m pytest -q -o addopts='' tests/test_runner_trade_ntfy.py tests/test_no_trade_reason_rotation_economic.py tests/test_prefilter_is_not_a_rotation.py tests/test_no_trade_priority.py tests/test_broker_readonly_tag.py tests/test_tournament_verdict.py`
             → 170 passed.
           * `tests/test_alerts.py tests/test_tournament_retrain_marker.py tests/test_train_104_acceptance_wiring.py tests/test_train_104_no_wf_bypass.py tests/test_train_104_hardware_threads.py tests/test_check_readonly_e2e_classification.py tests/test_operator_script_env.py tests/test_wf_promote_outcome_claim.py tests/test_preopen_cancel_gate.py tests/test_tournament_acceptance.py`
             → 147 passed, 2 failed — both failures
             (`test_manual_promote_uses_project_venv`,
             `test_multirepo_shell_wrappers_use_shared_strict_helper`)
             assert on `manual_promote.sh` / `conditional_retrain_104.sh`,
             untouched here, and fail identically on pristine
             `origin/main` (`git stash -u` → same 2 failed) `[VERIFIED]`.
             Pre-existing; not addressed in this PR.
           * Revision 2 re-run at head `37a7676` (same interpreter,
             detached worktree, 2026-08-30): the contract command's six
             files plus `tests/test_alerts.py
             tests/test_train_104_acceptance_wiring.py` → 199 passed
             `[VERIFIED — pytest -q -o addopts='' this session]`.
           * New coverage: readonly variant → no `urlopen`, no curl
             `subprocess.run`, log line carries title+body;
             `RENQUANT_SHADOW_NTFY` in `0`/``/`false`/`yes` → no send;
             `1` → one send with the unchanged shadow payload; live
             variant → exactly one call with identical title/priority/body
             under unset/`1`/`0`; live PENDING still pushes; gate sits
             below composition (source-order test); verdict composition
             for 0 / 1 / 2 / 13 rejections, missing, stale, malformed,
             unreadable; CLI two-line contract; shell wiring order (export
             before launch, CLI in the CERTIFIED branch, ✓ literal absent);
             `train_104.py` helper exercised end-to-end into the composer
             (no-op under `--skip-baseline` and under unset env).
           * Existing shadow-composition tests (4 in
             `test_runner_trade_ntfy.py`, 1 in `test_broker_readonly_tag.py`)
             now opt in with `RENQUANT_SHADOW_NTFY=1` — they assert
             composition, which the override pushes verbatim.
           * `bash -n scripts/weekly_tournament_retrain.sh` OK.

NEXT:      * Live tree and `-run` checkout untouched; this lands in
             daily104 only when the operator syncs the checkout (merged is
             not deployed).
           * The `TOURNAMENT ACCEPTANCE WARN` raw curl in `train_104.py` is
             left as is (still useful as the immediate per-ticker reason
             list); it is now complemented, not replaced, by the verdict.
           * `renquant-orchestrator/scripts/check_twin_parity.py`'s
             self-disabling baseline (noted in orch#886) is an
             orchestrator-side fix, not touched here.
           * orch#886 can be closed on merge with a pointer to #585 + this
             PR's tests.
