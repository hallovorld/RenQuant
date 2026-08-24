# Three jobs alarmed; two root causes, and only one of them is a bug

STATUS:   delivered for the bug. The other two causes are diagnosed, not fixed
          here — one is a decision nobody has made, one is the gate working.

WHAT:     The run-health scan flagged `weekly-wf-promote` (49 non-acting),
          `conditional-retrain104` (27) and `retrain-panel104` (12). Reading
          the logs rather than the counts, there are exactly two causes:

          (1) orch#799 — A DECISION NOBODY HAS MADE. Verbatim from
              logs/retrain_panel/2026-08-16.log, and identically in
              conditional_retrain_104 on its failing days:

                ERROR: no PINNED strategy config declares kind=xgb — cannot
                resolve a kind-matched GBDT production reference (orch#799).
                ... The gate refuses rather than simulate a phantom config.
                Decision needed (orch#799 item 'blend-prod reference rule')

              The served primary is a BLEND; the candidate is XGB; the gate
              needs a same-kind production reference and correctly refuses to
              invent one. That is not a malfunction — it is a design question
              left open, and it blocks TWO of the three jobs.

          (2) THE GATE IS WORKING. weekly_wf_promote 2026-08-20:
              VERDICT: FAIL with genuine_ic=+0.0002 against a >+0.020 bar and
              shuf_ic=+0.0055 against |·|<0.005; then RFC#210 REFUSE because
              production is 18d old (<= 28d SLA). A candidate with no
              demonstrated skill was declined, and the fallback declined to
              force it onto a fresh book. Both refusals are correct.

          (3) THE ONE ACTUAL BUG, fixed here. `conditional_retrain_104.sh`
              branched on the child's EXIT CODE alone, and a refusal exits 0
              deliberately ("governance nominal, calm notify, exit 0",
              weekly_wf_promote.sh:517). So the wrapper printed "chain
              complete" and paged "RenQuant 104 WF promote OK" on 08-19 and
              08-20 while nothing was promoted. It pages a POSITIVE result for
              a non-event, so the operator has no reason to look.

          FIX: the outcome is established POSITIVELY from the child's own two
          terminal markers (`PASSED`, `FALLBACK-PROMOTED`). A clean exit with a
          recorded refusal reports "RAN, NOTHING PROMOTED"; anything else
          reports "OUTCOME UNVERIFIED". The polarity is the point — an outcome
          the wrapper cannot establish must never read as success.

WHY/DIR:  Fourth instance this week of the same class (RenQuant#598/#599/#600):
          a message naming an outcome that is not the outcome. This one is the
          worst of them, because the false report is POSITIVE and arrives by
          push notification.

SECOND WRAPPER, MEASURED LIVE ON 2026-08-23. `retrain_panel.sh` carried the
          identical defect and fired while this PR was in review: it logged
          "delegated weekly_wf_promote PASS" for a chain whose own verdict was
          `VERDICT: FAIL` — genuine_ic=+0.0000, with aligned_real_ic and
          placebo_ic equal to four decimals (+0.0434) — and which promoted
          nothing. That log line is also what the run-health scan reads to
          decide whether the job "acted", so the false PASS corrupted the scan
          as well as the reader.

          Both wrappers now share ONE classifier
          (`scripts/lib/wf_promote_outcome.sh`). Two copies of this rule would
          drift, and that is precisely how the second wrapper kept the bug for
          two days after the first one's fix was written.

TESTS REWRITTEN FROM SCRATCH [codex on #603]. The first suite read
          `logs/weekly_wf_promote/*.log` and re-applied the wrapper's regex in
          Python. In a clean checkout that was **8 passed, 3 skipped** — the
          three incident cases, which are the entire point, silently did not
          run, because those logs are workstation state. And re-applying the
          regex verifies neither the shell branch, nor the notification title
          and body, nor the exit code, nor the seams it added. It measured a
          proxy and called it coverage.

          Now: a hermetic fake repo (stub `python`, stub `subrepo_env.sh`, stub
          child), the REAL wrappers and the REAL classifier, driven end to end,
          asserting on what the operator would actually receive. 11 tests, four
          outcomes each where applicable — promoted (both marker forms), calm
          refusal, nonzero child, zero-exit/unclassifiable. **10 of 11 fail
          against main.** Gated by a new required workflow, because the file
          this replaces was named by no workflow at all — the same gap #601
          fixed for the notification contract, repeated four days later in a
          new file.

EVIDENCE:
  artifact:      `scripts/lib/wf_promote_outcome.sh` (new, shared),
                 `scripts/conditional_retrain_104.sh`, `scripts/retrain_panel.sh`,
                 `tests/conftest_harness.py`, `tests/test_wf_promote_outcome_claim.py`,
                 `.github/workflows/job-outcome-contract.yml`.
  prod or exp:   **exp** — scratch copies pushed via the contents API; the live
                 tree was not written.
  existing data: measured, not assumed —
                 - the orch#799 refusal text [VERIFIED — retrain_panel
                   2026-08-16, conditional_retrain 2026-08-17]
                 - the gate numbers and the RFC#210 REFUSE [VERIFIED —
                   weekly_wf_promote 2026-08-20 lines 637-669]
                 - the child emits exactly two positive markers [VERIFIED —
                   grep of weekly_wf_promote.sh: lines 362, 695, 699]
                 - 5 fallback_verdict.json files exist, ALL `REFUSE`; no
                   promotion appears anywhere in the recent record [VERIFIED]
                 - 11 hermetic tests pass; 10 fail against main's wrappers
                   [VERIFIED — mutation, both scripts swapped back]
                 - the shared classifier labels the three real incident logs
                   (08-19, 08-20, 08-23) NOTHING_PROMOTED [VERIFIED — smoke run
                   against the actual files, separately from the hermetic suite]
                 - today's retrain_panel PASSED the orch#799 stage for the
                   first time in 11 weeks, then the gate declined on merit
                   [VERIFIED — logs/retrain_panel/2026-08-23.log]
  best-known?:   yes. Parsing the child's log is less robust than a distinct
                 exit code, but changing the child's exit contract touches the
                 promotion path itself; this is the smaller change and it fails
                 LOUD (UNVERIFIED) if the markers ever move.
  scope:        the wrapper's outcome reporting, its EXIT CONTRACT, and two
                test seams. No change to what is promoted, to any gate, or to
                the child.

  EXIT CONTRACT (revised 2026-08-24, codex review — the fix's last hole):
                An earlier revision classified UNVERIFIED correctly and then
                exited 0, which handed launchd a SUCCESSFUL job for an outcome
                nobody could establish — the same false OK this file exists to
                remove, relocated from the text to the exit status. It was
                worst in `retrain_panel.sh`, which emits no notification at
                all: a renamed terminal marker would have produced one log line
                and a green process.
                  PROMOTED         -> 0
                  NOTHING_PROMOTED -> 0   (a gate declining is the gate working)
                  FAILED           -> 1
                  UNVERIFIED       -> 2   (NOT 0)
                2 rather than 1 so automation can separate "the child failed"
                (repair the child) from "the child's contract drifted and we
                cannot say what it did" (repair the markers or this classifier)
                — different faults, different repairs. No consumer branches on
                a specific nonzero code [VERIFIED — grepped every caller of
                both scripts; launchd records status only].
                Load-bearing [VERIFIED — mutation: reverting both to exit 0
                gives 2 failed / 9 passed; restored gives 11 passed].

  NOT FIXED:    orch#799 (needs a decision) and the gate's own verdict (it is
                correct). Neither is in scope here and neither should be
                "fixed" by loosening anything.

REVIEW:    codex (haorensjtu-dev).
