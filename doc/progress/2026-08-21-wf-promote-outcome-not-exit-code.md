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

EVIDENCE:
  artifact:      `scripts/conditional_retrain_104.sh`,
                 `tests/test_wf_promote_outcome_claim.py` (11 tests).
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
                 - 11 tests pass against REAL recorded child logs; 9 of them
                   fail against the pre-fix script [VERIFIED — mutation]
  best-known?:   yes. Parsing the child's log is less robust than a distinct
                 exit code, but changing the child's exit contract touches the
                 promotion path itself; this is the smaller change and it fails
                 LOUD (UNVERIFIED) if the markers ever move.
  scope:        the wrapper's outcome reporting and two test seams. No change
                to what is promoted, to any gate, or to the child.

  NOT FIXED:    orch#799 (needs a decision) and the gate's own verdict (it is
                correct). Neither is in scope here and neither should be
                "fixed" by loosening anything.

REVIEW:    codex (haorensjtu-dev).
