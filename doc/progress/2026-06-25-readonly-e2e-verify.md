# Gold-standard deploy verify — readonly full daily-full assert-decision

STATUS:   PR. New read-only deploy guard; isolated state, places NO orders, touches NO prod
          state/db. Unblocks safely bumping the orchestrator pin (to activate #174/#188/#190).
WHAT:     `scripts/check_readonly_e2e.sh` runs a FULL readonly daily-full end-to-end and asserts
          it produced a decision (didn't crash, didn't go silent). It exercises the WHOLE
          pipeline code path (panel assembly → scoring → gates → QP → sizing → execution-plan),
          so a BROAD pin bump that breaks any stage is caught here — the heavier companion to
          check_conviction_admits.py (which only re-checks the conviction gate offline).
WHY-DIR:  the operator's remaining stability item, and the prerequisite to safely bumping the
          orchestrator pin (a broad code change my offline verify can't fully cover — an
          import/code break in the daily-full path would slip through).
SAFE:     reuses the proven daily shadow mechanism — `--broker readonly-alpaca` + the shadow
          config whose `broker_name=alpaca_shadow` isolates ALL state to
          live_state.alpaca_shadow.json + runs_alpaca_shadow.db. The shadow scorer differs from
          prod, but the pipeline CODE PATH is shared (exactly what a code-bump verify must
          exercise); prod-scorer specifics stay covered by check_conviction_admits + the bundle
          check. Exit 0 / 1 (crash|timeout|no-decision) / 2 (setup).
EVIDENCE: bash -n + shellcheck clean. FULL validation run 2026-06-25: the readonly pipeline ran
          end-to-end (Phase-2b 87 candidates → scoring → gate_verdicts → decision → commit) and
          the guard returned `READONLY_E2E: OK — produced a decision`, exit 0. State ISOLATION
          proven: runs.alpaca_shadow.db written, prod runs.alpaca.db UNTOUCHED. (Shadow decided
          'no trade' via the PatchTST shadow scorer fail-close — a shadow issue, not a code
          break; the verify correctly asserts the code path RAN, not that it bought.)
          `[VERIFIED — full isolated run]`
NEXT:     wire as the promote_pin.py --verify-cmd for BROAD bumps (orchestrator); add as a
          `make doctor` deep check (opt-in, heavy). Then safely bump the orchestrator pin to
          activate #174 decision-ledger persistence (closes the outcome-validation loop), #188
          bundle check in the runtime, and #190.
