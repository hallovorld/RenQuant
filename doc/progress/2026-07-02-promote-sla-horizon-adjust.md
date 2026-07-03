# Promote gate: horizon-adjust the transformer-panel SLA (S12 B2)

STATUS:   PR (fix to the merged #419 validated promote). SHADOW-SCOPED — moves no
          capital; changes ONE freshness criterion inside the fail-closed gate.
WHAT:     S12's panel-refresh diagnosis (renquant-orchestrator
          `doc/research/2026-07-02-s12-panel-refresh-diagnosis.md` §4-B2) found the
          promote's `transformer_panel` source applies a RAW 28d calendar SLA
          (`age = now - max(date)`) to a fwd-label-CLIPPED axis: the corpus build
          dropna's the fwd_60d label, so `max(date)` is structurally ~60 TRADING
          days (~86 calendar days) behind the bar frontier even for a same-day
          rebuild. `age <= 28` is unsatisfiable BY CONSTRUCTION — the gate returned
          RC_NOT_FRESH forever, even after a perfect refresh. This is the #26
          fund-freshness failure PATTERN recurring inside the new gate, and the
          exact label-horizon case the merged orchestrator #213 monitor (+ #223
          amendment A1) already corrects on the monitoring side.
FIX:      `scripts/promote_shadow_patchtst.py` — a source declaring
          `label_clipped: true` (ONLY `transformer_panel` in DEFAULT_SOURCES) is
          judged from its ACHIEVABLE FRONTIER: `max(date) + lookahead_days` trading
          days vs `now`, i.e. the SLA ceiling is WIDENED by the expected label lag
          (mirroring #213's `label_observation_cutoff` semantics; the reported raw
          `age_days` is never adjusted — a new `age_beyond_frontier_days` field
          carries the judged number). The lookahead comes from the CANDIDATE
          artifact's stamped `lookahead_days` (#223 A1), validated as a genuine
          positive int (<= 250 bdays; a self-declared absurd horizon must not grant
          an unbounded allowance); missing/invalid falls back to the documented
          constant 60 (S12 §2 ground truth for the one recipe served today). An
          implied frontier AFTER `now` (labels observed before their forward window
          closed) FAILS CLOSED as a look-ahead signal, extending the existing
          Codex-#419-review-3 future-dated discipline. The reported/receipt tier
          keys the fast axis on the frontier-adjusted age (a raw ~86d age would
          stamp every receipt "breach" even on a perfect refresh). Every other
          source's SLA (rawlabel raw 28d; fundamentals two-axis) is UNTOUCHED.
TESTS:    `tests/test_promote_shadow_patchtst.py` — panel at the achievable
          frontier (max(date)=2026-04-06 vs now=2026-07-01, raw age 86d) PASSES and
          dry-run-promotes end-to-end with tier=healthy; the CURRENT real state
          (max(date)=2026-02-10, 57d beyond the Mon-Fri frontier) FAILS with the
          age-beyond-frontier stated in the verdict; the raw-28d criterion is
          pinned REMOVED (regression test fails against the pre-fix script,
          verified); scope pins (only transformer_panel is label_clipped;
          non-clipped sources keep the raw SLA); stamp validation + impossible-
          frontier fail-closed + boundary cases. 98/98 promote-gate tests pass;
          full-suite failure set vs pristine origin/main is identical (the
          pre-existing local-data/xdist-order flakes only; zero new failures).
NOTE:     Adjacent seam observed, NOT fixed here (out of B2 scope): the receipts
          written by `_write_promote_log` carry no `gate_version`/
          `candidate_sha256`, which the orchestrator #213 monitor requires before
          a receipt can certify `healthy` (it fails closed to escalate on an
          under-populated receipt). Needs its own follow-up.
NEXT:     B1 (point the refresh `builder_fn` at the TRUE corpus recipe) and B3
          (advance the weekly retrain cutoff with the corpus) per S12 §5; then the
          launchd cadence. Without B1 this gate still (correctly) refuses on the
          frozen corpus — after B1+refresh it now passes at the achievable
          frontier instead of being structurally unsatisfiable.
