# 2026-08-11 — Restore the `live/alerts.py` twin: `encode_header` on the stack that trades

STATUS:   PROPOSED (2026-08-11). Restores byte-identity for the
          `live/alerts.py` <-> `src/renquant_execution/alerts.py` pair, which
          `tests/test_twin_parity.py::test_live_twin_parity_manifest_current`
          currently reports as **FAIL / DIVERGED**
          `[VERIFIED — make test on the deploy machine: 1 failed, 6259 passed,
          2 skipped; the sole failure is byte_identical:alerts.py]`.

WHAT:     Copies `renquant-execution` `src/renquant_execution/alerts.py` at the
          commit this umbrella already pins (`91c7bf88`, pinned by #583) over
          `live/alerts.py`. The whole delta is three lines from execution#40:
          the `from renquant_common.notify import encode_header` import and its
          two call sites (the urllib `headers={"Title": ...}` and the curl
          `-H "Title: ..."` fallback). No other line changes
          `[VERIFIED — diff of the two files is exactly 19a20 / 20a22 /
          195c197,198 / 217c220]`.

WHY/DIR:  A byte-identical twin that has silently diverged is worse than no
          twin rule: the tripwire that exists to catch "a fix landed on one
          side only" is now red for a known reason, and a permanently-red
          guard is one people learn to skip. This is that exact class
          (audit C1-a, cf. the 2026-06 `self._config` incident).

          **This is hygiene on a red guard, not a live defect.** The honest
          impact statement, after enumerating rather than sampling: the only
          WIRED alert title is `live/runner.py:1266`
          `f"{label} [{run_mode}] {tag}"`, and every component is ASCII —
          `label` from a directory name plus a `[READONLY]...` prefix,
          `run_mode` in {sell-only, full}, `tag` one of eight ASCII literals.
          The one non-ASCII title in the tree, `live/stream_watchdog.py:135`
          (`WATCHDOG {sym} −{pct}%`, U+2212), is in a module nothing invokes —
          its only references are its own test and a doc, and no launchd job
          names it. So `encode_header`'s absence changes nothing today; it is
          a trap armed for whoever wires that module or adds a non-ASCII title.

EVIDENCE:
  artifact:      `live/alerts.py` (3-line delta), `tests/test_alerts_header_encoding.py`
                 (new committed oracle), + this record.
  oracle:        the new tests FAIL on the pre-sync copy and PASS on this head —
                 they distinguish the bug from the fix rather than exercising the
                 path. Swapping `origin/main`'s `live/alerts.py` into this tree:
                 **2 failed, 1 passed, 1 skipped**, both failures
                 `UnicodeEncodeError: 'latin-1' ... '\u2212'` at the urllib
                 header and the curl argv. Restoring this head's copy:
                 **3 passed, 1 skipped**
                 `[VERIFIED — .venv/bin/python -m pytest -q tests/test_alerts_header_encoding.py,
                 run both ways in a fresh clone of this branch]`.
                 Related suite: `test_alerts / test_alert_lifecycle /
                 test_runner_trade_ntfy / test_sell_ntfy_pnl /
                 test_daily_104_shadow_notify / test_reject_notify_disposition`
                 plus the new file -> **147 passed, 1 skipped, 1 failed**; the one
                 failure is `test_sell_ntfy_pnl.py::test_pnl_computation_uses_current_price`,
                 which fails IDENTICALLY with the pre-sync `live/alerts.py` in place
                 — a drifted source-text assertion, pre-existing and unrelated
                 `[VERIFIED — same test run against both copies]`.
  prod or exp:   prod — the live alert sender. Behaviour is unchanged for every
                 title currently produced, because `encode_header` is identity-
                 preserving for pure-ASCII input; the change is only observable
                 on a non-ASCII title, and no wired path emits one.
  existing data: yes — read the deploy machine's `make test` result, the
                 umbrella tree, and the pinned assembly READ-ONLY. Nothing
                 generated, no live state or config touched.
  best-known?:   yes — copying the already-pinned execution revision verbatim
                 is the only option that restores the byte-identity the manifest
                 asserts. Hand-porting the three lines would re-open drift on
                 the next edit; reclassifying the pair as `diverged_pin` would
                 retire a guard that is doing its job.
  scope:         "this is `live/alerts.py` (prod), vs existing best = today's
                 pre-#40 copy that passes a raw title into an HTTP header. The
                 import is satisfiable in the runtime that actually trades:
                 under `env -i` with only the pinned assembly's PYTHONPATH,
                 `from renquant_common.notify import encode_header` resolves —
                 `renquant-common/src` is first on that path and the pinned copy
                 defines it. NOTE this is the first `renquant_common` import
                 under `live/` (0 files today), so it is a real new cross-package
                 edge, deliberately accepted to keep the twin byte-identical."

NEXT: merge is not deploy — this machine's umbrella is behind (`f85a6393` vs
`2aa9de46`) and picks the change up only when the operator syncs, which is the
same pending grant as #583's pin cutover. After that, `make test` on the deploy
machine should return to 6260 passed with `byte_identical:alerts.py` PASS; that
is the acceptance check. Separately, `live/stream_watchdog.py` is unwired — if
it is meant to run, that is its own decision, not a side effect of this PR.
