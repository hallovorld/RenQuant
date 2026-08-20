# My own no-trade fix pooled two different gates — naming the dominant one

STATUS:   delivered. `_no_trade_reason` + updated/extended tests
          (71 + 10 passed; the rotation file was 6 before review).
          Notification composition only — no order path, no gate, no config.

WHAT:     RenQuant#598 (merged, deployed) made a post-scoring rotation decline
          outrank the pre-scoring vol drop. Correct, and it fixed the wrong
          cause being named. But its label POOLED two distinct gates:

            no trade (rotation_nonpositive_expected_return(60))

          On the real 2026-08-20 payload only **13** of those 60 were
          nonpositive-expected-return. The other **47** were
          `negative_raw_signal` — a different gate — and **25 of those had a
          POSITIVE expected return** (AFRM +34%, META +19%, SOFI +9%), declined
          on panel score alone. Calling that "nonpositive expected return" is
          false, and it reproduces one layer finer the exact defect #598 was
          written to fix: a message naming a cause that is not the cause.

          FIX: `_rotation_signal_block` now counts per reason and returns the
          DOMINANT one with its own name and count —
          `rotation_negative_raw_signal_no_long(47)` on that payload.

          TWO REVIEW CORRECTIONS, both of which I had gotten wrong [codex on
          RenQuant#599]. (1) This doc previously said "ties break
          deterministically" — they did NOT. `-ord(kv[0][0])` compares ONE
          character and both reasons start with "n", so equal counts fell back
          to dict insertion order and reversing the payload changed the
          notification. Now max-count then `min()` over the tied FULL strings.
          (2) `"expected_return" in reason` would classify a future
          `missing_expected_return` — a plumbing fault — as an economic
          decline; replaced by an exact frozenset of the two reason constants.

          On (2): an enumerated allowlist is the shape I called a defect in
          orch#1013, so the difference is worth stating. There, an unlisted
          order type was silently DROPPED, so the default had to be "include".
          Here an unlisted reason merely fails to be ELEVATED above the
          vol-gate fall-through, so the default is "do not claim this is the
          cause" — what this function exists to guarantee.

          Mirrors the `qp_counts` max() convention already
          used below it for the same multi-reason situation.

WHY/DIR:  G-F. I caught this by exercising my own merged fix against the real
          payload instead of trusting that it read correctly — the fix was
          already deployed and would have shipped a subtler version of the
          error it removed. Precision in a cause label is not cosmetic here:
          the previous mislabel had already moved the operator toward loosening
          a live risk limit.

EVIDENCE:
  artifact:      `live/runner.py`, `tests/test_no_trade_reason_rotation_economic.py`.
  prod or exp:   **exp** — scratch copy + contents API; live tree not written.
  existing data: the 13/47 split and the 25 positive-ER names are read from
                 logs/daily_104/2026-08-20.log `BuildPairsTask ... (prefilter)`
                 lines [VERIFIED].
  best-known?:   yes. `test_the_dominant_reason_is_named_never_pooled` FAILS
                 against the pooled version now in production and passes after,
                 so the suite distinguishes the fix from decoration. The three
                 preservation tests (vol gate still named when it IS the cause;
                 non-economic rotation blocks do not hijack; admission/QP still
                 outrank) pass against both.
  scope:        the no-trade reason string only.

NEXT:      Deploy alongside the next umbrella FF. Separately: the 47-vs-25 split
           is itself the substantive finding — a panel score with no
           demonstrated skill in BULL_CALM (genuine_ic -0.032) is vetoing names
           the expected-return model likes. Filed for its own measurement.

REVIEW:    codex (haorensjtu-dev).
