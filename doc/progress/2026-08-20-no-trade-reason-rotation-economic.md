# A vol gate was blamed for an economic no-trade, and it nearly moved a risk limit

STATUS:   delivered. `_no_trade_reason` + 5 tests. Notification composition
          only — no order path, no sizing, no gate, no config, no state.

WHAT:     The 2026-08-20 live message read

            no trade (risk_gate_vol_dropped(30))

          while the SAME run's funnel-integrity line read

            verdict=ECONOMIC_NO_TRADE fired=0 structural=False
            candidates_final=84 buys=0

          and 61 rotations were blocked `nonpositive_expected_return_no_long`.
          So 84 candidates survived every gate, WERE scored, and the model
          declined all of them on economics. **The vol gate was not binding.**

          CAUSE: `risk_gate_vol_dropped` is the LAST entry in the priority
          list, commented "only the cause when nothing later applied".
          Something later DID apply — but the rotation-side economic block has
          no counter (`grep -c nonpositive_expected_return live/runner.py` → 0),
          so the loop fell through and named the vol gate.

          The function's own docstring says the 2026-06-01 rewrite reordered
          the list for exactly this failure ("Old ordering put
          risk_gate_vol_dropped ahead of admission/QP and surfaced 'no trade
          (vol_dropped(10))' even when 72 of 82 candidates survived"). It fixed
          the ORDERING and never added the counter, so the same wrong answer
          returned through the gap.

          FIX: `_rotation_economic_blocks(ctx)` counts rotations declined for
          `expected_return` / `negative_raw_signal` reasons and is consulted
          immediately ABOVE the vol-gate fall-through — a post-scoring economic
          decline outranks a pre-scoring drop. Earlier binding blocks
          (admission, QP) keep precedence; non-economic rotation blocks do not
          hijack the reason; the vol gate is still named when it IS the cause.

WHY/DIR:  G-F (observability). This is not a cosmetic label. **The operator
          read that message as evidence the vol cap was starving the book and
          moved to loosen a live risk limit** — a message naming the wrong
          cause does not merely confuse, it steers capital decisions. The vol
          cap may well deserve loosening (that question is under preregistered
          test at orch#1017), but it must be decided on its own evidence, not
          on a mislabel.

EVIDENCE:
  artifact:      `live/runner.py` (`_rotation_economic_blocks` + the priority
                 entry), `tests/test_no_trade_reason_rotation_economic.py`.
  prod or exp:   **exp** — edited in a scratch copy, pushed via the contents
                 API; the live umbrella working tree was not written.
  existing data: measured from the live run, not assumed —
                 - the ntfy body, the funnel-integrity verdict and the 61
                   rotation blocks all from logs/daily_104/2026-08-20.log
                   [VERIFIED]
                 - `nonpositive_expected_return` appears 0 times in
                   live/runner.py before this change [VERIFIED]
  best-known?:   yes. The 2 tests pinning the NEW behaviour fail against the
                 unpatched runner; the 3 pinning PRESERVED behaviour pass
                 against both — so the suite distinguishes the fix from
                 decoration. Full notification suite: 83 passed, no regression.
  scope:        the no-trade reason string. Does NOT touch the vol cap, any
                gate, or any order path.

NEXT:      none blocking. orch#1017's preregistered screen decides the cap on
           its own evidence; this only stops the message from pre-judging it.

REVIEW:    codex (haorensjtu-dev).
