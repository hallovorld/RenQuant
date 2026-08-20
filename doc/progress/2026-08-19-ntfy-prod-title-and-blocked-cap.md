# prod ntfy was unreadable: no action in the title, 59 BLOCKED-ROTATION in the body

STATUS:   fix implemented in `live/runner.py`; `python -m py_compile` clean and
          `_action_headline` exercised against today's REAL order/exit payload
          shapes plus the failed-exit, summarisation and empty cases.
          Notification composition only — no order path, no sizing, no gate,
          no state. Prepared without touching the live working tree (scratch
          copy + contents API); the umbrella tree is untouched at `cf3df09`.

WHAT:     The operator asked: "那我怎么看不到prod干啥了？prod坏了？还是ntfy msg
          没发？" **Neither — prod ran, the POST succeeded, and the message was
          unreadable.**

          Measured on `logs/daily_104/2026-08-19.log`; the log format is
          `ntfy sent: <TITLE> | <BODY>`:

            lane                 title                          body
            prod                 RENQUANT-104 [full] PENDING    3830 B
            shadow vol-window    ...SHADOW-ACTION                188 B
            shadow blend x5      ...SHADOW-ACTION               ~3700 B each

          Two compounding defects:
          (1) the prod TITLE is a status word carrying NO content, and the
              title is the only part a phone lock screen reliably shows — the
              body is collapsed;
          (2) the body held **59 BLOCKED-ROTATION segments against 2 real
              order segments**. At 3830 B it exceeded `_NTFY_BODY_MAX_BYTES`
              (3800), so it WAS truncated, and the cut landed inside the
              blocked list — the trailing `regime=/conf=/held=/eq=` context
              never reached the phone at all.

          So the operator could read every SHADOW lane and not production.

          FIX 1 — `_action_headline()` appends the action to the title:
            OLD  RENQUANT-104 [full] PENDING
            NEW  RENQUANT-104 [full] PENDING: BUY PANW x3, EXIT CRWD x5.0
          Failed exits sort FIRST (a rejected sell is the one case that may
          need a human at the broker, so it must not be the token summarised
          away); >3 actions summarise as `+N`; **no actions leaves the title
          byte-identical to today**, so quiet days and any existing
          title-matching consumer are unaffected. Sizes are kept
          (`BUY PANW x3` != `BUY PANW x300`), prices are not — they are in the
          body and the title's budget buys more tickers instead of cents.

          FIX 2 — `BLOCKED-ROTATION` capped at 3 in the body plus a count
          line. Bug L's fix (`ROT-BLOCKED-NTFY`, 2026-04-25) STAYS: the
          operator should know when the system wanted to swap and was vetoed.
          What changed is the volume. The count is what carries the signal;
          the full 59-item list stays in the run log.

          Effect on today's actual message: **body 3830 B -> 440 B**, no
          truncation, context restored.

WHY/DIR:  `93adb20` ("fleet callsigns + terse shadow bodies (operator:
          简练,人话)") gave the SHADOW lanes terse bodies on this exact
          operator request. Production never got it — so the readability gap
          ran the wrong way round: the lanes that move no money were legible
          and the one that does was not. An alert the operator cannot read is
          not an alert; they either stop looking or act on a guess, and both
          are worse than silence. Direction: the decision goes where it
          survives truncation, and every diagnostic added to this surface must
          justify its bytes against the decision it is annotating — a
          diagnostic that evicts that decision is worse than no diagnostic.

EVIDENCE:
  artifact:      `live/runner.py` (+80/-3): `_action_headline`,
                 `_ROT_BLOCKED_NTFY_MAX`, `_TITLE_ACTION_MAX`, the capped
                 blocked-rotation block, and the title composition.
  prod or exp:   **exp** — scratch copy + GitHub contents API onto a branch;
                 the live umbrella working tree was never written and remains
                 clean at `cf3df09`. Deploy is operator-gated as always.
  existing data: read by me from the live log, not assumed —
                 - per-message title/body byte counts for all 7 notifications
                   emitted on 2026-08-19 [VERIFIED]
                 - 59 BLOCKED-ROTATION segments vs 2 order segments; 3830 B
                   vs the 3800 B cap, i.e. truncation CONFIRMED, not inferred
                   (the `…[truncated]` marker is present in the logged body)
                 - the prod POST succeeded: `renquant_common.notify.send`
                   logs `ntfy send failed` on any failure and no such line
                   exists in today's log [VERIFIED]
  best-known?:   yes for what it fixes. Recomputed on today's real payload:
                 title -> `RENQUANT-104 [full] PENDING: BUY PANW x3, EXIT CRWD
                 x5.0`; body -> 440 B. Failed-exit precedence, `+N`
                 summarisation and the empty-headline no-op all exercised.
  scope:        notification composition only. Does not touch order emission,
                sizing, gates, rotation logic or state.

NEXT:      NOT fixed here, and it is the sibling defect on the same surface:
           every SHADOW notification lost its `SHADOW/HYPOTHETICAL (no live
           orders)` body banner on 2026-08-05 — 15 days in which shadow alerts
           have read like real orders, which is how this investigation
           started. Filed as `renquant-orchestrator#1014`; the removing commit
           is still unlocated (`git log -S` across five repos found nothing,
           and the literal exists nowhere in current code), so it needs its
           own change rather than being folded in here.
