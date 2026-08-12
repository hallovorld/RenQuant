# 2026-08-12 — the stack that trades still hung on an account read

STATUS:   FIXED (2026-08-12). `tests/test_live_alpaca_bounded_timeout.py` ->
          **11 passed** `[VERIFIED — .venv/bin/python -m pytest -q]`.

WHAT:     Ports renquant-execution#41's bounded account-read timeout to
          `live/alpaca_broker.py`: a `_bounded_account_timeout()` context
          manager that temporarily WRAPS the SDK session's `request` with
          `(connect=5s, read=10s)` around the two P-BROKER-CONNECT preflight
          reads — the `get_account()` at the end of `connect()` and the one in
          `get_account_value()` — restoring the session exactly on exit.

WHY/DIR:  execution#41 fixed the subrepo copy. **The order path does not import
          it.** `live/runner.py:33` is `from .alpaca_broker import AlpacaBroker`
          — a relative import — so the module that actually places orders was
          still unbounded `[VERIFIED — grep of the import + `_bounded_account_timeout`
          count: 6 in the execution copy, 0 in live/]`. The pair is a deliberate
          `diverged_pin`, so the behaviour is ported, not the file.

          The exposure is real, not theoretical: the alpaca-py SDK exposes no
          timeout knob — `RESTClient.__init__` has no `timeout` parameter and
          `_one_request` never passes one — so the read inherits requests'
          default `timeout=None` `[VERIFIED — inspect.signature on the installed
          SDK]`. That is the 2026-08-11 07:00 abort that cost a ~12 min intraday
          cycle.

          The twin-parity tripwire is what surfaced this, via
          `diverged_pin:alpaca_broker` going red after #41 merged. I initially
          read that as manifest staleness needing a `--write-manifest` re-pin.
          It was not: re-pinning would have silenced a tripwire that was
          pointing at a live gap.

EVIDENCE:
  artifact:      `live/alpaca_broker.py` (+1 context manager, 2 call sites
                 wrapped, 2 module constants) and
                 `tests/test_live_alpaca_bounded_timeout.py` (11 tests).
  prod or exp:   prod — the live broker adapter. Order submission is deliberately
                 OUTSIDE the context and is asserted unbounded by test.
  existing data: yes — read the umbrella tree, the execution twin, and the
                 installed SDK READ-ONLY. Nothing generated; no live state,
                 config, or order touched.
  best-known?:   yes — wrap-not-replace is the same minimal surface Codex
                 settled on for execution#41: swapping in a fresh session would
                 silently drop the SDK's seeded proxies/verify/cert/cookies/
                 hooks/params/auth and mounted adapters. An unusable session
                 RAISES rather than yielding unbounded, because a silent
                 fallback defeats the fast-fail contract.
  scope:         "this is `live/alpaca_broker.py`'s two preflight account reads
                 (prod), vs existing best = today's unbounded read on the module
                 the runner actually imports. It bounds NO-PROGRESS stalls —
                 requests' timeout is an inactivity timer, not a wall-clock cap
                 on the whole request — so a peer that keeps trickling bytes is
                 still unbounded. Order submission is untouched."

NEXT: the `diverged_pin:alpaca_broker` manifest entry still needs re-pinning
once both stacks carry the change — that is a renquant-orchestrator PR against
`scripts/check_twin_parity.py --write-manifest`, deliberately NOT bundled here,
so the tripwire keeps pointing at the gap until the gap is actually closed on
this machine. Merge is not deploy: the umbrella sync is still pending.
Not addressed: the trickling-peer mode, which no timeout of this kind closes.
