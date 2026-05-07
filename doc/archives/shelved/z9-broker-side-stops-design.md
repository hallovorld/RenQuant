# Z9 — Broker-side stop orders (design)

**Date:** 2026-04-28
**Status:** Broker layer + tests **shipped**. Runner integration **deferred** to operator review.

## Why

NVTS post-mortem: ticker dropped −12% in 24h between two 30-min cron ticks. The polled stop_loss check ran on a stale price; by the time the next cron fired, the bot was already deep in the loss. **Polled stops are gated by cron cadence; broker-side stops trigger in milliseconds regardless of when our process polls.**

## What shipped (commit-ready, behind feature flag)

1. `BaseBroker.supports_broker_side_stops() / place_stop_order() / cancel_order()` — new abstract layer with safe NotImplementedError defaults so non-supporting brokers fail loudly rather than silently no-op.
2. `AlpacaBroker.place_stop_order()` — submits a `StopOrderRequest` GTC at the requested price; reuses the same `ALPACA-ACCT-STATUS` precheck as `place_order()`.
3. `PaperBroker.place_stop_order() + _check_stops()` — in-memory simulation; tests + sim-mode runner can call `_check_stops()` after `set_price()` to verify trigger semantics.
4. 19 regression tests in `tests/test_broker_side_stops.py` — covering: defaults, place/cancel, no-fire-above-stop, fire-at-stop, gap-down fill, partial-position clipping, account-status guard.

**Default behavior unchanged**: no broker has automatic stops yet. The runner does not call `place_stop_order()` anywhere. Production state is identical to pre-Z9 commit.

## What's NOT shipped — needs operator decision

The runner integration has 4 design forks I'm not comfortable deciding alone:

### Fork A: which stop level to use

| Option | Pros | Cons |
|---|---|---|
| `regime_params.{regime}.stop_loss_pct` | Symmetric with polled stop | BULL_CALM=15% wouldn't have caught NVTS at −12% |
| `regime_params.{regime}.max_single_day_loss_pct` | Tighter (10% in BULL_CALM); would have caught NVTS | Wider trigger surface; will fire more often on normal vol |
| `entry_price × (1 - vol_scaled_pct)` | Per-position; respects ticker vol | Vol-scaled stops are wider for high-vol names — paradoxically anti-protective for the cases we care about |
| Constant `0.10` (10%) regardless of regime | Predictable | Loses the regime-specific design |

**My recommendation**: `max_single_day_loss_pct` — tightest available limit, panel-derived (not made-up), would have caught NVTS.

### Fork B: when to place + cancel the stop

| Trigger | When |
|---|---|
| `BUY` fills successfully | Place stop at entry × (1 − pct) |
| `SELL` (full liquidation) | Cancel any associated stop |
| `TRIM` (partial sell) | Reduce stop quantity proportionally OR cancel + re-place |
| `TOPUP` (add to existing) | Cancel + re-place at weighted-avg entry × (1 − pct) |
| `STATE-EXT-SELL` (manual disposition detected) | Cancel any stale stop for that ticker |

The TRIM/TOPUP edge cases need real thought — easy to get into a state where the broker has a stop for shares we no longer own, or for a different qty than current holding.

### Fork C: storing the order_id

`live_state.{broker}.json` needs a new field:

```json
"stop_orders": {
  "NVDA": {"order_id": "abc-123", "stop_price": 425.0, "qty": 10}
}
```

with garbage collection on every `commit()` for tickers no longer held. Simple but adds another state contract that auto-revert + state-paths needs to handle.

### Fork D: which broker to enable on

- `paper`: fine — simulation-only, no real capital
- `alpaca-paper`: fine — Alpaca paper account, no real capital
- `alpaca` (live): **operator must explicitly opt in** — this places real GTC orders that survive across cron restarts

I'd default to `enabled=false` everywhere. Operator flips per-broker after reviewing one cron cycle's worth of stop placements + cancels in alpaca-paper.

## Proposed integration sequence (when ready)

1. Add `live.broker_side_stops.enabled` (default false) + `pct_source` (default `max_single_day_loss_pct`) to strategy config schema.
2. Extend `live_state.{broker}.json` schema: `stop_orders: {ticker: {order_id, stop_price, qty}}`.
3. Wire into `runner.py::commit()` after the BUY loop:
   - For each successful BUY: `stop_price = price × (1 − pct)`; `place_stop_order()`; record id in state.
4. Wire into the SELL/TRIM/TOPUP code paths to cancel/replace as in Fork B.
5. Wire into `STATE-EXT-SELL` (Z2 path) to cancel orphaned stops on manual disposition.
6. Add 5+ regression tests at the runner-integration level.
7. Rehearse rollback per principle 5.5: simulate a stale stop, run a paper cycle, verify stop is cancelled cleanly.
8. Operator enables on `alpaca-paper` for one trading day; reviews stop-fill log.
9. If clean → enable on `alpaca` live.

## What this protects

- **NVTS-style intraday gap-downs** between cron ticks (the actual incident)
- Cron failures / latency: even if our process is dead, the broker still enforces the stop
- Partial fills on illiquid names: broker handles the ladder, we just consume the resulting fills next cron

## What this does NOT protect

- **Pre-market gaps** below stop: stop fills at the gap, possibly far below the requested price
- **Halts** (CB, news halt): stop fills at re-open
- **Ticker delistings**: stop becomes inactive

These are residual risks that need separate mitigations (position size cap on illiquids; halt detection; etc.).

## Decision gate

Operator review answer needed for forks A–D before runner-integration step 3 lands.
