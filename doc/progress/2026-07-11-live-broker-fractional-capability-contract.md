# Live broker fractional-capability contract wiring

Date: 2026-07-11
Finding: renquant-orchestrator#471 (evidence packet for s104#55/#56)

## What

`adapters/commit_contract.py::fractional_capability_gate` requires the
broker to expose a callable `is_fractionable` and a callable no-submit
classifier (`classify_broker_result` or `is_no_submit_status`) before
`execution.fractional_shares.enabled=True` may admit a BUY. The LIVE
runner's broker (`live/alpaca_broker.py::AlpacaBroker(live.broker.
BaseBroker)`) had neither — `renquant-execution/src/renquant_execution/
alpaca_broker.py` has both, but that adapter is not wired into the live
path, and the two `BaseBroker` classes are unrelated (different repos,
different hierarchies).

Consequence: setting the flag today trips the gate and fail-closes ALL
buy emission (not just fractional) — worse than status quo, and NOT what
enabling fractional trading is supposed to do.

## Fix

Additive only, mirrors renquant-execution's implementation exactly (same
caching/fail-closed contract, same vocabulary) rather than swapping the
live broker for the renquant-execution adapter — that would be a much
larger, riskier change to the live order-placement path for a currently
inert (flag-off) capability probe:

- `live/broker.py::BaseBroker`: added the `NO_SUBMIT_STATUSES` vocabulary
  + module-level `is_no_submit_status` + the shared, delegating
  `@staticmethod is_no_submit_status` — identical to renquant-execution's
  `BaseBroker.is_no_submit_status`, so every current/future broker
  subclass answers the gate the same way without each reimplementing it.
- `live/alpaca_broker.py::AlpacaBroker`: added `is_fractionable`
  (cached-on-success, not-cached-on-failure, fail-closed to `False`) —
  same contract as renquant-execution's `AlpacaBroker.is_fractionable`.

`execution.fractional_shares.enabled` is untouched and stays default-off.
This PR does not enable fractional trading; it makes the mechanism
mechanically exercisable once separate operator authorization (tracked
via strategy-104#55/#56, currently held pending renquant-orchestrator's
R-PIN deployment-authority migration) actually lands.

## Tests

- `tests/test_s_frac_stage0_commit_contract.py::TestCapabilityGate::
  test_real_alpaca_broker_satisfies_broker_fractional_contract` — the
  actual `AlpacaBroker` (not the existing `FakeBroker` stand-in) now
  satisfies `broker_fractional_contract`; no network calls (construction
  only).
- `TestLiveAlpacaBrokerFractionalContract` (7 new tests): `is_fractionable`
  caching (True/False both cached, failure NOT cached, case-insensitive
  key), `BaseBroker.is_no_submit_status` classification (known statuses,
  case-insensitivity, non-matching statuses), and that `AlpacaBroker`
  correctly inherits the static method.

All 7 confirmed meaningful via stash-revert (fail against pre-fix code).
Full relevant suite (`test_s_frac_stage0_commit_contract.py` +
`test_broker_side_stops.py`): 78 passed. Broader repo suite run from this
scratch clone shows ~200 unrelated failures (`ModuleNotFoundError:
renquant_pipeline`/`renquant_artifacts` — missing sibling-package
PYTHONPATH wiring specific to a fresh clone outside the live tree's conda
env); spot-checked several against this exact root cause, none touch
`live/`, `broker`, `alpaca`, `fractional`, or `commit_contract`.
