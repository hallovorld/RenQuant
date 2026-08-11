# 2026-08-11 - joint broker retry and timeout pin update

STATUS: candidate repair complete; umbrella CI and independent review remain
required before merge.

WHAT: Advance the `renquant-pipeline` and `renquant-execution` pins together,
and synchronize the Phase-1 umbrella broker-preflight mirror to the pinned
pipeline implementation.

WHY/DIR: The candidate must exercise the same broker retry code in the pinned
daily runtime and the umbrella preflight/simulation mirror. This is an
integration change only, not a production cutover or an alpha claim.

EVIDENCE:
- artifact: `subrepos.lock.json`; synchronized `preflight.py` and
  `preflight_pipeline/tasks/broker.py`; `tests/test_broker_connect_retry.py`.
- prod or exp: integration tests only; no production strategy, model,
  artifacts, market data, or execution configuration was written or changed.
- existing data: the exact source merge commits were independently observed
  CI-green. With pipeline at its candidate pin, the repaired combination
  passed 125 targeted umbrella tests. Strict parity measured 169 common
  files: 76 identical, 93 pre-existing allowlisted drifts, and 0 new drifts.
- best-known?: yes for the integration estimand: the first candidate's strict
  parity failure named `preflight_pipeline/tasks/broker.py`; synchronizing the
  task and its imported helper reduced that concrete newly introduced drift to
  zero. This does not establish production reliability or alpha.
- scope: lock pair plus the required umbrella mirror and regression tests.
  The live runtime remains the pinned pipeline; the umbrella mirror remains a
  Phase-1 compatibility surface.

## Pin delta

| Subrepo | Previous pin | Candidate pin | Source PR | Change |
| --- | --- | --- | --- | --- |
| `renquant-pipeline` | `e13cd3eba37856a43acb0cd16b147bf9a2cf452e` | `4aec0e35e8200c623c5353c74bd175a0871d3a9d` | [renquant-pipeline#286](https://github.com/hallovorld/renquant-pipeline/pull/286) | Three broker-connect attempts with two-second backoff, shared by runtime and legacy entry points; hard failure remains after exhaustion. |
| `renquant-execution` | `5724dc74ec2b020dac6f567d6e0d049b2c006b4e` | `91c7bf8873fda9d2806963da7a23032a6e8fbdc4` | [renquant-execution#41](https://github.com/hallovorld/renquant-execution/pull/41) | Apply timeout defaults around account reads without replacing the SDK session or its transport state. |

## Limits

- `subrepos.lock.json` is valid JSON and the candidate diff passes
  `git diff --check`.
- Combined re-queries to GitHub later encountered intermittent local
  certificate failures. That environment failure is not a green aggregate
  result; `subrepo-pin-ci-green` remains the candidate merge gate.
- Requests connect/read timeouts bound a no-progress stall. They do not create
  a strict whole-request or daily-run deadline for a peer that trickles data.

NEXT: Require green umbrella candidate-pin checks, including pin import
integrity and strict parity, plus independent approval before merge. A later
production cutover additionally requires the designated runtime promotion,
an auditable run bundle, and operator authorization. Rollback is a single
revert that restores both prior pins together.
