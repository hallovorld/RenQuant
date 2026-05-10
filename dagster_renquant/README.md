# dagster_renquant — cron-tier asset graph

Side-by-side with launchd. **Not** a replacement (yet). The point is to encode
the cron-tier dependency chain as a Dagster asset graph so that
`promote_decision` mechanically cannot fire without a fresh `wf_gate_pass`
upstream — killing the `RQ_ALLOW_NO_WF=1` bypass class **by construction**,
per CLAUDE.md §5.13.15.

## Layout

```
dagster_renquant/
├── __init__.py            # re-exports `defs`
├── definitions.py         # top-level Dagster Definitions
├── _paths.py              # repo-relative artifact path constants
└── assets/
    ├── __init__.py        # ALL_ASSETS list
    ├── data.py            # ohlcv_data, sec_fundamentals
    ├── training.py        # regime_artifact, panel_features, panel_model, calibrator
    └── promote.py         # wf_gate_pass, promote_decision
```

Tests live in `tests/test_dagster_assets.py` (13 tests, all pass under the
project's pytest with `xdist`).

## Asset graph

```
ohlcv_data ────┬──→ regime_artifact
               │
               └──→ panel_features ──→ panel_model ──┬──→ calibrator ─────────┐
                                                     │                        │
sec_fundamentals ──┘                                 └──→ wf_gate_pass ───────┤
                                                                              │
                                                                              ▼
                                                                    promote_decision
```

Freshness policies (cron-tier explicit, per §5.13.6):

| asset             | freshness            | rationale                                                |
|-------------------|----------------------|----------------------------------------------------------|
| `ohlcv_data`      | 1d                   | end-of-day refresh                                       |
| `sec_fundamentals`| 7d                   | quarterly data                                           |
| `regime_artifact` | 1d                   | SPY GMM updated daily                                    |
| `panel_features`  | 1d                   | feature parquet rebuild                                  |
| `panel_model`     | 7d                   | fwd_60d label → weekly retrain floor                     |
| `calibrator`      | 7d                   | tracks `panel_model`                                     |
| `wf_gate_pass`    | 7d                   | weekly walk-forward gate                                 |
| `promote_decision`| (none — event-gated) | fires only when both upstreams fresh; cadence = upstream |

## Running locally

```bash
source .venv/bin/activate
dagster dev -m dagster_renquant.definitions
# UI: http://127.0.0.1:3000
# GraphQL info: curl -s http://127.0.0.1:3000/server_info
```

Smoke test confirmed (2026-05-10): server boots in ~3s, daemons load
(`AssetDaemon`, `FreshnessDaemon`, `SchedulerDaemon`, …), GraphQL responds.

## Why this kills `RQ_ALLOW_NO_WF=1`

`scripts/train_104.py` historically wrote `RQ_ALLOW_NO_WF=1` into the env
of every promote shell, so every promote bypassed the gate (CLAUDE.md
§5.13.15 — "the gate was theatrical"). The Dagster graph encodes the
dependency at the topology level:

- `promote_decision.deps = [wf_gate_pass, calibrator]`
- Materialising `promote_decision` requires materialising **both** upstreams
  in the same run (or having them already materialised within freshness).
- The `wf_gate_pass` asset body **fails fast** if the sentinel JSON is
  missing or `passed=False`. There is no env-var override.

Removing `wf_gate_pass` from `ALL_ASSETS` breaks `Definitions(...)`
construction itself — `tests/test_dagster_assets.py::
test_removing_wf_gate_pass_breaks_definitions_load` pins this.

## Migration plan from launchd

We migrate **plist by plist**, only after the corresponding asset has
been verified to materialise correctly twice in a row through Dagster.
Until each cell is checked off, the launchd plist remains the source of
truth and the Dagster asset is observational only.

| launchd plist                                       | replacing asset(s)              | status                |
|-----------------------------------------------------|----------------------------------|-----------------------|
| `com.renquant.conditional-retrain104.plist`         | `regime_artifact`, `panel_features`, `panel_model`, `calibrator` | **observe-only**      |
| `com.renquant.weekly-wf-promote.plist`              | `wf_gate_pass`, `promote_decision`                               | **observe-only**      |
| `com.renquant.monthly-calibrator-refresh.plist`     | `calibrator` (monthly variant)                                   | **observe-only**      |
| `com.renquant.retrain-alpha158-linear.plist`        | (no Dagster mirror yet — keeps own cron)                         | **launchd-only**      |
| `com.renquant.screen-watchlist.plist`               | (no Dagster mirror yet — keeps own cron)                         | **launchd-only**      |

A plist gets disabled (`launchctl unload <plist>`) only **after**:

1. The mirroring Dagster asset has materialised successfully twice on the
   same hardware, with fresh upstreams.
2. A side-by-side run shows the artifact mtimes and content match what
   launchd produced.
3. `tests/test_dagster_assets.py` has at minimum the same coverage as
   the launchd shell's smoke checks (e.g. `tests/test_smoke_test_model.py`).

The intent is that until step 3, launchd remains the system of record
and Dagster is purely a graph-aware monitor.

## What this is **not**

- Not a re-implementation of training. Asset bodies are validate-output
  stubs (existence + size sanity). The actual compute remains in
  `scripts/conditional_retrain_104.sh`, `scripts/weekly_wf_promote.sh`,
  `scripts/train_104.py`, etc.
- Not a replacement for `tests/test_smoke_test_model.py` or any of the
  runtime-invariant pytest suites. The Dagster layer adds a *graph*
  invariant on top.
- Not yet wired into `pyproject.toml` (intentional — this is opt-in and
  observation-only until the migration plan is checked off).
