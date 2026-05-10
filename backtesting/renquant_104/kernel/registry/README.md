# kernel.registry — MLflow Artifact Registry

**Status (2026-05-10):** foundation laid; one proof-of-concept wiring
(`panel-rank-calibration.json`) is in place. All other artifacts are still
on the legacy local-file path. Migration is opt-in per artifact.

## Why

Every model / calibrator / data parquet should become traceable via an
MLflow run-id, so that a question like *"which panel-LTR + which
calibrator + which fundamentals snapshot produced this trade?"* has a
provenance chain instead of a guess. Per CLAUDE.md §5.6 ("definition of
fixed = 24h audit clean") and §5.13 (single source of truth, range-bounded
outputs, side-config aliasing), having durable run-id-stamped artifacts
is foundational for the audit gates.

## Public surface

```python
from kernel.registry import (
    init_tracking,           # set MLFLOW_TRACKING_URI
    start_run,               # context manager → run_id
    log_artifact_with_meta,  # atomic upload + JSON sidecar
    resolve_uri,             # mlflow:// or local path → local Path
    register_model,          # Model Registry handle
)
```

See `mlflow_registry.py` for the full docstrings. URI scheme:
`mlflow://<32-hex-run-id>/<artifact_path>`.

## Tracking backend

Default: local file backend at `file:./mlruns` (no server required).
Override via env var `RENQUANT_MLFLOW_TRACKING_URI` (e.g. SQLite for
multi-host: `sqlite:///mlflow.db`). Migrate to a real server only when
the workstation needs to share artifacts with other machines.

## Artifact registry — writers, readers, migration order

The table below is the canonical map. **Code is source of truth** — when
moving an artifact onto MLflow, update both this file and the adapter
that resolves the URI.

| Artifact | Writer (production code path) | Readers | Status |
|---|---|---|---|
| `panel-rank-calibration.json` | `training_panel/global_calibrator.py::GlobalPanelCalibration.save` (driven by `scripts/fit_panel_calibrator.py`) | `kernel/panel_pipeline/job_panel_scoring.py::LoadGlobalCalibrationTask`, `kernel/preflight.py`, `scripts/fit_panel_calibrator.py` (overwrite guard) | **PoC wired** — parallel-write gated on `RENQUANT_MLFLOW_LOG=1` |
| `panel-ltr.alpha158_fund.json` (and stub `panel-ltr.json`) | `training_panel/pp_panel_training.py::SaveArtifactTask` | `kernel/panel_pipeline/job_panel_scoring.py::PanelScorer`, every adapter (`adapters/{lean,runner,sim}_adapter.py`), `scripts/audit_oos_ic_drift.py`, `scripts/finalize_challenger.py`, `kernel/model_acceptance.py` | Legacy file only — **next migration target** |
| `ngboost-head.alpha158_fund.json` | `training_panel/pp_panel_training.py::NGBoostSaveTask` | `kernel/panel_pipeline/job_panel_scoring.py::ApplyNGBoostHeadTask` (off by default) | Legacy file only |
| `spy-gmm-regime.json` | `kernel/regime.py::fit_spy_gmm` (driven by `scripts/fit_spy_gmm.py`) | `kernel/pipeline/job_regime.py`, `kernel/preflight.py`, `kernel/macro.py` | Legacy file only |
| `earnings-calendar.json` | `scripts/fetch_earnings_calendar.py` | `kernel/earnings_surprise.py`, `kernel/pipeline/job_earnings_blackout.py`, panel feature builder | Legacy file only |
| `watchlist-correlation.json` | `scripts/build_watchlist_correlation.py` | `kernel/portfolio_qp/`, `kernel/sizing.py`, rotation tasks | Legacy file only |
| `panel-calibration-{BULL_CALM,BULL_VOLATILE,BEAR,CHOPPY}.json` | `scripts/fit_panel_calibrator.py` (when `--regime` passed) | regime-conditional calibration in `kernel/panel_pipeline/job_panel_scoring.py` | Legacy file only |
| `data/sec_fundamentals_daily.parquet` | `scripts/fetch_sec_fundamentals.py` | `kernel/fundamentals.py` | Legacy file only |
| `data/runs.db` | `kernel/persistence.py::record_pipeline_run` (every InferencePipeline) | `scripts/dashboard_*.py`, audit scripts | **Stays local** — schema mutates every run, not an artifact |
| Model checkpoints under `models/<TICKER>/*.json` | `training/` (legacy 101-103) | archived strategies only | Skip — archived |

**Migration order recommendation** (smallest blast radius first):

1. `panel-rank-calibration.json` ← currently PoC wired. Promote to
   default-on once Track D / WF gate cron is verified writing
   `mlflow://` artifacts daily for ≥ 7 days without incident.
2. `panel-calibration-{REGIME}.json` (4 artifacts, same writer, no live
   reader contention besides regime-conditional path).
3. `spy-gmm-regime.json` (single weekly writer; readers are concentrated
   in `kernel/regime.py`).
4. `panel-ltr.alpha158_fund.json` — **highest risk**. Touched by every
   adapter + LEAN backtest. Migrate only after #1-3 burn in for ≥ 30 days.
5. `ngboost-head.alpha158_fund.json` (currently OFF in production; safe
   to migrate any time, but low value until NGBoost is re-enabled).
6. `earnings-calendar.json`, `watchlist-correlation.json` (independent
   data feeds; can migrate in parallel once #1-3 are stable).
7. `data/sec_fundamentals_daily.parquet` (parquet, not JSON; mlflow
   handles both, but schema-evolution stories differ).

**Non-targets:** `data/runs.db` is per-run mutating SQLite — keep local.
Anything under `_archive/` is by definition out of scope.

## Opt-in env vars

| Env var | Default | Purpose |
|---|---|---|
| `RENQUANT_MLFLOW_LOG` | `0` (off) | Master switch — when `1`, parallel-write artifacts to MLflow on save. |
| `RENQUANT_MLFLOW_TRACKING_URI` | `file:./mlruns` | Override the tracking backend. |
| `RENQUANT_MLFLOW_EXPERIMENT` | `renquant-panel-calibration` (calibrator only) | MLflow experiment name. |

When `RENQUANT_MLFLOW_LOG` is unset or `0`, **all production code paths
are byte-identical to pre-PoC behaviour**. The MLflow side-effect is
strictly additive.

## Failure semantics

Per CLAUDE.md §5.13.13 (side configs are loaded weapons), §5.13.11
(NaN/inf must be guarded), and §5.13.10 (don't ship dead code paths):

* The MLflow parallel-write is **non-fatal**. If mlflow is missing,
  flaky, mis-configured, or the tracking server is unreachable, the
  legacy local-file save still succeeds. The error is logged as a
  warning and the calling pipeline keeps going.
* The legacy local-file save is **always** the source of truth. Readers
  that go through `resolve_uri()` accept BOTH `mlflow://...` and plain
  paths, so callers can stage migrations one reader at a time.
* When upgrading a reader to mlflow URIs, write paired tests pinning
  both branches (legacy file path AND `mlflow://` URI) to defend
  against §5.13.10's "is_not_None" dead-code regression.

## Tests

`tests/test_mlflow_registry.py` (16 tests, all green 2026-05-10):
* URI helpers (`is_mlflow_uri`, `parse_mlflow_uri` — 4 tests)
* `init_tracking` (1 test)
* `start_run` lifecycle + failure marking (2 tests)
* `log_artifact_with_meta` happy + error path (2 tests)
* `resolve_uri` local + mlflow round-trip (3 tests)
* `register_model` after log (1 test)
* PoC: `GlobalPanelCalibration.save` walks the real prod path with and
  without the env-var switch, and tolerates mlflow outages (3 tests)
