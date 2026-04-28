# 2026-04-28 deep audit — last 4 hours of M-series + broker-isolation code

**Trigger:** user feedback "你的 code quality 太差了" after a 4-hr burst that
shipped ~5300 lines across 24 files (commits `68b1c03` … `bfb181a`) with
**zero new tests** and a stub script committed without a working impl —
direct violations of CLAUDE.md project rule #2 ("Tests for Every Feature").

This audit is self-imposed. Findings drive the immediate follow-up commit.

---

## CRITICAL — fix immediately

### CRIT-1 — drift detector hard-fail silently admits ALL candidates through Gate B

**Where:** `kernel/panel_pipeline/job_panel_scoring.py::ApplyNGBoostTask`
(line 477 — `if pct_miss > drift_thr: log.error(...); return`)

**The bug:**
- Drift detector `return`s early without setting `cand.mu / cand.sigma` on candidates → they stay `None`.
- Gate B's `_gate_b_edge_sharpe` (line 120-121 in `task_quality_floor.py`) explicitly says:
  ```python
  if mu is None or sigma is None:
      return True, None  # no NGBoost → no signal to gate; pass
  ```
- **Net effect:** when drift fires the hard-fail (intended as "block bad scoring"), every candidate sails through Gate B with no μ/σ check. **Less safe than the silent zero-fill it replaced.**
- Live impact: if drift fires tomorrow morning's open, the model could place buys with NO quality control — exactly the opposite of the design intent.

**Fix:** when drift is over threshold, set `cand.mu = NaN` and `cand.sigma = NaN` for all candidates + holdings. Gate B's NaN check (line 129) then explicitly rejects. Better safety: also clear `ctx.candidates = []` to block buys entirely until operator retrains.

**Test gap:** no test asserted "drift hard-fail blocks subsequent buys."

### STUB-1 — `scripts/train_horizon_blender.py` ships a stub that cannot run

**Where:** `scripts/train_horizon_blender.py::build_holdout_predictions`

**The issue:**
```python
def build_holdout_predictions(...) -> tuple[list[dict], list[float]]:
    log.warning("STUB — actual run is gated on M1 panel completion.")
    # TODO when M1 panels exist: ...
    return [], []
```

The script's `main()` then short-circuits at "Hold-out build returned empty — see TODO." The script is non-functional but committed as if it were ready. Future-self picks it up, runs it, gets cryptic empty-set error and no clue what to do.

**Fix options:**
- (a) Revert from commit and re-add when M1 panels exist + impl is complete.
- (b) Replace `build_holdout_predictions` body with `raise NotImplementedError("Run after M1 (panel-ltr.{10d,20d,60d}.json) exists. See doc/components/m2-blender-design.md.")` so anyone running it gets an immediate clear error.

Going with (b) + a design doc.

---

## HIGH — fix in same follow-up commit

### TEST-1 — `kernel/state_paths.py` (72 lines, 4 functions) has zero tests

**Required tests** (`tests/test_state_paths.py`):
- `_safe_broker` returns `"unknown"` for None / empty.
- `_safe_broker` replaces `-` with `_` (alpaca-paper → alpaca_paper).
- `live_state_path` includes the safe broker tag.
- `live_state_legacy_path` is the bare `live_state.json`.
- `resolve_live_state_read` returns primary path + `False` when primary exists.
- `resolve_live_state_read` falls back to legacy + `True` when only legacy exists.
- `resolve_live_state_read` returns primary + `False` when NEITHER exists.
- `runs_db_path` tag insertion: `data/runs.db` + `alpaca` → `data/runs.alpaca.db`.
- `runs_db_path` is **idempotent / safe** when base is already tagged (currently broken — `data/runs.alpaca.db` + `paper` → `data/runs.alpaca.paper.db`, double-tagging).

### TEST-2 — drift detector behaviour change has zero tests

**Required tests** (`tests/test_panel_scoring_drift.py`):
- `test_zero_fills_when_drift_under_threshold` — 1 col missing of 27 → fills with 0, predicts normally.
- `test_skips_scoring_when_drift_over_threshold` — 14/27 missing → ApplyNGBoostTask returns without setting μ/σ.
- `test_post_fix_blocks_candidates_through_gate_b` (CRIT-1 regression) — after drift hard-fail, Gate B rejects all candidates.
- `test_no_warning_when_no_missing` — clean panel, no log spam.
- `test_threshold_configurable_via_max_feature_drift_pct` — operator override works.

### TEST-3 — conformal Gate B reader has zero tests

**Required tests** (`tests/test_quality_floor_conformal.py`):
- Returns None when `gate_b_thresholds.json` missing.
- Returns None when regime missing in artifact.
- Returns None when JSON corrupt (no crash).
- Returns the τ value when present and well-formed.
- Falls back to config threshold when reader returns None.
- Use_conformal=false bypasses reader.

### VAL-1 — broker_name path-traversal hardening

**Where:** `kernel/state_paths.py::_safe_broker`

`broker_name` is currently sanitised only by `replace("-", "_")`. A malicious or buggy caller passing `"../../etc/passwd"` would produce `live_state.../../etc/passwd.json` which `Path` resolves outside `strategy_dir`. In practice broker_name comes from argparse choices, so this is defense-in-depth.

**Fix:** allowlist:
```python
_ALLOWED = {"paper", "alpaca", "alpaca-paper", "alpaca_paper", "ibkr", "unknown"}
def _safe_broker(broker_name):
    if broker_name in _ALLOWED:
        return broker_name.replace("-", "_")
    raise ValueError(f"Unknown broker_name {broker_name!r}; expected one of {_ALLOWED}")
```

Test: `test_unknown_broker_rejected`.

### STALE-1 — `gate_b_thresholds.json` has no max-age check

**Where:** `task_quality_floor.py::_gate_b_conformal_tau`

**The issue:** an artifact written 6 months ago and never refreshed will still be used silently. Regime distribution shifts; stale τ is potentially worse than the static config default.

**Fix:** read `data["fitted_at"]`, compare to today, return None if older than `gate_b.conformal_max_age_days` (default 7). Log warning when artifact is stale.

Test: `test_returns_none_when_artifact_stale`.

---

## MEDIUM — defer to follow-up session

### MED-1 — Migration warning is noisy

`adapters/runner.py` logs the legacy-fallback warning on EVERY run if `live_state.json` still exists alongside `live_state.alpaca.json`. The migration should auto-rename legacy → archive after one successful broker-tagged write so the warning self-clears.

### MED-2 — No telemetry counter for drift events

When drift fires (warning or error path), nothing increments a counter. Operator has to grep logs to detect regressions. Should write to `runs.db::pipeline_runs::n_drift_events` or similar.

### MED-3 — `compare_ablation_emb.py` & `ab_bypass_ticker_gate.py` are one-shot research scripts

These are clearly NOT production code. Should be moved to `archive/scripts/` or similar to keep `scripts/` tight.

### MED-4 — `backup_state.sh` SQLite atomic-backup ignores errors

`sqlite3 ... ".backup ..."` may silently fail (locked db, missing source). Bash `|| true` swallows. Should explicitly check exit code.

### MED-5 — `train_asset_embeddings.py` LocalStore fallback comparison shadowing

```python
fallback_store = LocalStore(data_dir=REPO_ROOT / "data" / "ohlcv")
...
if (df is None or df.empty) and fallback_store.data_dir != primary_store.data_dir:
```

The fallback is constructed unconditionally even when primary covers all tickers. Two file-handle ctors per run when one would do. Trivial perf, but messy.

---

## LOW — style & DRY

### STY-1 — Drift detector duplicates n_total/n_missing/pct_miss formatting twice (error + warning paths). Use a single message template.

### STY-2 — Repeated import inside loops in `_gate_b_conformal_tau` (Path, json). Module-level imports preferred.

### STY-3 — `BaseBroker.broker_name = "unknown"` then `PaperBroker.broker_name = "paper"` mixes class-attribute style with `AlpacaBroker.broker_name` as `@property`. Pick one (property everywhere).

---

## Tracker

| ID | Severity | Location | Status |
|---|---|---|---|
| CRIT-1 | HARD | job_panel_scoring.py | TBD this commit |
| STUB-1 | HARD | train_horizon_blender.py | TBD this commit |
| TEST-1 | HARD | tests/test_state_paths.py | TBD this commit |
| TEST-2 | HARD | tests/test_panel_scoring_drift.py | TBD this commit |
| TEST-3 | HARD | tests/test_quality_floor_conformal.py | TBD this commit |
| VAL-1 | HARD | state_paths.py | TBD this commit |
| STALE-1 | HARD | task_quality_floor.py | TBD this commit |
| MED-1..5 | MED | various | follow-up |
| STY-1..3 | LOW | various | follow-up |

7 hard items in this commit; 5 medium + 3 style deferred.
