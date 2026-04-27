# Session Handoff — 2026-04-23

End state for the next session. Short version: golden APY moved from +40.1% to **+44.20%** on the back of hourly panel features (Plan G). Two other experiments ran to completion and shelved with clean evidence.

## The headline

| | Before session | After session |
|---|---|---|
| Golden panel | 25 features, daily only | **31 features, hourly-enhanced** |
| OOS mean IC (CPCV 15-split) | +0.0411 | +0.0326 |
| After-tax APY (27-mo OOS sim) | +40.1% | **+44.20%** |
| Win rate | 77% | **82%** |
| Max no-trade streak | 22 d | 26 d |

IC went down, APY went up. Hourly features add signal that averaged-IC doesn't capture. Full diagnosis in `doc/panel_training_runs.md` (2026-04-23 late PT entry).

## What shipped (in order)

1. **A — HWM guard** (`ab1006d`). `resolve_hwm()` snaps a stale stored HWM to current equity. 10 regression tests. Unblocked live trading.
2. **B — LightGBM A/B** (`8d6b08a`, `67e95af`). Shelved at −12.7 APY pts. Per-row-weight bug fixed on the way in.
3. **C — σ-penalty sweep** (`0c80443`). Shelved — λ=0.25 +2 APY pts only (below +3 promotion floor).
4. **D — Sustainability watch** (`67e95af`). JSONL audit log + `scripts/weekly_apy_check.py` + launchd plist (Sun 12 PT).
5. **G — hourly features** — the win. Staged as 4 commits:
   - `8c65537` — aggregator + 17 tests
   - `f03d1eb` — wire into `PanelDataJob` + 8 wiring tests
   - `0c80443` — fetcher script
   - `3b1d2e2` — inference-path wiring fix (`prepare_inference_panel_frames` + `fit_panel_calibrator.py`)
   - `e65b081` — A/B verdict + promote to golden
6. **F — regime-conditional calibration** (`26c40ae`, `7f68a40`). Shelved at −3.78 APY pts. In-sample per-regime IC was 3.5×–17× pooled but didn't survive live. Infra kept behind off-by-default flag.
7. **H — transformer on hourly panel** (`c9ee50b`). Shelved at 0.20× XGBoost (was 0.49× daily-only). Panel needs > 200k rows to revisit.
8. **Env doc** (`c9ee50b`). `requirements.lock.txt` (310 packages) + `doc/environment.md`.
9. **Test fix** (`d857195`). Stale `test_mu_minus_lambda_sigma_defers_to_ngboost` updated to match post-reorder semantics.

## What you can pick up next

From `doc/improvement_roadmap.md`, in order:

1. **I — accumulate 4 weeks of live sustainability data** (passive — just wait for Sunday alerts).
2. **J — hourly feature pruning** (1 day). Drop the 3 weakest hourly cols (|IC| < 0.016), retrain, A/B. Target: lift IC +0.005 without hurting APY.
3. **K — CHOPPY regime diagnosis** (1 day). F's fit showed IC = −0.116 in CHOPPY. Understand whether the inversion is universal / ticker-scoped / feature-scoped — possibly derive a CHOPPY-specific ScoreBuyTask tier offset.
4. **L — per-ticker hourly effectiveness** (1 day). Which tickers carry the +4.18 APY gain from hourly features?
5. **M, N** — small cleanups (schema fix + doc consolidation).

## Sanity checks to run after any G-touching change

- `python -m pytest tests/test_hourly_features.py tests/test_panel_hourly_wiring.py -v` (25 tests)
- `python scripts/fetch_hourly_bars.py --strategy renquant_104 --dry-run` — should list 44 symbols × 9 chunks
- Spot-check `panel-ltr.json` has 31 feature cols and includes `intraday_realized_vol_z`

## Rollback instructions

If hourly golden regresses in live:

```bash
cd backtesting/renquant_104/artifacts
cp panel-ltr.golden-daily.json panel-ltr.json
cp ngboost-head.golden-daily.json ngboost-head.json
```

And flip `panel_ltr.hourly.enabled: false` in `strategy_config.json`. Daily-only golden is preserved.

## Machine envelope (from `doc/environment.md`)

M4 Pro, 48 GB RAM, 14 cores, 665 GB free. Python 3.10.20, XGBoost 3.2, NGBoost 0.5.10, PyTorch 2.11 (MPS with `PYTORCH_ENABLE_MPS_FALLBACK=1` for transformer). Docker 29.3.1, LEAN 1.0.225. Everything below 2.5 GB including `.git/`.

Any new session can reproduce end-to-end:

```bash
conda activate renquant
pip install -r requirements.lock.txt  # 310 packages pinned
python -m pytest tests/                 # ~1060 tests
python scripts/train_104.py --skip-baseline --skip-recalibrate --force  # panel-only retrain
```
