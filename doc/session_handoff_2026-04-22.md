# Session handoff — 2026-04-22 (23:00 PT)

Restart-safe state dump. Read top-to-bottom.

---

## 1. Current HEAD: `14f80e2`

Pushed, synced with `origin/main`. 988 tests pass. 2 opt-in full-sim invariants gated behind `RENQUANT_FULL_SIM=1`.

---

## 2. THE REGRESSION (highest priority to fix)

**User observation**: renquant_104 APY dropped from ~20% "this morning" to ~2% after today's work.

**Confirmed by foreground sim (27-month OOS)**:
- Baseline (panel OFF): `+2.8%, 1.3% APY, 63 trades`
- Panel-LTR:            `+3.7%, 1.7% APY, 59 trades`
  (versus user's morning state around 14-16% APY baseline / 11-12% panel)

**Root causes (strongest first)**:

### a) Config drift — my defensive fix never landed in any commit

Every commit today has:
```json
"defensive_tickers":           ["GLD", "XLU"]       // should be 4
"regime.confidence_veto_threshold": 0.55            // should be 0.30
```

I ran `python3 -c "..."` twice to change these; output confirmed the disk write; but later commits show the file reverted. **Re-applied at 23:xx PT** — disk now has:
```json
"defensive_tickers":           ["GLD", "TLT", "XLV", "XLU"]
"regime.confidence_veto_threshold": 0.30
```

NOT yet committed — **commit this before anything else**.

This means the no-trade bug fix I claimed (6-month slice showed 12d longest streak, user saw 21% APY) was correct in memory but NEVER SHIPPED. Subsequent retrains have been running with the broken `[GLD, XLU]` / `veto=0.55` config. Systemic no-trade periods would have returned.

**Suspect culprit for the revert**: likely `scripts/recalibrate_scores.py` or `scripts/train_104.py --skip-recalibrate` overwriting the file. Need to audit. Candidates:
- `scripts/recalibrate_scores.py:275` does `config_path.write_text(json.dumps(config, indent=2))` with in-memory `config` that was loaded at start. If my edit happened between its read and write, it would revert.
- But I passed `--skip-recalibrate` — check if train_104 still writes anyway.

### b) Panel hyperparameter + label-horizon changes

Changes committed today that likely hurt returns:

| Config key | This morning | Now | Likely impact |
|---|---|---|---|
| `panel_ltr.lookahead_days` | 5 | **10** | Different label horizon → different score semantics |
| `panel_ltr.beta_window` | 60 | 252 | Slower beta adaptation |
| `panel_ltr.num_boost_round` | 400 | 150 | Fewer boosting rounds |
| `panel_ltr.xgb_params` | `{}` (defaults) | `max_depth:2, min_child_weight:100, lambda:10, alpha:5` | Much tighter trees — may underfit |
| `panel_ltr.training_window_years` | unset | **5.0** | Panel now sees last 5yr only; training data ~halved |
| `panel_ltr.recency_weighting` | unset | `exp_decay, hl=252` | Newer samples weighted ~2x older |
| `ranking.panel_scoring.ngboost.score_mode` | `mu_minus_lambda_sigma` | **`additive`** | **MAJOR**: ranking path changed; μ−λσ no longer in rank_score |
| `ranking.panel_scoring.ngboost.lambda_sigma` | 1.0 | **0.0** | σ-penalty disabled |
| `ranking.universe_floor` | (none) | `sharpe: 0.5` | Filters ~5 tickers from universe |
| `rotation.enabled` | false | **true** | Swaps held positions |

**The biggest single regression suspect is the ngboost score_mode flip**. The old `mu_minus_lambda_sigma` mode had NGBoost compute `rank_score = μ − λ·σ` which is a volatility-aware composite. The new `additive` mode leaves rank_score as the raw panel output (then calibrated). These produce very different portfolio selections.

**Why I flipped it**: in the current code, `mu_minus_lambda_sigma` mode makes `ApplyGlobalCalibrationTask` skip — so `rank_score` is raw μ−λσ ∈ [-0.06, +0.04], below the 0.10 tier threshold → zero trades. My "fix" was to switch to `additive` mode so global calibration runs. But this sacrifices the σ-aware ranking.

**Better fix** (not yet done): keep `mu_minus_lambda_sigma` mode AND have `ApplyGlobalCalibrationTask` run regardless, calibrating the `μ−λσ` composite. This preserves σ-awareness while giving the tier-threshold a calibrated score.

### c) Panel artifact never retrained with 5yr window

`artifacts/panel-ltr.json` has `panel_shape.dates: 2247` — that's the FULL 9-year history. My 5-year window retrain (background `b1w42o6m0`) reported exit 0 but didn't update the artifact. Either the config wasn't read by that process, or something failed silently.

---

## 3. Pending tasks (what to pick up)

### PRIORITY 1 — Restore to a known-good config
1. `git diff backtesting/renquant_104/strategy_config.json` — see the current defensive/veto changes that are uncommitted
2. Commit them: `git add ... && git commit -m "fix(config): restore defensive_tickers + confidence_veto" && git push`
3. Audit which script reverted the earlier edit (see §2a culprits) and patch it
4. Run `python -m pytest tests/ --ignore=tests/test_no_trade_invariant.py` to confirm no regressions — should be ~988 passing

### PRIORITY 2 — Identify which hyperparameter change broke performance
Roll back changes one at a time and run the full-OOS sim to measure impact:
1. Revert `ngboost.score_mode` to `mu_minus_lambda_sigma` + `lambda_sigma: 1.0` — BUT fix the calibration bug first (see §2b): modify `ApplyGlobalCalibrationTask` in `backtesting/renquant_104/kernel/panel_pipeline/job_panel_scoring.py` to calibrate the μ−λσ output too (currently skipped).
2. Revert `lookahead_days` to 5 (label horizon).
3. Revert `xgb_params` to `{}` (default depth/weight).
4. Measure each in isolation via a full-OOS sim (`~/miniconda3/envs/renquant/bin/python -c "..."` with `load_strategy_config` + `run_backtest`).

Commit tip: **keep a known-good-config branch** so we can A/B faster.

### PRIORITY 3 — Transformer (was next major item, paused)
Design landed in `doc/renquant_104_transformer_design.md`. User's answers to my questions:
- **Deterministic + reproducible** (Q1 → (a))
- **MPS** (Q4's device)
- **Ensemble first, then ratio > 1.10 → replace** (Q2)
- **Weekly via Sunday retrain** (Q3 → (a))
- **Train both on last 5yr + exp_decay** (Q4)
- **Early-stopping patience=6** (Q5)
- **Anomaly retrain directly replaces Sunday model** (Q6)
- **VIX via yfinance `^VIX`** (Q7)

Implementation NOT started. Task #23 is pending.

### PRIORITY 4 — Anomaly-triggered retrain (Task #24)
`scripts/check_retrain_triggers.py` + `scripts/conditional_retrain_104.sh` + plist at 13:10 PT. SPY |daily change| > 2% OR VIX |daily change| > 5% → `train_104.py --force --trigger=anomaly_spy_2pct` (or vix). Not started.

### PRIORITY 5 — Panel-transformer backend (Task #23)
See `doc/renquant_104_transformer_design.md` for full spec.

---

## 4. What landed today that's real + useful (don't accidentally revert)

These are keepers. Don't roll back:
- **LoadUniverseJob** + configurable `ranking.universe_floor` (types: none/sharpe/ic). `FilterUniverseFloorTask` always exempts `defensive_tickers`.
- **MonitorIdleStreakTask** + SimResult streak fields + opt-in invariant test. Catches systemic no-trade periods.
- **Round 3 orthogonal factors**: Amihud, volume_shift, price_to_high, realized_vol, drawdown_peak. With tests.
- **Round 4 short_pct_float** from yfinance. With tests.
- **Round 5 earnings_surprise** pipeline. With tests.
- **Round 5 SEC Form 4 insider trades** pipeline (executive-only, P/S codes). Cache at `data/insider_trades/`. With tests.
- **Training-run audit** (SQLite + JSONL at `logs/training/{date}.jsonl`). Auto-captures every retrain with elapsed time, device, deterministic flag, panel shape, trigger.
- **Panel IC bumped 0.038 → 0.064** (CPCV 15-split).
- **Notebook fast-path cell** (skip heavy training when on-disk artifacts are < 7 days old). Flags `FORCE_TOURNAMENT/PANEL/RECALIB` to force.
- **E2E test framework** (`tests/test_e2e_execution.py`) — opt-in via `RENQUANT_E2E_NOTEBOOK=1` / `RENQUANT_E2E_LEAN=1`.
- **`CLAUDE.md` rule 2**: bug → fix → regression test in the same commit.

Tests total: **988 passing** + 2 skipped + 4 opt-in.

---

## 5. Background processes — verify none are stuck

Before restart:
```bash
ps aux | grep -E "train_104|run_backtest|fetch_insider|nbconvert" | grep -v grep
# Kill any stragglers:
kill -9 $(ps aux | grep -E "train_104|run_backtest|fetch_insider|nbconvert" | grep -v grep | awk '{print $2}') 2>/dev/null
```

---

## 6. Uncommitted changes on disk (to stage + commit after restart)

```bash
git status -s
# Expect:
# M backtesting/renquant_104/strategy_config.json   <-- defensive + veto fix
```

Suggested commit message:
```
fix(config): restore defensive_tickers + confidence_veto_threshold

My earlier edits to these two fields (in commit 3c366b6) got reverted
by a downstream script and never actually shipped. Re-applied.

  defensive_tickers:           [GLD, XLU]           -> [GLD, TLT, XLV, XLU]
  regime.confidence_veto_threshold: 0.55            -> 0.30

Expected impact: resolves the persistent no-trade streaks that returned
after the revert; APY should recover toward the 20% user saw earlier today.

Follow-up: audit which script overwrites strategy_config.json and guard
against the revert pattern. Likely suspect: recalibrate_scores.py's
read-edit-write pattern racing with concurrent edits.
```

---

## 7. Quick-start for the next session

```bash
cd /Users/renhao/git/github/RenQuant
git pull
git status   # confirm the one uncommitted strategy_config.json change

# Sanity check tests
~/miniconda3/envs/renquant/bin/python -m pytest tests/ --ignore=tests/test_no_trade_invariant.py -q

# Full-OOS sim to measure current state (takes ~3-5 min):
~/miniconda3/envs/renquant/bin/python <<'PY'
import sys; sys.path.insert(0, 'backtesting/renquant_104')
from pathlib import Path
from kernel.config import load_strategy_config
from kernel.data import fetch_ohlcv
from training_panel.pipeline import prepare_inference_panel_frames
from sim.runner import run_backtest

STRATEGY_DIR = Path('backtesting/renquant_104')
CFG = load_strategy_config(STRATEGY_DIR / 'strategy_config.json')
CFG['_strategy_dir'] = str(STRATEGY_DIR)
CFG.setdefault('initial_cash', 100_000)
symbols = set(CFG['watchlist']) | set(CFG.get('sector_etf_map', {}).values()) | {'SPY'}
ohlcv = {s: fetch_ohlcv(s) for s in symbols}
ohlcv = {s: df for s, df in ohlcv.items() if df is not None and not df.empty}

# Panel ON
ff, fac = prepare_inference_panel_frames(
    watchlist=CFG['watchlist'], ohlcv=ohlcv,
    ticker_sectors={t: CFG['sector_map'][t] for t in CFG['watchlist'] if t in CFG.get('sector_map', {})},
    config={**CFG, '_strategy_dir': str(STRATEGY_DIR)})
r = run_backtest(config=CFG, strategy_dir=STRATEGY_DIR, ohlcv=ohlcv, spy_df=ohlcv['SPY'],
    sector_etf_map=CFG.get('sector_etf_map', {}),
    panel_feature_frames=ff, panel_factor_frames=fac)
r.print_summary()
PY
```

If APY is still ~2% after the defensive/veto fix, the hyperparameter changes (§2b) are the remaining culprit — roll those back next, one at a time, measuring APY after each.

---

## 8. Honest self-assessment

- Claimed a fix that never shipped (defensive/veto). The 6-month slice APY=21.2% I reported was genuine at that moment but in-memory-only; the state got lost.
- Too much parallel work today → changes didn't get measured individually → can't tell which one tanked performance. Should have A/B'd each change against a pinned baseline.
- Should have caught "0.064 OOS IC but 1.7% sim APY" as inconsistent earlier and diagnosed it before landing more changes.
- User is right to be angry. Infrastructure improvements (MonitorIdleStreakTask, LoadUniverseJob, SEC pipeline, training audit) are real, but net portfolio performance regressed, which is what actually matters.
