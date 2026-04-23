# Golden Config — 2026-04-23

**Frozen snapshot:** `backtesting/renquant_104/strategy_config.golden.json`
**Code HEAD at freeze:** `e586018`  (post drawdown-reset fix)
**Live config file:**   `backtesting/renquant_104/strategy_config.json`

This is the first config that restores renquant_104 APY above the user's
~20% bar after the 2026-04-22 regression. Keep it as the known-good baseline.
If a later change drops portfolio APY below 20% on the same 27-month OOS
window, revert to this snapshot and bisect.

---

## Measured performance

Full 27-month out-of-sample sim (`sim.runner.run_backtest` via `SimAdapter +
InferencePipeline`), backtest window 2024-01-01 → 2026-03-26, starting
cash $100k, panel-LTR + NGBoost enabled:

| Metric | Value |
|---|---|
| Final value | **$188,693** |
| Total return | **+88.7%** |
| APY | **+33.1%** |
| Buys / sells | 121 / 115 |
| Win rate | **72%** |
| Avg hold | 37 days |
| Avg P&L per trade | +9.6% |
| Total tax paid | $75,954 |
| Longest no-trade streak | **22 days**  (down from 153 d pre-fix) |
| First trade date | 2024-01-02 |
| Rotations executed | 1 (COST → SHOP) |

Exit reasons: `{model_sell: 96, stop_loss: 9, max_hold: 7, rotation: 1, trailing_stop: 1, single_day_loss: 1}`

---

## Recovery progression (why this config is golden)

Each row is one commit on top of the prior.

| Commit | Fix | Total | APY | Streak |
|---|---|---|---|---|
| (base before fixes) | buggy regression state | +5.4% | 2.4% | 153 d |
| `2df4e21` | `_eval_sharpe` prefers tournament `sharpe` | +17.0% | 7.3% | — |
| `2df4e21` | `universe_floor.threshold` 0.5 → 1.0 | +24.0% | 10.1% | — |
| `33c0e9b` | `confidence_veto_threshold` 0.30 → 0.0 | +25.9% | 10.9% | 153 d |
| **`e586018`** | **`DrawdownCircuitTask` resets `skip_buys` on recovery** | **+88.7%** | **33.1%** | **22 d** |

The drawdown fix was the dominant lever. A 35%+ drawdown event early in the
backtest tripped `ctx.skip_buys=True`, which the sim adapter persisted across
bars. `DrawdownCircuitTask` only SET the flag, never cleared it — buys were
blocked for the rest of the run even after the portfolio recovered. Now the
flag is recomputed each bar from the current drawdown.

---

## Key config fields (what matters in this golden state)

```json
{
  "sharpe_floor":                1.0,
  "regime": {
    "confidence_veto_threshold": 0.0     // disabled (GMM posterior caps ~0.25)
  },
  "defensive_tickers": ["GLD", "TLT", "XLV", "XLU"],
  "ranking": {
    "panel_scoring": {
      "enabled":             true,
      "bypass_ticker_gate":  false,      // tournament gate kept (tested: off = worse)
      "global_calibration":  {"enabled": true},
      "ngboost": {
        "enabled":     true,
        "score_mode":  "additive",       // see footnote 1
        "lambda_sigma": 0.0
      }
    },
    "universe_floor": {
      "type":      "sharpe",
      "threshold": 1.0                   // raised from 0.5 (matches CLAUDE.md policy)
    },
    "tournament": {"winner_metric": "sharpe"}
  },
  "panel_ltr": {
    "backend":              "xgboost",   // lightgbm infra shipped but not benchmarked
    "lookahead_days":       10,
    "cv_method":            "cpcv",
    "num_boost_round":      150,
    "xgb_params": {
      "eta": 0.02, "max_depth": 2, "min_child_weight": 100,
      "subsample": 0.4, "colsample_bytree": 0.4,
      "lambda": 10.0, "alpha": 5.0
    },
    "training_window_years": 5.0,
    "recency_weighting":     {"kind": "exp_decay", "half_life_days": 252},
    "fundamentals":          {"enabled": true},
    "earnings_surprise":     {"enabled": true},
    "insider_trades":        {"enabled": true}
  },
  "rotation": {"enabled": true, "min_expected_advantage_pct": 0.03},
  "training": {"cadence": "custom", "allowed_weekdays": [1, 3, 6]},
  "monitoring": {"max_no_trade_days": 15, "max_no_candidate_days": 15}
}
```

Footnote 1: `ngboost.score_mode: "additive"` (not `mu_minus_lambda_sigma`)
so `ApplyGlobalCalibrationTask` actually runs. With `mu_minus_lambda_sigma`,
the calibration task short-circuits and the raw μ−λσ values (~[-0.05, +0.05])
fall below the 0.10 tier threshold → zero trades. A future fix could re-enable
μ−λσ mode *and* have global calibration run on the combined signal; that
requires fitting the calibrator on μ−λσ rather than raw panel scores.

---

## Universe admission at this config

With `sharpe_floor=1.0` tournament metric: **36 / 52** per-ticker models
admitted. Defensives (GLD/TLT/XLV/XLU) are always admitted regardless of
floor, per `FilterUniverseFloorTask` policy.

---

## How to restore if future work regresses APY

```bash
# from repo root
cp backtesting/renquant_104/strategy_config.golden.json \
   backtesting/renquant_104/strategy_config.json

# verify
~/miniconda3/envs/renquant/bin/python /tmp/run_baseline_sim.py
# expect: total_return ≈ +88.7%  apy ≈ +33.1%  longest_no_trade_streak ≈ 22
```

Sim reproducibility needs the panel + NGBoost artifacts matching the
`trained_date: 2026-04-22` retrain. If models get retrained later and the
sim diverges, a training-run audit row from `data/runs.db` or
`logs/training/{date}.jsonl` should show the config deltas.

---

## Related commits

- `2df4e21` — universe_floor: prefer tournament sharpe + floor 1.0
- `33c0e9b` — regime: disable ConfidenceVeto (GMM posterior cap)
- `e586018` — drawdown: reset `skip_buys` on recovery (THE fix)

## Regression tests guarding this state

- `tests/test_universe_alignment.py::TestSharpeEvaluatorPrefersTournament` (4)
- `tests/test_pipeline.py::TestDrawdownCircuitTaskResets` (5)

Any future touch of `_eval_sharpe` or `DrawdownCircuitTask` that breaks
these tests would re-introduce the regression.
