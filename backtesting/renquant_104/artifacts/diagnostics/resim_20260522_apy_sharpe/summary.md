# renquant_104 XGB vs PatchTST clean resim (2026-05-22)

Window: 2024-07-02 to 2026-02-10, 404 trading days.

Purpose: isolate model scorer behavior from shadow/NGBoost overlays after fixing
the sim leakage guard, concurrent OHLCV cache writes, and PatchTST holdings
scoring. Both sims ran with `--no-persist --skip-preflight --no-compare`.

## Result

| Model | Config | Total return | APY | Sharpe | MaxDD | Beta vs SPY | Ann. alpha vs SPY | Trades | Closed win rate | Avg closed hold |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| XGB strict-cutoff | `strategy_config.sim_xgb_truly_oos_pure_20260522.json` | +1.88% | +1.17% | +0.20 | 8.90% | 0.101 | -0.19%/yr | 100 buys / 83 sells | 61.98% | 49.1d |
| PatchTST clean diagnostic | `strategy_config.sim_patchtst_clean_20260522.json` | +2.40% | +1.49% | +0.23 | 7.39% | 0.109 | +0.03%/yr | 147 buys / 139 sells | 58.16% | 36.0d |
| SPY buy-and-hold | same window | +26.07% | +15.59% | +0.91 | 19.00% | 1.000 | n/a | n/a | n/a | n/a |

## Tax lens

| Model | Closed gross P&L | Event-level tax debited | Event-tax net P&L | Annual-net tax estimate | Annual-net APY estimate | Gross no-tax APY approximation |
|---|---:|---:|---:|---:|---:|---:|
| XGB strict-cutoff | +$12,371.67 | $10,964.03 | +$1,407.64 | $6,185.83 | +4.17% | +7.90% |
| PatchTST clean diagnostic | +$18,300.26 | $15,627.95 | +$2,672.31 | $9,150.13 | +6.07% | +11.51% |

The old "gross < tax" pathology is not present in this run: closed gross P&L is
positive and larger than event-level tax. Event-level tax is still extremely
punitive because nearly all realized gains are short-term; annual-net reporting
is materially less punitive but still leaves both strategies far behind SPY.

## Interpretation

Both scorers are positive after the structural fixes, but neither converts a
positive cross-sectional signal into attractive portfolio performance. The
portfolio has very low market beta (~0.10), near-zero annual alpha vs SPY, and
negative information ratio vs SPY. In other words, the execution stack is not
leveraging the bull-market opportunity; it is producing a low-volatility,
high-turnover, tax-heavy equity curve.

PatchTST is slightly better than XGB on APY/Sharpe and annual-net APY here, but
this specific PatchTST result is diagnostic only: the artifact lacks a
point-in-time trained-date/cutoff manifest, so it should not be promoted without
a strict walk-forward artifact and manifest.

## Files

- XGB equity/report/trades: `xgb_pure/`
- PatchTST equity/report/trades: `patchtst_clean/`
- XGB log: `logs/codex_resim_20260522/xgb_pure.log`
- PatchTST log: `logs/codex_resim_20260522/patchtst_clean.log`
